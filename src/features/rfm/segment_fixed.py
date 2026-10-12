"""Named RFM segments from the approved, fixed portfolio-depth KMeans centers.

No fitting, scaling or log transform. Each row uses its own prior-only raw RFM.
The saved discovery notebook exposes six decimal places; do not claim these
constants have the precision of the unavailable Colab model/CSV export.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FEATURE_NAME = "rfm_segment_fixed"
DEFINITION_VERSION = "portfolio_k5_notebook_6dp_v1"
RFM_COLUMNS = ("rfm_recency_months", "rfm_frequency", "rfm_monetary")
INPUT_COLUMNS = (*RFM_COLUMNS, "rfm_never_acquired_before")
SEGMENT_NAMES = ("At Risk", "Dormant", "Loyal", "Recently Engaged", "High Value")
CATEGORIES = (*SEGMENT_NAMES, "NEVER_ACQUIRED", "MISSING")
# Cluster IDs 1..5, from cell 40 (second_k=5) of the saved discovery notebook.
CENTROIDS = (
    (6.826671, 1.863885, 1.905420),
    (12.109778, 1.486835, 2.643686),
    (2.339837, 7.100343, 4.040797),
    (2.359419, 2.265528, 2.057406),
    (3.686910, 3.178257, 6.103350),
)
PROVENANCE = {
    "definition_version": DEFINITION_VERSION,
    "source_notebook": "notebooks/EDA/RFM_Segment_Discovery_KMeans_Porfolio_Deep.ipynb",
    "source_cell": 40, "source_canonical_version": 4,
    "reference_snapshot": "2016-05-28", "reference_eligible_customers": 192741,
    "source_candidate": "second_k=5", "centroid_precision": "6 decimals saved in notebook output",
    "input_features": list(RFM_COLUMNS), "monetary_proxy": "porfolio_deep",
    "transform": "none", "distance": "squared Euclidean", "tie_break": "lowest cluster ID",
    "centroids": [list(center) for center in CENTROIDS], "segment_names": list(SEGMENT_NAMES),
    "special_state_policy": "never flag=1 takes precedence; otherwise invalid/missing inputs -> MISSING",
    "categories": list(CATEGORIES),
}


def assign_fixed_rfm_segments(frame: pd.DataFrame) -> pd.Categorical:
    """Classify every input row, retaining explicit never/missing categories."""
    if missing := set(INPUT_COLUMNS).difference(frame.columns):
        raise ValueError(f"Fixed RFM segmentation missing inputs: {sorted(missing)}")
    values = frame[list(RFM_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    never = pd.to_numeric(frame["rfm_never_acquired_before"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    codes = np.full(len(frame), 6, dtype=np.int8)  # MISSING
    codes[never == 1] = 5
    eligible = (never == 0) & np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
    if eligible.any():
        distances = ((values[eligible, None, :] - np.asarray(CENTROIDS)[None, :, :]) ** 2).sum(axis=2)
        codes[eligible] = np.argmin(distances, axis=1).astype(np.int8)
    return pd.Categorical.from_codes(codes, categories=CATEGORIES, ordered=False)


def segment_distribution(path: str | Path, *, memory_limit="1GB", threads=2) -> pd.DataFrame:
    """Count customer-month records, including zero-count special categories."""
    with duckdb.connect(config={"memory_limit": memory_limit, "threads": threads}) as con:
        counts = dict(con.execute(f'SELECT "{FEATURE_NAME}", COUNT(*) FROM read_parquet(?) GROUP BY 1',
                                  [str(path)]).fetchall())
    if set(counts).difference(CATEGORIES):
        raise ValueError("Unexpected fixed segment categories")
    result = pd.DataFrame({FEATURE_NAME: CATEGORIES, "record_count": [counts.get(c, 0) for c in CATEGORIES]})
    total = int(result.record_count.sum())
    result["record_share_pct"] = result.record_count / total * 100 if total else 0.
    print(f"[rfm_segment_fixed] All customer-month records: {total:,}", flush=True)
    print(result.to_string(index=False, formatters={"record_count": "{:,.0f}".format,
                                                  "record_share_pct": "{:.4f}".format}), flush=True)
    return result


def materialize_fixed_rfm_segments(source_path, destination_path, *, force_process=False,
                                   batch_size=50_000, memory_limit="1GB", threads=2) -> Path:
    """Append a categorical feature in bounded batches; preserve other columns.

    Parquet dictionary encoding plus pandas categorical metadata preserves
    category dtype on pd.read_parquet. No rows are sampled or discarded.
    """
    source, destination = Path(source_path), Path(destination_path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parquet = pq.ParquetFile(source)
    if FEATURE_NAME in parquet.schema_arrow.names and not force_process:
        segment_distribution(source, memory_limit=memory_limit, threads=threads)
        return source
    if missing := set(INPUT_COLUMNS).difference(parquet.schema_arrow.names):
        raise ValueError(f"Fixed RFM segmentation missing inputs: {sorted(missing)}")
    fields = [field for field in parquet.schema_arrow if field.name != FEATURE_NAME]
    fields.append(pa.field(FEATURE_NAME, pa.dictionary(pa.int8(), pa.string()), nullable=False))
    metadata = dict(parquet.schema_arrow.metadata or {})
    pandas_meta = json.loads(metadata.get(b"pandas", b"{}"))
    category_meta = json.loads(pa.Table.from_pandas(pd.DataFrame({
        FEATURE_NAME: pd.Categorical([], categories=CATEGORIES)
    }), preserve_index=False).schema.metadata[b"pandas"])
    if not pandas_meta:
        pandas_meta = category_meta
    else:
        pandas_meta["columns"] = [c for c in pandas_meta["columns"] if c["field_name"] != FEATURE_NAME]
        pandas_meta["columns"].extend(category_meta["columns"])
    metadata[b"pandas"] = json.dumps(pandas_meta).encode()
    metadata[b"rfm_segment_fixed"] = json.dumps(PROVENANCE).encode()
    schema = pa.schema(fields, metadata=metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".pending")
    try:
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for batch in parquet.iter_batches(batch_size=batch_size):
                table = pa.Table.from_batches([batch])
                inputs = table.select(INPUT_COLUMNS).to_pandas()
                labels = assign_fixed_rfm_segments(inputs)
                category = pa.DictionaryArray.from_arrays(pa.array(labels.codes, type=pa.int8()), pa.array(CATEGORIES))
                if FEATURE_NAME in table.column_names:
                    table = table.drop([FEATURE_NAME])
                table = table.append_column(schema.field(FEATURE_NAME), category)
                writer.write_table(table.replace_schema_metadata(metadata))
        if pq.ParquetFile(temporary).metadata.num_rows != parquet.metadata.num_rows:
            raise AssertionError("Fixed segmentation changed row count")
        parquet.close()
        temporary.replace(destination)
    finally:
        parquet.close()
        if temporary.exists():
            temporary.unlink()
    segment_distribution(destination, memory_limit=memory_limit, threads=threads)
    return destination
