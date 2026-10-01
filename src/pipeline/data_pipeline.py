"""Notebook-facing canonical customer-month pipeline (historical train data).

Delegate versioning and RUN/SKIP decisions to the existing checkpoint store;
adapt existing preprocessing and feature-store functions into full snapshots.
Competition test has no observed product states and is not this dataset.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Sequence

import duckdb

from src.data.gcs_storage import download_object_if_missing, object_name
from src.features.checkpoint_store import (
    KEY_COLUMNS, CheckpointStore, ProcessingStep, run_processing_pipeline,
)
from src.features.customer_history import HISTORY_FEATURE_NAMES, quote
from src.features.feature_store import acquisition_feature_names, ensure_customer_month_feature_store
from src.features.persona import PERSONA_FEATURES, RAW_PERSONA_COLUMNS
from src.ingestion.checkpoint import build_interim_checkpoint
from src.preprocessing.customer_profile import preprocess_customer_profile_baseline
from src.products import PRODUCT_COLUMNS


PREPROCESSING_MARKER = "profile_preprocessed"
PROFILE_COLUMNS = (*RAW_PERSONA_COLUMNS, "conyuemp", "nomprov")
ENGINEERED_FEATURES = tuple(dict.fromkeys((
    *PERSONA_FEATURES, *HISTORY_FEATURE_NAMES,
    *("prev_" + name for name in PRODUCT_COLUMNS),
    *acquisition_feature_names(PRODUCT_COLUMNS),
)))


def run_data_pipeline(
    *,
    required_features: Sequence[str] = (),
    force_process: bool = False,
    publish_checkpoint: bool = True,
    data_root: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
    work_root: str | Path | None = None,
    raw_filename: str = "train_ver2.csv",
    download_raw: bool = True,
    chunksize: int = 150_000,
    show_progress: bool = True,
    memory_limit: str = "4GB",
    temp_directory: str | Path | None = None,
    renta_clip_upper_quantile: float = 0.99,
) -> dict[str, Any]:
    """Resolve a retained snapshot or compute and optionally publish a candidate.

    Empty requirements reuse the latest valid snapshot. ``force_process``
    bypasses reuse and recomputes on new files. Cleanup remains an explicit
    infrastructure operation; this entrypoint never deletes retained versions.
    """
    root = Path(data_root or os.getenv("SANTANDER_DATA_ROOT", "data"))
    store = CheckpointStore(checkpoint_root or root / "processed/canonical_customer_month")
    required = tuple(dict.fromkeys(required_features))
    resolved = store.resolve(required, required_artifacts=("train",))
    if resolved is not None and not force_process:
        path = store.root / resolved["artifacts"]["train"]["path"]
        print(f"[checkpoint] REUSE version={resolved['version']} | final validation=PASS")
        return {"status": "reused", "customer_month": path, "version": resolved,
                "processing": [], "published": False}

    parent = resolved or store.resolve([], required_artifacts=("train",))
    work = Path(work_root or root / ".pipeline_work")
    work.mkdir(parents=True, exist_ok=True)
    # Unique run directories also prevent accidental reuse of stale sidecars.
    run_dir = Path(mkdtemp(prefix="customer_month_", dir=work))
    spill = Path(temp_directory or os.getenv("SANTANDER_DUCKDB_TEMP_DIRECTORY", str(work / "duckdb_temp")))
    spill.mkdir(parents=True, exist_ok=True)

    def connect() -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(database=":memory:")
        con.execute("SET memory_limit = ?", [memory_limit])
        con.execute("SET temp_directory = ?", [str(spill)])
        return con

    def schema(path: Path) -> dict[str, str]:
        with connect() as con:
            return dict((row[0], row[1]) for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
            ).fetchall())

    def validate(artifacts: dict[str, Path]) -> None:
        for path in artifacts.values():
            columns = schema(path)
            if missing := set(KEY_COLUMNS).difference(columns):
                raise ValueError(f"Missing customer-month keys: {sorted(missing)}")
            with connect() as con:
                valid = con.execute(
                    "SELECT COUNT(*) = COUNT(DISTINCT (ncodpers, fecha_dato)) "
                    "AND COUNT(*) = COUNT(ncodpers) AND COUNT(*) = COUNT(fecha_dato) "
                    "FROM read_parquet(?)", [str(path)],
                ).fetchone()[0]
                if not valid:
                    raise ValueError(f"Non-null unique customer-month grain violated: {path}")
                counts = (
                    "rfm_frequency", "rfm_monetary", "rfm_recency_months",
                    "rfm_observed_history_records", "record_gap_months",
                    "acquisitions_last_1m", "acquisitions_last_3m",
                    "acquisitions_last_6m", "cumulative_drops",
                )
                binary = (
                    *PRODUCT_COLUMNS, *("prev_" + name for name in PRODUCT_COLUMNS),
                    *acquisition_feature_names(PRODUCT_COLUMNS),
                    "rfm_has_previous_record", "rfm_never_acquired_before",
                    "is_new_customer", "is_active_customer", "is_male",
                    "is_deceased", "is_domestic", "profile_missing_structural",
                )
                conditions = [f"{quote(name)} < 0" for name in counts if name in columns]
                conditions.extend(f"{quote(name)} NOT IN (0, 1)" for name in binary if name in columns)
                if PREPROCESSING_MARKER in columns:
                    conditions.append(f"{quote(PREPROCESSING_MARKER)} IS DISTINCT FROM TRUE")
                if conditions and con.execute(
                    f"SELECT COUNT(*) FROM {parquet(path)} WHERE {' OR '.join(conditions)}"
                ).fetchone()[0]:
                    raise ValueError(f"Feature-specific validation failed: {path}")

    def copy_query(query: str, destination: Path) -> Path:
        escaped = str(destination).replace("'", "''")
        with connect() as con:
            con.execute(f"COPY ({query}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        return destination

    def parquet(path: Path) -> str:
        return "read_parquet('" + str(path).replace("'", "''") + "')"

    def merge(source: Path, result: Path, destination: Path, outputs: Sequence[str]) -> Path:
        validate({"source": source, "result": result})
        with connect() as con:
            different_keys = con.execute(
                f"SELECT COUNT(*) FROM {parquet(source)} s FULL OUTER JOIN "
                f"{parquet(result)} r USING (ncodpers, fecha_dato) "
                "WHERE s.ncodpers IS NULL OR r.ncodpers IS NULL"
            ).fetchone()[0]
        if different_keys:
            raise ValueError("Processing changed the customer-month key set.")
        replaced = set(outputs)
        projection = ["s." + quote(name) for name in schema(source) if name not in replaced]
        projection.extend("r." + quote(name) for name in outputs)
        return copy_query(
            f"SELECT {', '.join(projection)} FROM {parquet(source)} s "
            f"JOIN {parquet(result)} r USING (ncodpers, fecha_dato)", destination,
        )

    def ingest() -> Path:
        raw, interim = root / "raw" / raw_filename, root / "interim/train.parquet"
        if not interim.is_file():
            if not raw.is_file() and download_raw:
                bucket = os.getenv("GCS_BUCKET")
                if not bucket:
                    raise ValueError("Set GCS_BUCKET or place the raw CSV under data/raw.")
                download_object_if_missing(
                    bucket, object_name(os.getenv("GCS_RAW_PREFIX", "raw"), raw_filename),
                    raw, os.getenv("GOOGLE_CLOUD_PROJECT"),
                )
            build_interim_checkpoint(raw, interim, chunksize=chunksize, show_progress=show_progress)
        validate({"train": interim})
        return interim

    working = {"train": store.root / parent["artifacts"]["train"]["path"]} if parent else {"train": ingest()}
    profile_outputs = tuple(name for name in PROFILE_COLUMNS if name in schema(working["train"]))

    def preprocess(current: dict[str, Path]) -> dict[str, Path]:
        print("[processing] RUN customer_profile_preprocessing")
        # Fit from upstream values rather than already imputed/clipped columns.
        raw_source = ingest()
        result = preprocess_customer_profile_baseline(
            raw_source, None, run_dir / "profile", force_process=True,
            memory_limit=memory_limit, temp_directory=spill,
            renta_clip_upper_quantile=renta_clip_upper_quantile,
        )
        cleaned = copy_query(
            f"SELECT *, TRUE AS {quote(PREPROCESSING_MARKER)} "
            f"FROM {parquet(Path(result['train']))}", run_dir / "profile_marked.parquet",
        )
        return {"train": merge(current["train"], cleaned, run_dir / "preprocessed.parquet",
                               (*profile_outputs, PREPROCESSING_MARKER))}

    def engineer(current: dict[str, Path]) -> dict[str, Path]:
        print("[processing] RUN customer_month_features")
        features = ensure_customer_month_feature_store(
            current["train"], run_dir / "features.parquet", force_process=force_process,
            memory_limit=memory_limit, temp_directory=spill,
        )
        return {"train": merge(current["train"], features, run_dir / "candidate.parquet", ENGINEERED_FEATURES)}

    steps = [
        ProcessingStep("customer_profile_preprocessing", (*profile_outputs, PREPROCESSING_MARKER),
                       (*KEY_COLUMNS, "age", "antiguedad", "renta"), preprocess, validate),
        ProcessingStep("customer_month_features", ENGINEERED_FEATURES,
                       (*KEY_COLUMNS, *RAW_PERSONA_COLUMNS, *PRODUCT_COLUMNS), engineer, validate),
    ]
    candidates, processing = run_processing_pipeline(
        working, steps, force_process=force_process, required_features=required,
    )
    validate(candidates)
    for summary in processing:
        print(json_summary(summary))
    print(f"[pipeline] required_features={list(required)} | final validation=PASS | publish_checkpoint={publish_checkpoint}")
    configuration = {
        "required_features": list(required), "force_process": force_process,
        "publish_checkpoint": publish_checkpoint, "raw_filename": raw_filename,
        "chunksize": chunksize, "memory_limit": memory_limit,
        "renta_clip_upper_quantile": renta_clip_upper_quantile,
        "preprocessing_stats": (
            json.loads((run_dir / "profile/preprocessing_stats.json").read_text(encoding="utf-8"))
            if processing[0]["decision"] == "RUN"
            else (parent or {}).get("configuration", {}).get("preprocessing_stats")
        ),
    }
    version = store.publish(
        candidates, parent_version=parent["version"] if parent else None,
        processing=processing, configuration=configuration,
        feature_definition_metadata={step.name: {"outputs": list(step.outputs)} for step in steps},
    ) if publish_checkpoint else None
    path = store.root / version["artifacts"]["train"]["path"] if version else candidates["train"]
    return {"status": "published" if version else "candidate", "customer_month": path,
            "version": version, "processing": processing, "published": version is not None}


def json_summary(summary: dict[str, Any]) -> str:
    """Keep execution decisions and declared-output validation visible."""
    return "[processing] " + json.dumps(summary, ensure_ascii=False)
