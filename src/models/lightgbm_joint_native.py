"""Disk-backed preparation and native LightGBM training for Pipeline B.

The compact panel remains one row per customer-month.  Candidate rows are
expanded only while a Parquet record batch is being written to a LightGBM CSV
input file; no full expanded pandas frame is ever created.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

try:  # ``resource`` is unavailable on Windows but present on Colab/Linux.
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None  # type: ignore[assignment]

from src.inference.predict import InputSchema, ModelArtifact, write_artifact_metadata
from src.models.lightgbm_joint import JOINT_ARTIFACT_PRODUCT
from src.products import PRODUCT_COLUMNS, PRODUCT_ID_MAP, categorical_product_id


TARGET_COLUMN = "target"


@dataclass(frozen=True)
class NativeDatasetFiles:
    compact_path: Path
    text_path: Path
    binary_path: Path
    candidate_rows: int
    positive_rows: int
    disk_bytes: int


class NativeJointBooster:
    """Pickle-friendly sklearn-style adapter around one native Booster."""

    def __init__(self, booster: Any, categorical_maps: Mapping[str, Mapping[str, int]] | None = None) -> None:
        self.booster = booster
        self.categorical_maps = {name: dict(values) for name, values in (categorical_maps or {}).items()}

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        # The native training cache deliberately uses safe positional feature
        # names (f0, f1, ...) because LightGBM rejects some source names.  Do
        # not let source DataFrame column labels leak back into prediction.
        encoded = frame.copy()
        for name, mapping in self.categorical_maps.items():
            encoded[name] = _encode_category(encoded[name], mapping)
        matrix = encoded.apply(pd.to_numeric, errors="raise").to_numpy(dtype="float32", copy=False)
        probability = np.asarray(self.booster.predict(matrix), dtype="float64")
        return np.column_stack((1.0 - probability, probability))


class CompactJointDatasetBuilder:
    """Create compact temporal partitions and disk-backed candidate input."""

    def __init__(self, model_panel_path: str | Path, *, memory_limit: str | None = None, temp_directory: str | Path | None = None) -> None:
        self.model_panel_path = Path(model_panel_path)
        self.memory_limit = memory_limit
        self.temp_directory = Path(temp_directory) if temp_directory else None

    def materialize_compact(self, destination: str | Path, *, feature_names: Sequence[str], months: Iterable[str], max_compact_rows: int | None = None) -> Path:
        """Write one filtered compact row per eligible customer-month."""
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        selected = ["ncodpers", "fecha_dato", "previous_observation", *feature_names,
                    *["acq_" + product for product in PRODUCT_COLUMNS]]
        # ``feature_names`` already contains prev_* history flags.  Preserve
        # only the 24 labels additionally needed for target reconstruction.
        selected = list(dict.fromkeys(selected))
        values = ", ".join("DATE '" + str(pd.Timestamp(value).date()) + "'" for value in months)
        if not values:
            raise ValueError("Compact partition requires at least one month.")
        limit = f" LIMIT {int(max_compact_rows)}" if max_compact_rows else ""
        source = str(self.model_panel_path).replace("'", "''")
        temporary = output.with_suffix(output.suffix + ".tmp")
        con = duckdb.connect(database=":memory:")
        try:
            _configure(con, self.memory_limit, self.temp_directory)
            columns = ", ".join(_quote(name) for name in selected)
            con.execute(
                f"COPY (SELECT {columns} FROM read_parquet('{source}') "
                f"WHERE previous_observation = 1 AND CAST(fecha_dato AS DATE) IN ({values}){limit}) "
                f"TO '{str(temporary).replace("'", "''")}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        finally:
            con.close()
        temporary.replace(output)
        return output

    def category_maps(self, feature_names: Sequence[str], categorical_feature_names: Sequence[str], months: Iterable[str]) -> dict[str, dict[str, int]]:
        """Create stable, train-partition-only integer maps for categoricals."""
        requested = set(categorical_feature_names)
        missing = requested.difference(feature_names)
        if missing:
            raise ValueError(f"Categorical features absent from feature_names: {sorted(missing)}")
        values = ", ".join("DATE '" + str(pd.Timestamp(value).date()) + "'" for value in months)
        if not values:
            raise ValueError("Category maps require at least one training month.")
        source = str(self.model_panel_path).replace("'", "''")
        con = duckdb.connect(database=":memory:")
        maps: dict[str, dict[str, int]] = {}
        try:
            _configure(con, self.memory_limit, self.temp_directory)
            for name in categorical_feature_names:
                quoted = _quote(name)
                rows = con.execute(
                    f"SELECT DISTINCT COALESCE(CAST({quoted} AS VARCHAR), '__MISSING__') "
                    f"FROM read_parquet('{source}') WHERE previous_observation = 1 "
                    f"AND CAST(fecha_dato AS DATE) IN ({values}) ORDER BY 1"
                ).fetchall()
                maps[name] = {str(row[0]): index for index, row in enumerate(rows)}
        finally:
            con.close()
        return maps

    def write_native_candidates(self, compact_path: str | Path, text_path: str | Path, *, feature_names: Sequence[str], batch_customer_months: int, categorical_maps: Mapping[str, Mapping[str, int]] | None = None) -> NativeDatasetFiles:
        """Sequentially expand compact record batches into a native CSV input."""
        if batch_customer_months < 1:
            raise ValueError("batch_customer_months must be positive.")
        compact, text = Path(compact_path), Path(text_path)
        text.parent.mkdir(parents=True, exist_ok=True)
        temporary = text.with_suffix(text.suffix + ".tmp")
        # LightGBM rejects some legitimate upstream column names containing
        # JSON-special characters. The disk cache has an internal positional
        # schema; the artifact still records the original inference contract.
        schema = pa.schema([(TARGET_COLUMN, pa.uint8()), *[(f"f{index}", pa.float32()) for index, _ in enumerate(feature_names)], ("product_id", pa.uint8())])
        # Headerless CSV prevents LightGBM from parsing labels originating in
        # the model panel. Feature names are passed explicitly and safely
        # below, so this cache is a purely positional internal format.
        writer = pacsv.CSVWriter(
            temporary,
            schema,
            write_options=pacsv.WriteOptions(include_header=False),
        )
        candidate_rows = positive_rows = 0
        try:
            parquet = pq.ParquetFile(compact)
            for record_batch in parquet.iter_batches(batch_size=batch_customer_months, columns=[*feature_names, *["acq_" + product for product in PRODUCT_COLUMNS]]):
                frame = record_batch.to_pandas()
                # Expand one product slice at a time. Peak expanded memory is
                # bounded by batch_customer_months, never 24x that batch.
                for product in PRODUCT_COLUMNS:
                    owned = pd.to_numeric(frame["prev_" + product], errors="raise").eq(0)
                    if not owned.any():
                        continue
                    part = frame.loc[owned, feature_names]
                    target = pd.to_numeric(frame.loc[owned, "acq_" + product], errors="raise").astype("uint8").to_numpy()
                    arrays = [pa.array(target, type=pa.uint8())]
                    arrays.extend(
                        pa.array(
                            _encode_category(part[name], categorical_maps[name]).to_numpy(dtype="float32", copy=False)
                            if categorical_maps and name in categorical_maps
                            else pd.to_numeric(part[name], errors="raise").to_numpy(dtype="float32", copy=False),
                            type=pa.float32(),
                        )
                        for name in feature_names
                    )
                    arrays.append(pa.array(np.full(len(part), PRODUCT_ID_MAP[product], dtype="uint8"), type=pa.uint8()))
                    writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
                    candidate_rows += len(part)
                    positive_rows += int(target.sum())
        finally:
            writer.close()
            del writer
        temporary.replace(text)
        return NativeDatasetFiles(compact, text, text.with_suffix(".bin"), candidate_rows, positive_rows, text.stat().st_size)


def train_native_joint_classifier(files: NativeDatasetFiles, artifact_directory: str | Path, *, feature_names: Sequence[str], input_dtypes: Mapping[str, str], model_version: str, random_state: int, n_estimators: int, n_jobs: int, categorical_maps: Mapping[str, Mapping[str, int]] | None = None, lightgbm_params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Construct one native Dataset and train one 500-tree shared Booster."""
    if n_estimators != 500:
        raise ValueError("The joint-classifier experiment requires n_estimators=500.")
    if files.candidate_rows == 0 or files.positive_rows == 0 or files.positive_rows == files.candidate_rows:
        raise ValueError("Candidate training representation must contain both target classes.")
    import lightgbm as lgb

    root = Path(artifact_directory); root.mkdir(parents=True, exist_ok=True)
    features = [*[f"f{index}" for index, _ in enumerate(feature_names)], "product_id"]
    start_rss = _peak_rss_bytes(); start_current_rss = _current_rss_bytes(); construct_start = time.perf_counter()
    dataset = lgb.Dataset(
        str(files.text_path),
        feature_name=features,
        categorical_feature=[index for index, name in enumerate(feature_names) if categorical_maps and name in categorical_maps] + [len(feature_names)],
        params={"header": False, "label_column": 0},
    )
    dataset.construct()
    construct_seconds = time.perf_counter() - construct_start
    dataset.save_binary(str(files.binary_path))
    after_construct_rss = _peak_rss_bytes(); after_construct_current_rss = _current_rss_bytes()
    train_start = time.perf_counter()
    booster = lgb.train({"objective": "binary", "seed": random_state, "num_threads": n_jobs, **dict(lightgbm_params or {})}, dataset, num_boost_round=500)
    train_seconds = time.perf_counter() - train_start
    after_train_rss = _peak_rss_bytes(); after_train_current_rss = _current_rss_bytes()
    model = NativeJointBooster(booster, categorical_maps)
    with (root / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    # The on-disk CSV uses safe positional headers only.  The artifact keeps
    # the public candidate-frame contract so existing validation/submission
    # adapters can select their normal source feature columns unchanged.
    artifact_features = [*feature_names, "product_id"]
    dtypes = {name: "category" if categorical_maps and name in categorical_maps else input_dtypes[name] for name in feature_names} | {"product_id": "category"}
    artifact = ModelArtifact(JOINT_ARTIFACT_PRODUCT, model_version, InputSchema(artifact_features, dtypes, "1"))
    write_artifact_metadata(root, artifact)
    metrics = {"n_estimators": 500, "candidate_rows": files.candidate_rows, "positive_rows": files.positive_rows, "positive_rate": files.positive_rows / files.candidate_rows, "categorical_features": [*(categorical_maps or {}), "product_id"], "product_id_map": PRODUCT_ID_MAP, "construct_seconds": construct_seconds, "train_seconds": train_seconds, "peak_rss_before_bytes": start_rss, "peak_rss_after_construct_bytes": after_construct_rss, "peak_rss_after_train_bytes": after_train_rss, "rss_before_bytes": start_current_rss, "rss_after_construct_bytes": after_construct_current_rss, "rss_after_train_bytes": after_train_current_rss, "text_cache_bytes": files.text_path.stat().st_size, "binary_cache_bytes": files.binary_path.stat().st_size}
    (root / "joint_metadata.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return {"artifact_directory": str(root), **metrics}


def benchmark_native_joint_training(builder: CompactJointDatasetBuilder, destination: str | Path, *, feature_names: Sequence[str], input_dtypes: Mapping[str, str], train_months: Sequence[str], compact_row_limits: Sequence[int], batch_customer_months: int, model_version: str, random_state: int, n_jobs: int, categorical_feature_names: Sequence[str] = (), lightgbm_params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run bounded native-Dataset benchmarks before attempting full training."""
    root = Path(destination); root.mkdir(parents=True, exist_ok=True)
    categorical_maps = builder.category_maps(feature_names, categorical_feature_names, train_months)
    results: list[dict[str, Any]] = []
    for limit in compact_row_limits:
        if limit < 1:
            raise ValueError("benchmark compact_row_limits must be positive.")
        name = f"compact_{limit}"
        compact = builder.materialize_compact(root / f"{name}.parquet", feature_names=feature_names, months=train_months, max_compact_rows=int(limit))
        files = builder.write_native_candidates(compact, root / f"{name}.csv", feature_names=feature_names, batch_customer_months=batch_customer_months, categorical_maps=categorical_maps)
        result = train_native_joint_classifier(files, root / name, feature_names=feature_names, input_dtypes=input_dtypes, model_version=model_version, random_state=random_state, n_estimators=500, n_jobs=n_jobs, categorical_maps=categorical_maps, lightgbm_params=lightgbm_params)
        result["compact_row_limit"] = int(limit)
        results.append(result)
    (root / "benchmark_metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return results


def _configure(con: duckdb.DuckDBPyConnection, memory_limit: str | None, temp_directory: Path | None) -> None:
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    if temp_directory:
        temp_directory.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{str(temp_directory).replace("'", "''")}'")


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _peak_rss_bytes() -> int | None:
    try:
        if resource is None:
            return None
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * 1024) if os.name != "nt" else int(value)
    except (AttributeError, OSError):
        return None


def _current_rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def _encode_category(values: pd.Series, mapping: Mapping[str, int]) -> pd.Series:
    """Encode raw categories identically for batch training and inference."""
    normalized = values.astype("string").fillna("__MISSING__")
    return normalized.map(mapping).fillna(-1).astype("int32")
