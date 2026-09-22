"""One shared XGBoost classifier trained with disk-backed external memory."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.inference.predict import InputSchema, ModelArtifact, write_artifact_metadata
from src.models.lightgbm_joint import JOINT_ARTIFACT_PRODUCT
from src.models.lightgbm_joint_native import (
    CompactJointDatasetBuilder,
    NativeDatasetFiles,
    _current_rss_bytes,
    _encode_category,
)


class XGBoostJointBooster:
    """Pickle-friendly probability adapter that preserves the public schema."""

    def __init__(self, booster: Any, categorical_maps: Mapping[str, Mapping[str, int]]) -> None:
        self.booster = booster
        self.categorical_maps = {name: dict(values) for name, values in categorical_maps.items()}

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        encoded = frame.copy()
        for name, mapping in self.categorical_maps.items():
            encoded[name] = _encode_category(encoded[name], mapping)
        matrix = xgb.DMatrix(encoded.apply(pd.to_numeric, errors="raise").to_numpy(dtype="float32", copy=False))
        probability = np.asarray(self.booster.predict(matrix), dtype="float64")
        return np.column_stack((1.0 - probability, probability))


def train_xgboost_external_joint_classifier(
    files: NativeDatasetFiles,
    artifact_directory: str | Path,
    *,
    feature_names: Sequence[str],
    input_dtypes: Mapping[str, str],
    categorical_maps: Mapping[str, Mapping[str, int]],
    cache_directory: str | Path,
    model_version: str,
    random_state: int,
    n_estimators: int,
    n_jobs: int,
    batch_rows: int,
    xgboost_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train exactly one 100-tree booster through ExtMemQuantileDMatrix.

    XGBoost caches pages under ``cache_directory`` and fetches them on demand;
    the full candidate CSV is never converted to one pandas DataFrame.
    """
    if n_estimators != 100:
        raise ValueError("The external-memory joint experiment requires n_estimators=100.")
    if files.candidate_rows == 0 or files.positive_rows == 0:
        raise ValueError("Candidate representation must contain both classes.")
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive.")
    import xgboost as xgb

    root = Path(artifact_directory); root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_directory); cache_root.mkdir(parents=True, exist_ok=True)
    cache_prefix = cache_root / "xgb-extmem"
    feature_count = len(feature_names) + 1

    class CandidateCsvIterator(xgb.DataIter):
        def __init__(self) -> None:
            self._chunks = None
            super().__init__(cache_prefix=str(cache_prefix), release_data=True)

        def reset(self) -> None:
            self._chunks = pd.read_csv(files.text_path, header=None, chunksize=batch_rows, dtype="float32")

        def next(self, input_data: Any) -> bool:
            if self._chunks is None:
                self.reset()
            try:
                chunk = next(self._chunks)
            except StopIteration:
                return False
            if chunk.shape[1] != feature_count + 1:
                raise ValueError(f"Expected target plus {feature_count} features, got {chunk.shape[1]} columns.")
            input_data(data=chunk.iloc[:, 1:].to_numpy(dtype="float32", copy=False), label=chunk.iloc[:, 0].to_numpy(dtype="float32", copy=False))
            return True

    before_rss = _current_rss_bytes(); construct_started = time.perf_counter()
    iterator = CandidateCsvIterator()
    matrix = xgb.ExtMemQuantileDMatrix(iterator, max_bin=int(dict(xgboost_params or {}).get("max_bin", 256)), nthread=n_jobs)
    construct_seconds = time.perf_counter() - construct_started
    after_construct_rss = _current_rss_bytes()
    train_started = time.perf_counter()
    booster = xgb.train(
        {"objective": "binary:logistic", "tree_method": "hist", "seed": random_state, "nthread": n_jobs, **dict(xgboost_params or {})},
        matrix,
        num_boost_round=100,
    )
    train_seconds = time.perf_counter() - train_started
    after_train_rss = _current_rss_bytes()
    booster.save_model(str(root / "model.json"))
    model = XGBoostJointBooster(booster, categorical_maps)
    with (root / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    public_features = [*feature_names, "product_id"]
    dtypes = {name: "category" if name in categorical_maps else input_dtypes[name] for name in feature_names} | {"product_id": "category"}
    write_artifact_metadata(root, ModelArtifact(JOINT_ARTIFACT_PRODUCT, model_version, InputSchema(public_features, dtypes, "1")))
    cache_bytes = sum(path.stat().st_size for path in cache_root.glob("xgb-extmem*"))
    metrics = {
        "n_estimators": 100,
        "candidate_rows": files.candidate_rows,
        "positive_rows": files.positive_rows,
        "positive_rate": files.positive_rows / files.candidate_rows,
        "categorical_features": [*categorical_maps, "product_id"],
        "construct_seconds": construct_seconds,
        "train_seconds": train_seconds,
        "rss_before_bytes": before_rss,
        "rss_after_construct_bytes": after_construct_rss,
        "rss_after_train_bytes": after_train_rss,
        "candidate_csv_bytes": files.text_path.stat().st_size,
        "external_cache_bytes": cache_bytes,
    }
    (root / "joint_metadata.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return {"artifact_directory": str(root), **metrics}


def benchmark_xgboost_external_joint_training(
    builder: CompactJointDatasetBuilder,
    destination: str | Path,
    *,
    feature_names: Sequence[str],
    input_dtypes: Mapping[str, str],
    train_months: Sequence[str],
    compact_row_limits: Sequence[int],
    expand_batch_customer_months: int,
    xgboost_batch_rows: int,
    cache_directory: str | Path,
    model_version: str,
    random_state: int,
    n_jobs: int,
    categorical_feature_names: Sequence[str],
    work_directory: str | Path,
    xgboost_params: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Benchmark while keeping large compact/CSV staging off persistent artifacts."""
    root = Path(destination); root.mkdir(parents=True, exist_ok=True)
    work_root = Path(work_directory); work_root.mkdir(parents=True, exist_ok=True)
    category_maps = builder.category_maps(feature_names, categorical_feature_names, train_months)
    results: list[dict[str, Any]] = []
    for limit in compact_row_limits:
        name = f"compact_{int(limit)}"
        compact = builder.materialize_compact(work_root / f"{name}.parquet", feature_names=feature_names, months=train_months, max_compact_rows=int(limit))
        files = builder.write_native_candidates(compact, work_root / f"{name}.csv", feature_names=feature_names, batch_customer_months=expand_batch_customer_months, categorical_maps=category_maps)
        result = train_xgboost_external_joint_classifier(
            files, root / name, feature_names=feature_names, input_dtypes=input_dtypes,
            categorical_maps=category_maps, cache_directory=Path(cache_directory) / name,
            model_version=model_version, random_state=random_state, n_estimators=100,
            n_jobs=n_jobs, batch_rows=xgboost_batch_rows, xgboost_params=xgboost_params,
        )
        result["compact_row_limit"] = int(limit)
        results.append(result)
    (root / "benchmark_metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return results
