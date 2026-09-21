"""Unified product-candidate LightGBM strategy.

This module is deliberately model-specific: it consumes the existing wide
model panel and creates a temporary long-form candidate view.  It does not
change upstream preprocessing, feature engineering, or the 24-model strategy.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import pandas as pd

from src.inference.predict import InputSchema, ModelArtifact, write_artifact_metadata
from src.products import PRODUCT_COLUMNS, PRODUCT_ID_MAP, categorical_product_id, validate_product_columns


METADATA_COLUMNS = ("ncodpers", "fecha_dato", "product_id")
TARGET_COLUMN = "target"
JOINT_ARTIFACT_PRODUCT = "__joint_product_candidate__"


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class JointDatasetSchema:
    feature_names: list[str]
    product_id_map: dict[str, int]


class UnifiedProductDatasetBuilder:
    """Materialise a temporary long candidate dataset with DuckDB.

    A row is emitted only for a product unowned at t-1 and only where a prior
    observation exists.  The output is a Parquet model view, not a replacement
    for the canonical wide panel.
    """

    def __init__(self, model_panel_path: str | Path, *, memory_limit: str | None = None, temp_directory: str | Path | None = None) -> None:
        self.model_panel_path = Path(model_panel_path)
        self.memory_limit = memory_limit
        self.temp_directory = Path(temp_directory) if temp_directory else None

    def schema(self, feature_names: Sequence[str] | None = None) -> JointDatasetSchema:
        columns = self._columns()
        products = [name.removeprefix("acq_") for name in columns if name.startswith("acq_")]
        validate_product_columns(products)
        excluded = {"ncodpers", "fecha_dato", *["acq_" + p for p in PRODUCT_COLUMNS]}
        selected = list(feature_names or [name for name in columns if name not in excluded])
        missing = set(selected).difference(columns)
        if missing:
            raise ValueError(f"Requested model-panel features are missing: {sorted(missing)}")
        if set(METADATA_COLUMNS).intersection(selected):
            raise ValueError("ncodpers, fecha_dato, and product_id are metadata, not joint-model features.")
        return JointDatasetSchema(feature_names=selected, product_id_map=dict(PRODUCT_ID_MAP))

    def materialize(
        self,
        destination_path: str | Path,
        *,
        feature_names: Sequence[str] | None = None,
        target_months: Iterable[str] | None = None,
        with_target: bool = True,
    ) -> JointDatasetSchema:
        """Write a candidate-only Parquet view and return its feature contract.

        ``target_months`` selects an already-defined temporal partition.  The
        split must be chosen on the wide panel before this method is called.
        """
        if not self.model_panel_path.is_file():
            raise FileNotFoundError(f"Model panel not found: {self.model_panel_path}")
        schema = self.schema(feature_names)
        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        feature_sql = ", ".join(_quote(name) for name in schema.feature_names)
        months = list(target_months or [])
        for month in months:
            pd.Timestamp(month)  # validate before interpolating into SQL
        month_filter = ""
        if months:
            values = ", ".join("DATE '" + str(pd.Timestamp(month).date()) + "'" for month in months)
            month_filter = f" AND CAST(fecha_dato AS DATE) IN ({values})"
        branches: list[str] = []
        for product in PRODUCT_COLUMNS:
            target_sql = f", CAST({_quote('acq_' + product)} AS UTINYINT) AS {TARGET_COLUMN}" if with_target else ""
            branches.append(
                f"SELECT ncodpers, CAST(fecha_dato AS DATE) AS fecha_dato"
                f"{', ' if feature_sql else ''}{feature_sql}, "
                f"CAST({PRODUCT_ID_MAP[product]} AS UTINYINT) AS product_id{target_sql} "
                f"FROM read_parquet(?) WHERE previous_observation = 1 "
                f"AND {_quote('prev_' + product)} = 0{month_filter}"
            )
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        con = duckdb.connect(database=":memory:")
        try:
            if self.memory_limit:
                con.execute(f"SET memory_limit = '{self.memory_limit}'")
            if self.temp_directory:
                self.temp_directory.mkdir(parents=True, exist_ok=True)
                escaped_directory = str(self.temp_directory).replace("'", "''")
                con.execute(f"SET temp_directory = '{escaped_directory}'")
            query = "COPY (" + " UNION ALL ".join(branches) + ") TO ? (FORMAT PARQUET, COMPRESSION ZSTD)"
            con.execute(query, [str(self.model_panel_path)] * len(branches) + [str(temporary)])
        finally:
            con.close()
        temporary.replace(destination)
        return schema

    def _columns(self) -> list[str]:
        con = duckdb.connect(database=":memory:")
        try:
            return [row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(self.model_panel_path)]).fetchall()]
        finally:
            con.close()


def train_joint_classifier(
    long_dataset_path: str | Path,
    artifact_directory: str | Path,
    *,
    feature_names: Sequence[str],
    model_version: str = "lightgbm-joint-v1",
    random_state: int = 42,
    n_estimators: int = 500,
    n_jobs: int = 1,
    lightgbm_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train exactly one binary LightGBM with unordered ``product_id``."""
    if n_estimators != 500:
        raise ValueError("The joint-classifier experiment requires n_estimators=500.")
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("LightGBM is required. Install dependencies from requirements.txt.") from exc
    source, root = Path(long_dataset_path), Path(artifact_directory)
    if not source.is_file():
        raise FileNotFoundError(f"Joint dataset not found: {source}")
    model_features = [*feature_names, "product_id"]
    frame = pd.read_parquet(source, columns=[*model_features, TARGET_COLUMN])
    if frame.empty or set(frame[TARGET_COLUMN].dropna().unique()).difference({0, 1}) or frame[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Joint target must contain both binary classes 0 and 1.")
    non_numeric = [name for name in feature_names if not pd.api.types.is_numeric_dtype(frame[name])]
    if non_numeric:
        raise TypeError(f"Joint model features must be numeric/encoded: {non_numeric}")
    frame["product_id"] = categorical_product_id(frame["product_id"])
    X, y = frame.loc[:, model_features], frame[TARGET_COLUMN].astype("uint8")
    classifier = lgb.LGBMClassifier(
        objective="binary", n_estimators=500, random_state=random_state, n_jobs=n_jobs,
        **dict(lightgbm_params or {}),
    )
    classifier.fit(X, y, categorical_feature=["product_id"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "model.pkl").open("wb") as handle:
        pickle.dump(classifier, handle)
    artifact = ModelArtifact(
        product=JOINT_ARTIFACT_PRODUCT,
        model_version=model_version,
        schema=InputSchema(feature_names=model_features, dtypes={name: str(X[name].dtype) for name in model_features}, version="1"),
    )
    write_artifact_metadata(root, artifact)
    metrics = {
        "n_estimators": 500,
        "candidate_rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "positive_rate": float(y.mean()),
        "categorical_features": ["product_id"],
        "product_id_map": PRODUCT_ID_MAP,
    }
    (root / "joint_metadata.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return {"artifact_directory": str(root), **metrics}


def expand_joint_inference_candidates(prepared_input: pd.DataFrame, *, feature_names: Sequence[str]) -> pd.DataFrame:
    """Create candidate rows for inference while retaining reconstruction metadata."""
    required = {"ncodpers", "fecha_dato", *feature_names, *["prev_" + p for p in PRODUCT_COLUMNS]}
    missing = required.difference(prepared_input.columns)
    if missing:
        raise ValueError(f"Prepared input is missing required joint-inference columns: {sorted(missing)}")
    parts: list[pd.DataFrame] = []
    for product in PRODUCT_COLUMNS:
        candidate = prepared_input.loc[pd.to_numeric(prepared_input["prev_" + product], errors="raise").eq(0), ["ncodpers", "fecha_dato", *feature_names]].copy()
        candidate["product_id"] = PRODUCT_ID_MAP[product]
        parts.append(candidate)
    result = pd.concat(parts, ignore_index=True)
    result["product_id"] = categorical_product_id(result["product_id"])
    return result


def rank_joint_candidates(candidates: pd.DataFrame, probabilities: Sequence[float], *, top_k: int = 7) -> pd.DataFrame:
    """Rank P(y=1) per customer and map categorical IDs back to product names."""
    if len(candidates) != len(probabilities):
        raise ValueError("Candidate rows and probability scores must have equal length.")
    scored = candidates.loc[:, ["ncodpers", "product_id"]].copy()
    scored["score"] = list(probabilities)
    scored["product_id"] = scored["product_id"].astype("uint8")
    scored["product"] = scored["product_id"].map({value: key for key, value in PRODUCT_ID_MAP.items()})
    ranked = scored.sort_values(["ncodpers", "score", "product_id"], ascending=[True, False, True], kind="stable")
    ranked["rank"] = ranked.groupby("ncodpers", sort=False).cumcount().add(1).astype("uint8")
    return ranked.loc[ranked["rank"].le(top_k)].copy()
