"""End-to-end pipeline for one shared product-candidate LightGBM model."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import duckdb
import pandas as pd
import yaml

from src.features.model_panel import build_model_panel
from src.inference.predict import load_artifact
from src.ingestion.checkpoint import build_interim_checkpoint
from src.models.lightgbm_joint import (
    UnifiedProductDatasetBuilder,
    expand_joint_inference_candidates,
    rank_joint_candidates,
    train_joint_classifier,
)
from src.products import PRODUCT_COLUMNS, PRODUCT_NAME_BY_ID, categorical_product_id
from src.submission.competition import run_ranked_candidate_submission
from src.submission.prepare import load_competition_input, materialize_competition_test_input
from src.tracking.run_context import build_run_manifest, create_run_id, write_run_manifest


def run_lightgbm_joint_v1_from_config(
    config_path: str | Path = "configs/baselines/lightgbm_joint_v1.yaml",
) -> dict[str, Any]:
    """Load the versioned experiment configuration and run Pipeline B."""
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Joint LightGBM config must be a YAML mapping.")
    return run_lightgbm_joint_v1(config, resolved_config_path=path)


def run_lightgbm_joint_v1(
    config: Mapping[str, Any], *, resolved_config_path: str | Path,
) -> dict[str, Any]:
    """Build the existing panel, train one model, validate, and submit."""
    pipeline, data, model = config["pipeline"], config["data"], config["model"]
    runtime, split, competition = config.get("runtime", {}), config["split"], config["competition"]
    if int(model["n_estimators"]) != 500:
        raise ValueError("The joint-model experiment must use n_estimators=500.")
    root = Path(os.getenv("SANTANDER_DATA_ROOT", "data"))
    # Bootstrap supplies the fast Colab spill location; retain the config value
    # for local runs where that environment variable is absent.
    temp_directory = os.getenv("SANTANDER_DUCKDB_TEMP_DIRECTORY", str(runtime.get("temp_directory", "data/.duckdb_tmp")))
    runtime = {**runtime, "temp_directory": temp_directory}
    run_id = create_run_id(str(pipeline["version"]))
    run_directory = root / data["artifact_dir"] / "runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=True)

    panel_path = build_model_panel(
        root / data["interim_train"], root / data["model_panel"],
        force_process=bool(config.get("features", {}).get("force_process", False)),
        memory_limit=runtime.get("memory_limit"), temp_directory=temp_directory,
    )
    features = _numeric_feature_names(panel_path)
    validation_date = str(split["validation_date"])
    builder = UnifiedProductDatasetBuilder(panel_path, memory_limit=runtime.get("memory_limit"), temp_directory=temp_directory)
    train_long, validation_long = run_directory / "train_candidates.parquet", run_directory / "validation_candidates.parquet"
    builder.materialize(train_long, feature_names=features, target_months=_months_before(panel_path, validation_date), with_target=True)
    builder.materialize(validation_long, feature_names=features, target_months=[validation_date], with_target=True)

    artifact_directory = run_directory / "joint_model"
    training = train_joint_classifier(
        train_long, artifact_directory, feature_names=features, model_version=str(model["version"]),
        random_state=int(model.get("random_state", 42)), n_estimators=500,
        n_jobs=int(runtime.get("threads", 1)), lightgbm_params=model.get("lightgbm_params"),
    )
    validation = _validate(artifact_directory, validation_long, features, int(competition.get("top_k", 7)))
    validation_path = run_directory / "validation_metrics.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    submission = _competition_submission(
        root, root / data["interim_train"], run_directory, artifact_directory, features, runtime, competition,
    )
    resolved_copy = run_directory / "config_resolved.yaml"
    resolved_copy.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    manifest = build_run_manifest(
        run_id=run_id, config=config,
        inputs={"interim_train": root / data["interim_train"], "model_panel": panel_path},
        outputs={"joint_model": artifact_directory, "validation_metrics": validation_path, "submission": submission, "config": resolved_copy},
    )
    manifest_path = write_run_manifest(manifest, run_directory / "lineage_manifest.json")
    return {"run_id": run_id, "artifacts": str(run_directory), "model": training, "validation": validation, "submission": str(submission), "manifest": str(manifest_path)}


def _numeric_feature_names(panel_path: Path) -> list[str]:
    con = duckdb.connect(database=":memory:")
    try:
        schema = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(panel_path)]).fetchdf()
    finally:
        con.close()
    excluded = {"ncodpers", "fecha_dato", "previous_observation", *["acq_" + product for product in PRODUCT_COLUMNS]}
    prefixes = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "FLOAT", "DOUBLE", "DECIMAL", "BOOLEAN")
    features = [row.column_name for row in schema.itertuples(index=False) if row.column_name not in excluded and str(row.column_type).upper().startswith(prefixes)]
    if not features:
        raise ValueError("Existing model panel contains no numeric/encoded model features.")
    return features


def _months_before(panel_path: Path, validation_date: str) -> list[str]:
    con = duckdb.connect(database=":memory:")
    try:
        dates = con.execute("SELECT DISTINCT CAST(fecha_dato AS DATE) FROM read_parquet(?) WHERE CAST(fecha_dato AS DATE) < CAST(? AS DATE) ORDER BY 1", [str(panel_path), validation_date]).fetchall()
    finally:
        con.close()
    if not dates:
        raise ValueError("Temporal train partition is empty.")
    return [str(value[0]) for value in dates]


def _validate(artifact_directory: Path, long_path: Path, features: list[str], top_k: int) -> dict[str, float | int]:
    artifact, classifier = load_artifact(artifact_directory)
    frame = pd.read_parquet(long_path, columns=["ncodpers", "product_id", "target", *features])
    X = frame.loc[:, artifact.schema.feature_names].copy()
    X["product_id"] = categorical_product_id(X["product_id"])
    ranked = rank_joint_candidates(frame[["ncodpers", "product_id"]], classifier.predict_proba(X)[:, 1], top_k=top_k)
    positives = frame.loc[frame["target"].eq(1), ["ncodpers", "product_id"]].copy()
    positives["product_id"] = positives["product_id"].astype("uint8")
    hits = ranked.merge(positives.drop_duplicates(), on=["ncodpers", "product_id"], how="inner")
    return {"top_k": top_k, "candidate_rows": int(len(frame)), "positive_rows": int(len(positives)), "positive_product_recall_at_k": float(len(hits) / len(positives)) if len(positives) else 0.0, "customer_recall_at_k": float(hits["ncodpers"].nunique() / positives["ncodpers"].nunique()) if len(positives) else 0.0}


def _competition_submission(root: Path, train_path: Path, run_directory: Path, artifact_directory: Path, features: list[str], runtime: Mapping[str, Any], competition: Mapping[str, Any]) -> Path:
    test_checkpoint = root / competition["interim_test"]
    build_interim_checkpoint(root / competition["raw_test"], test_checkpoint, chunksize=int(competition["chunksize"]), force_rebuild=bool(competition.get("force_rebuild", False)), show_progress=bool(competition.get("show_progress", True)))
    prepared = materialize_competition_test_input(test_checkpoint, train_path, root / competition["prepared_input"], product_names=PRODUCT_COLUMNS, feature_names=features, history_date=str(competition["history_date"]), memory_limit=runtime.get("memory_limit"), temp_directory=runtime.get("temp_directory"))
    frame = load_competition_input(prepared, columns=["ncodpers", "fecha_dato", *features, *["prev_" + product for product in PRODUCT_COLUMNS]])
    artifact, classifier = load_artifact(artifact_directory)
    def score_joint_batch(customer_batch: pd.DataFrame) -> pd.DataFrame:
        """Model adapter: replace only this function when changing models."""
        candidates = expand_joint_inference_candidates(customer_batch, feature_names=features)
        X = candidates.loc[:, artifact.schema.feature_names].copy()
        X["product_id"] = categorical_product_id(X["product_id"])
        return pd.DataFrame({
            "ncodpers": candidates["ncodpers"].to_numpy(),
            "product": candidates["product_id"].astype("uint8").map(PRODUCT_NAME_BY_ID).to_numpy(),
            "score": classifier.predict_proba(X)[:, 1],
        })
    template_path = root / competition["sample_submission"]
    path = run_directory / "submission.csv"
    return run_ranked_candidate_submission(
        frame, template_path if template_path.is_file() else None, path,
        score_batch=score_joint_batch, top_k=int(competition["top_k"]),
        batch_customers=int(competition.get("batch_customers", 50_000)),
    )
