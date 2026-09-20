"""End-to-end, cache-aware popularity baseline v0 pipeline."""
from __future__ import annotations

import os
import shutil
import re
from pathlib import Path
from typing import Any, Mapping

import yaml
import duckdb

from src.data.gcs_storage import download_object_if_missing, object_name, upload_directory, upload_file
from src.data.preprocessing import preprocess_customer_profiles
from src.ingestion.checkpoint import build_interim_checkpoint
from src.models.popularity import run_popularity_baseline
from src.splits.temporal import build_validation_split
from src.tracking.progress import PipelineProgress
from src.tracking.run_context import build_run_manifest, create_run_id, write_run_manifest


def run_baseline_v0_from_config(config_path: str | Path = "configs/baselines/v0.yaml") -> dict[str, Any]:
    """Load a versioned config and execute every v0 pipeline stage."""
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Baseline config must be a YAML mapping.")
    return run_baseline_v0(_resolve_environment(config), resolved_config_path=path)


def run_baseline_v0(config: Mapping[str, Any], *, resolved_config_path: str | Path) -> dict[str, Any]:
    """Run GCS cache -> ingest -> preprocess -> train/evaluate -> upload."""
    paths = _paths(config)
    run_id = create_run_id("baseline-v0")
    artifacts = paths["artifact_root"] / run_id
    artifacts.mkdir(parents=True, exist_ok=True)
    progress = PipelineProgress(paths["artifact_root"] / "pipeline_timing.json", ["gcs_raw_cache", "ingest", "split", "preprocessing", "train_evaluate", "upload_artifacts"])
    storage, runtime = config["storage"], config["runtime"]
    raw_path = paths["raw_train"]
    raw_object = object_name(storage["raw_prefix"], Path(config["data"]["raw_train_filename"]))
    with progress.stage("gcs_raw_cache"):
        downloaded = download_object_if_missing(storage["bucket"], raw_object, raw_path, storage.get("project_id"), bool(config["ingestion"].get("force_download", False)))
    with progress.stage("ingest"):
        interim = build_interim_checkpoint(raw_path, paths["source_train"], chunksize=int(config["ingestion"]["chunksize"]), force_rebuild=bool(config["ingestion"].get("force_rebuild", False)), show_progress=bool(config["ingestion"].get("show_progress", True)))
        interim_uri = _upload_file_if_enabled(interim["interim"], storage, "train.parquet")

    with progress.stage("split"):
        split = build_validation_split(interim["interim"], paths["split_dir"], validation_date=str(config["split"]["validation_date"]), force_split=bool(config["split"].get("force_split", False)), memory_limit=runtime.get("memory_limit"), temp_directory=_temp_path(runtime))
        split_uris = _upload_directory_if_enabled(paths["split_dir"], storage, f"splits/{config['split']['name']}")

    preprocessing = config["preprocessing"]
    with progress.stage("preprocessing"):
        profiles = preprocess_customer_profiles(split["train"], None, paths["profile_dir"], force_process=bool(preprocessing.get("force_process", False)), memory_limit=runtime.get("memory_limit"), temp_directory=_temp_path(runtime), renta_clip_lower_quantile=float(preprocessing["renta_clip_lower_quantile"]), renta_clip_upper_quantile=float(preprocessing["renta_clip_upper_quantile"]))
        profile_uris = _upload_directory_if_enabled(paths["profile_dir"], storage, "baseline_v0/profiles")
    with progress.stage("train_evaluate"):
        run_scratch = _temp_path(runtime) / run_id
        run_scratch.mkdir(parents=True, exist_ok=True)
        scoring_panel = _materialize_scoring_panel(split, run_scratch / "scoring_panel.parquet", runtime)
        model = run_popularity_baseline(scoring_panel, artifacts, validation_date=str(config["split"]["validation_date"]), top_k=int(config["model"]["top_k"]), memory_limit=runtime.get("memory_limit"), temp_directory=run_scratch, threads=int(runtime.get("threads", 1)))
        scoring_panel.unlink(missing_ok=True)
        shutil.rmtree(run_scratch, ignore_errors=True)
    resolved_copy = artifacts / "config_resolved.yaml"
    resolved_copy.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    manifest_path = artifacts / "lineage_manifest.json"
    manifest = build_run_manifest(run_id=run_id, config=config, inputs={"raw_train": raw_path, "source_train": interim["interim"], "split_train": split["train"], "validation_input": split["validation_input"], "validation_target": split["validation_target"]}, outputs={"profile_train": profiles["train"], "metrics": model["metrics_path"], "ranking": artifacts / "popularity_ranking.parquet", "predictions": artifacts / "validation_predictions.parquet", "resolved_config": resolved_copy})
    write_run_manifest(manifest, manifest_path)
    mlflow_run_id = None
    tracking = config.get("tracking", {}).get("mlflow", {})
    if tracking.get("enabled", False):
        from src.tracking.mlflow_utils import log_baseline_run
        mlflow_run_id = log_baseline_run(experiment_name=str(tracking["experiment_name"]), tracking_uri=str(tracking["tracking_uri"]), manifest_path=manifest_path, config_path=resolved_copy, artifact_dir=artifacts, metrics=model["metrics"], tags={"pipeline_version": config["pipeline"]["version"], "run_id": run_id})
    with progress.stage("upload_artifacts"):
        artifact_uris = _upload_artifacts_if_enabled(artifacts, storage, run_id)
    progress.save()
    return {"run_id": run_id, "raw_downloaded": str(downloaded) if downloaded else None, "interim": interim, "interim_gcs_uri": interim_uri, "profiles": profiles, "profile_gcs_uris": profile_uris, "split": split, "split_gcs_uris": split_uris, "model": model, "mlflow_run_id": mlflow_run_id, "artifacts": str(artifacts), "artifact_gcs_uris": artifact_uris, "manifest": str(manifest_path)}


def _paths(config: Mapping[str, Any]) -> dict[str, Path]:
    root = Path(os.getenv("SANTANDER_DATA_ROOT", "data"))
    data = config["data"]
    return {"raw_train": root / data["raw_dir"] / data["raw_train_filename"], "source_train": root / data["interim_dir"] / "train.parquet", "split_dir": root / data["interim_dir"] / "splits" / config["split"]["name"], "profile_dir": root / data["processed_dir"] / "baseline_v0" / "profiles", "artifact_root": root / data["artifact_dir"] / "runs"}


def _materialize_scoring_panel(split: Mapping[str, object], destination: Path, runtime: Mapping[str, Any]) -> Path:
    """Recombine cached train/input/target solely for disk-backed evaluation."""
    con = duckdb.connect(database=":memory:")
    try:
        if runtime.get("memory_limit"): con.execute(f"SET memory_limit = '{runtime['memory_limit']}'")
        previous_columns = [row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [split["validation_input"]]).fetchall() if row[0].startswith("prev_")]
        excluded = ", ".join('"' + column + '"' for column in previous_columns)
        temp = destination.with_suffix(".tmp")
        escaped_temp = str(temp).replace("'", "''")
        con.execute(f"COPY (SELECT * FROM read_parquet(?) UNION ALL BY NAME SELECT input.* EXCLUDE ({excluded}), target.* EXCLUDE (ncodpers, fecha_dato) FROM read_parquet(?) input JOIN read_parquet(?) target USING (ncodpers, fecha_dato)) TO '{escaped_temp}' (FORMAT PARQUET, COMPRESSION ZSTD)", [split["train"], split["validation_input"], split["validation_target"]])
        temp.replace(destination)
        return destination
    finally:
        con.close()


def _temp_path(runtime: Mapping[str, Any]) -> Path:
    return Path(runtime["temp_directory"])


def _upload_file_if_enabled(path: str | Path, storage: Mapping[str, Any], relative: str) -> str | None:
    if not storage.get("upload_checkpoints", True): return None
    return upload_file(path, storage["bucket"], object_name(storage["checkpoint_prefix"], relative), storage.get("project_id"))


def _upload_directory_if_enabled(path: str | Path, storage: Mapping[str, Any], prefix: str) -> list[str]:
    if not storage.get("upload_checkpoints", True): return []
    return upload_directory(path, storage["bucket"], object_name(storage["checkpoint_prefix"], prefix), storage.get("project_id"))


def _upload_artifacts_if_enabled(path: str | Path, storage: Mapping[str, Any], run_id: str) -> list[str]:
    if not storage.get("upload_artifacts", True): return []
    return upload_directory(path, storage["bucket"], object_name(storage["artifact_prefix"], run_id), storage.get("project_id"))


def _resolve_environment(value: Any) -> Any:
    """Replace ${NAME} config values from bootstrap-provided environment variables."""
    if isinstance(value, dict): return {key: _resolve_environment(item) for key, item in value.items()}
    if isinstance(value, list): return [_resolve_environment(item) for item in value]
    if not isinstance(value, str): return value
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.getenv(name)
        if not resolved: raise ValueError(f"Missing required environment variable: {name}")
        return resolved
    return re.sub(r"\$\{([A-Z0-9_]+)\}", replace, value)
