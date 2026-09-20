"""Reproducible 24-product LightGBM acquisition training pipeline."""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.features.model_panel import build_model_panel
from src.models.lightgbm_binary import train_product_classifiers
from src.tracking.progress import PipelineProgress
from src.tracking.run_context import build_run_manifest, create_run_id, write_run_manifest


def run_lightgbm_v1_from_config(config_path: str | Path = "configs/baselines/lightgbm_v1.yaml") -> dict[str, Any]:
    """Load an experiment config and persist complete lineage for this run."""
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("LightGBM config must be a YAML mapping.")
    return run_lightgbm_v1(config, resolved_config_path=path)


def run_lightgbm_v1(config: Mapping[str, Any], *, resolved_config_path: str | Path) -> dict[str, Any]:
    """Build the leakage-safe panel, train 24 artifacts, then write lineage."""
    pipeline = config.get("pipeline", {})
    description = str(pipeline.get("description", "")).strip()
    version = str(pipeline.get("version", "")).strip()
    if not version or not description:
        raise ValueError("pipeline.version and pipeline.description are required for every run.")
    data, runtime, model = config["data"], config.get("runtime", {}), config["model"]
    root = Path(os.getenv("SANTANDER_DATA_ROOT", "data"))
    source = root / data["interim_train"]
    panel = root / data["model_panel"]
    artifact_root = root / data["artifact_dir"] / "runs"
    run_id = create_run_id(version)
    artifacts = artifact_root / run_id
    artifacts.mkdir(parents=True, exist_ok=True)
    progress = PipelineProgress(artifact_root / "pipeline_timing.json", ["model_panel", "train", "tracking"])
    with progress.stage("model_panel"):
        panel_path = build_model_panel(
            source, panel,
            force_process=bool(config.get("features", {}).get("force_process", False)),
            memory_limit=runtime.get("memory_limit"), temp_directory=runtime.get("temp_directory"),
        )
    with progress.stage("train"):
        models = train_product_classifiers(
            panel_path, artifacts / "models", model_version=str(model["version"]),
            max_rows_per_product=model.get("max_rows_per_product"),
            random_state=int(model.get("random_state", 42)),
            n_estimators=int(model.get("n_estimators", 300)), n_jobs=int(runtime.get("threads", 1)),
        )
    model_index = artifacts / "model_artifacts.json"
    model_index.write_text(json.dumps(models, indent=2, sort_keys=True), encoding="utf-8")
    resolved_copy = artifacts / "config_resolved.yaml"
    resolved_copy.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    manifest = build_run_manifest(run_id=run_id, config=config, inputs={"source_train": source, "model_panel": panel_path, "declared_config": resolved_config_path}, outputs={"resolved_config": resolved_copy, "model_index": model_index})
    manifest["description"] = description
    manifest["pipeline_version"] = version
    manifest["model_version"] = str(model["version"])
    manifest_path = write_run_manifest(manifest, artifacts / "lineage_manifest.json")
    mlflow_run_id = None
    with progress.stage("tracking"):
        tracking = config.get("tracking", {}).get("mlflow", {})
        if tracking.get("enabled", False):
            from src.tracking.mlflow_utils import log_baseline_run
            mlflow_run_id = log_baseline_run(
                experiment_name=str(tracking["experiment_name"]),
                tracking_uri=str(tracking["tracking_uri"]),
                manifest_path=manifest_path,
                config_path=resolved_copy,
                artifact_dir=artifacts,
                metrics={},
                tags={"run_id": run_id, "pipeline_version": version, "model_version": model["version"], "description": description},
            )
    progress.save()
    return {"run_id": run_id, "description": description, "pipeline_version": version, "model_version": model["version"], "model_panel": str(panel_path), "models": models, "artifacts": str(artifacts), "manifest": str(manifest_path), "mlflow_run_id": mlflow_run_id}
