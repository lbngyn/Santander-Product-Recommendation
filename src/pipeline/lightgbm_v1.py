"""Reproducible 24-product LightGBM acquisition training pipeline."""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Mapping

import yaml
import pandas as pd

from src.features.model_panel_v1 import build_model_panel_v1
from src.inference.predict import load_artifact_metadata
from src.ingestion.checkpoint import build_interim_checkpoint
from src.models.lightgbm_binary import train_product_classifiers
from src.submission.competition import run_competition_test_inference
from src.submission.prepare import load_competition_input, materialize_competition_test_input
from src.tracking.progress import PipelineProgress
from src.tracking.run_context import build_run_manifest, create_run_id, write_run_manifest


def run_lightgbm_v1_from_config(config_path: str | Path = "configs/baselines/lightgbm_v1.yaml") -> dict[str, Any]:
    """Load an experiment config and persist complete lineage for this run."""
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("LightGBM config must be a YAML mapping.")
    return run_lightgbm_v1(config, resolved_config_path=path)


def run_lightgbm_v1(
    config: Mapping[str, Any],
    *,
    resolved_config_path: str | Path,
    panel_builder: Any = build_model_panel_v1,
    require_adjacent_month: bool = False,
    categorical_feature_names: list[str] | None = None,
) -> dict[str, Any]:
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
        panel_path = panel_builder(
            source, panel,
            force_process=bool(config.get("features", {}).get("force_process", False)),
            memory_limit=runtime.get("memory_limit"), temp_directory=runtime.get("temp_directory"),
        )
    with progress.stage("train"):
        models = train_product_classifiers(
            panel_path, artifacts / "models", model_version=str(model["version"]),
            product_names=model.get("products"),
            max_rows_per_product=model.get("max_rows_per_product"),
            max_negative_rows_per_product=model.get("max_negative_rows_per_product"),
            max_total_rows_per_product=model.get("max_total_rows_per_product"),
            random_state=int(model.get("random_state", 42)),
            n_estimators=int(model.get("n_estimators", 300)), n_jobs=int(runtime.get("threads", 1)),
            memory_limit=runtime.get("memory_limit"), temp_directory=runtime.get("temp_directory"),
            lightgbm_params=model.get("lightgbm_params"),
            training_log_period=int(model.get("training_log_period", 5)),
            require_adjacent_month=require_adjacent_month,
            categorical_feature_names=categorical_feature_names,
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


def run_lightgbm_v1_competition_from_config(
    model_index_path: str | Path,
    config_path: str | Path | None = None,
) -> Path:
    """Prepare ``test_ver2.csv`` and create the competition submission.

    ``model_index_path`` is the ``model_artifacts.json`` returned by the
    training run, keeping the submission tied to a specific immutable run.
    """
    index = Path(model_index_path)
    run_directory = index.parent
    # Prefer the immutable config copied beside the trained models. This keeps
    # submission preparation traceable to the exact model run.
    resolved_config = Path(config_path) if config_path else run_directory / "config_resolved.yaml"
    if not resolved_config.is_file():
        raise FileNotFoundError(f"Run config not found: {resolved_config}")
    config = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("LightGBM config must be a YAML mapping.")
    competition = config["competition"]
    root = Path(os.getenv("SANTANDER_DATA_ROOT", "data"))
    artifact_directories = json.loads(index.read_text(encoding="utf-8"))
    if not isinstance(artifact_directories, dict) or len(artifact_directories) != 24:
        raise ValueError("model_artifacts.json must map exactly 24 products to artifact directories.")
    metadata = {product: load_artifact_metadata(path) for product, path in artifact_directories.items()}
    if any(metadata[product].product != product for product in artifact_directories):
        raise ValueError("A model artifact does not match its product index key.")
    product_names = list(artifact_directories)
    feature_names = sorted({name for artifact in metadata.values() for name in artifact.schema.feature_names})
    expected_dtypes: dict[str, str] = {}
    for artifact in metadata.values():
        for name, dtype in artifact.schema.dtypes.items():
            if name in expected_dtypes and expected_dtypes[name] != dtype:
                raise ValueError(f"Conflicting artifact dtypes for {name!r}: {expected_dtypes[name]!r}, {dtype!r}")
            expected_dtypes[name] = dtype
    runtime = config.get("runtime", {})
    test_checkpoint = root / competition["interim_test"]
    build_interim_checkpoint(
        root / competition["raw_test"], test_checkpoint,
        chunksize=int(competition.get("chunksize", 50_000)),
        force_rebuild=bool(competition.get("force_rebuild", False)),
        show_progress=bool(competition.get("show_progress", True)),
    )
    prepared = materialize_competition_test_input(
        test_checkpoint, root / config["data"]["interim_train"], root / competition["prepared_input"],
        product_names=product_names, feature_names=feature_names,
        history_date=str(competition["history_date"]), memory_limit=runtime.get("memory_limit"),
        temp_directory=runtime.get("temp_directory"),
    )
    required_columns = ["ncodpers", *sorted({"prev_" + product for product in product_names}), *feature_names]
    prepared_frame = load_competition_input(
        prepared, columns=list(dict.fromkeys(required_columns)), dtypes=expected_dtypes,
    )
    sample_submission = root / competition["sample_submission"]
    if not sample_submission.is_file():
        if not bool(competition.get("generate_template_if_missing", False)):
            raise FileNotFoundError(
                f"sample_submission.csv is missing: {sample_submission}. "
                "Upload the official template or set competition.generate_template_if_missing=true "
                "to create a schema-compatible template from test_ver2 row order."
            )
        sample_submission = run_directory / "generated_sample_submission.csv"
        pd.DataFrame(
            {"ncodpers": prepared_frame["ncodpers"].to_numpy(), "added_products": ""}
        ).to_csv(sample_submission, index=False)
    submission = run_competition_test_inference(
        prepared_frame, sample_submission, artifact_directories,
        run_directory / "submission.csv", top_k=int(competition.get("top_k", 7)),
    )
    (run_directory / "submission_manifest.json").write_text(
        json.dumps(
            {
                "model_index": str(index),
                "run_config": str(resolved_config),
                "prepared_input": str(prepared),
                "sample_submission": str(sample_submission),
                "submission": str(submission),
                "history_date": str(competition["history_date"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return submission
