"""Optional MLflow adapter; imported only when experiment tracking is enabled."""
from __future__ import annotations

import json
import math
from pathlib import Path
from numbers import Number
from typing import Any, Mapping


def log_baseline_run(*, experiment_name: str, tracking_uri: str, manifest_path: str | Path, config_path: str | Path, artifact_dir: str | Path, metrics: Mapping[str, float], tags: Mapping[str, Any]) -> str:
    try:
        import mlflow
    except ImportError as error:  # pragma: no cover - depends on optional package
        raise RuntimeError("MLflow tracking is enabled but mlflow is not installed. Run: pip install -r requirements.txt") from error
    mlflow.set_tracking_uri(_normalise_tracking_uri(tracking_uri))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.set_tags({key: str(value) for key, value in tags.items() if value is not None})
        mlflow.log_params({"config_path": str(config_path)})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items() if isinstance(value, Number)})
        mlflow.log_artifact(str(config_path), artifact_path="config")
        mlflow.log_artifact(str(manifest_path), artifact_path="lineage")
        mlflow.log_artifacts(str(artifact_dir), artifact_path="artifacts")
        return run.info.run_id


def collect_lightgbm_training_metrics(artifact_dir: str | Path) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Load per-product operational metrics and derive parent-run summaries.

    These are training-resource metrics, not model-quality metrics. Quality
    metrics are intentionally absent until a temporal evaluator exists.
    """
    root = Path(artifact_dir) / "models"
    per_product: dict[str, dict[str, Any]] = {}
    for metrics_path in sorted(root.glob("*/training_metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            per_product[metrics_path.parent.name] = payload

    numeric = lambda value: isinstance(value, Number) and not isinstance(value, bool) and math.isfinite(float(value))
    rows = list(per_product.values())
    eligible = sum(float(row["eligible_rows"]) for row in rows if numeric(row.get("eligible_rows")))
    acquisitions = sum(float(row["acquisitions"]) for row in rows if numeric(row.get("acquisitions")))
    durations = [float(row["duration_seconds"]) for row in rows if numeric(row.get("duration_seconds"))]
    peaks = [float(row["rss_peak_during_fit_bytes"]) for row in rows if numeric(row.get("rss_peak_during_fit_bytes"))]
    summary: dict[str, float] = {
        "products_trained": float(len(per_product)),
        "eligible_rows_total": eligible,
        "acquisitions_total": acquisitions,
        "positive_rate_weighted": acquisitions / eligible if eligible else 0.0,
        "train_duration_seconds_total": sum(durations),
        "train_duration_seconds_max": max(durations, default=0.0),
        "rss_peak_during_fit_bytes_max": max(peaks, default=0.0),
    }
    return summary, per_product


def log_lightgbm_run(
    *,
    experiment_name: str,
    tracking_uri: str,
    manifest_path: str | Path,
    config_path: str | Path,
    artifact_dir: str | Path,
    model_config: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    tags: Mapping[str, Any],
) -> str:
    """Log one parent pipeline run and one nested run per product model."""
    try:
        import mlflow
    except ImportError as error:  # pragma: no cover - depends on optional package
        raise RuntimeError("MLflow tracking is enabled but mlflow is not installed. Run: pip install -r requirements.txt") from error

    parent_metrics, per_product = collect_lightgbm_training_metrics(artifact_dir)
    parent_metrics.update(_numeric_metrics(validation_metrics))
    mlflow.set_tracking_uri(_normalise_tracking_uri(tracking_uri))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=str(tags.get("run_id", "lightgbm-pipeline"))) as parent:
        parent_tags = {
            **{key: str(value) for key, value in tags.items() if value is not None},
            "tracking_scope": "pipeline",
            "validation_status": "complete" if validation_metrics else "not_configured",
        }
        mlflow.set_tags(parent_tags)
        mlflow.log_params({
            "config_path": str(config_path),
            "n_estimators": model_config.get("n_estimators"),
            "random_state": model_config.get("random_state"),
            "negative_sampling_strategy": model_config.get("negative_sampling_strategy", "head"),
            "max_negative_rows_per_product": model_config.get("max_negative_rows_per_product"),
            "max_total_rows_per_product": model_config.get("max_total_rows_per_product"),
        })
        mlflow.log_metrics(parent_metrics)
        mlflow.log_artifact(str(config_path), artifact_path="config")
        mlflow.log_artifact(str(manifest_path), artifact_path="lineage")

        for product, metrics in per_product.items():
            numeric_metrics = _numeric_metrics(metrics)
            with mlflow.start_run(run_name=product, nested=True):
                mlflow.set_tags({"product": product, "tracking_scope": "product-model"})
                mlflow.log_params({
                    "negative_sampling_strategy": metrics.get("negative_sampling_strategy", "head"),
                    "negative_sampling_seed": metrics.get("negative_sampling_seed"),
                })
                mlflow.log_metrics(numeric_metrics)
                metrics_path = Path(artifact_dir) / "models" / product / "training_metrics.json"
                if metrics_path.is_file():
                    mlflow.log_artifact(str(metrics_path), artifact_path="training")

        # This keeps the entire model, schema, memory trace, resolved config
        # and lineage discoverable from the parent run without duplicating the
        # potentially large model pickle into every nested run.
        mlflow.log_artifacts(str(artifact_dir), artifact_path="artifacts")
        return parent.info.run_id


def _numeric_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, Number) and not isinstance(value, bool) and math.isfinite(float(value))
    }


def _normalise_tracking_uri(value: str) -> str:
    """Use SQLite for local tracking because MLflow no longer enables file stores."""
    if value.startswith("file:") and not value.startswith("file://"):
        database = Path(value.removeprefix("file:")).resolve().with_suffix(".db")
        return f"sqlite:///{database.as_posix()}"
    if value.startswith("file://"):
        database = Path(value.removeprefix("file://")).resolve().with_suffix(".db")
        return f"sqlite:///{database.as_posix()}"
    return value
