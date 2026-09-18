"""Optional MLflow adapter; imported only when experiment tracking is enabled."""
from __future__ import annotations

from pathlib import Path
from numbers import Number
from typing import Any, Mapping


def log_baseline_run(*, experiment_name: str, tracking_uri: str, manifest_path: str | Path, config_path: str | Path, artifact_dir: str | Path, metrics: Mapping[str, float], tags: Mapping[str, Any]) -> str:
    try:
        import mlflow
    except ImportError as error:  # pragma: no cover - depends on optional package
        raise RuntimeError("MLflow tracking is enabled but mlflow is not installed. Run: pip install -r requirements.txt") from error
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.set_tags({key: str(value) for key, value in tags.items() if value is not None})
        mlflow.log_params({"config_path": str(config_path)})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items() if isinstance(value, Number)})
        mlflow.log_artifact(str(config_path), artifact_path="config")
        mlflow.log_artifact(str(manifest_path), artifact_path="lineage")
        mlflow.log_artifacts(str(artifact_dir), artifact_path="artifacts")
        return run.info.run_id
