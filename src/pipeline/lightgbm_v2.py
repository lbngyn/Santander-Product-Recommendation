"""History-feature v2 of the 24 independent LightGBM pipeline.

Pipeline v1 remains an immutable baseline.  This module selects the v2 panel
and nearest-observed-record transition semantics while reusing the shared orchestration
and submission contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from src.features.feature_store import ensure_customer_month_feature_store
from src.features.model_panel import build_model_panel_from_feature_store
from src.features.persona import CATEGORICAL_PERSONA_FEATURES
from src.pipeline.lightgbm_v1 import (
    run_lightgbm_v1,
    run_lightgbm_v1_competition_from_config,
)


def run_lightgbm_v2_from_config(config_path: str | Path = "configs/baselines/lightgbm_v2.yaml") -> dict[str, Any]:
    """Load the v2 config and train 24 history-feature models."""
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("LightGBM v2 config must be a YAML mapping.")
    return run_lightgbm_v2(config, resolved_config_path=path)


def run_lightgbm_v2(config: Mapping[str, Any], *, resolved_config_path: str | Path) -> dict[str, Any]:
    """Build the v2 feature panel and train on nearest-record acquisition events."""
    declared_feature_store = config.get("data", {}).get("feature_store")

    def panel_builder(source: str | Path, destination: str | Path, **kwargs: Any) -> Path:
        return build_lightgbm_v2_panel(source, destination, feature_store_path=declared_feature_store, **kwargs)

    return run_lightgbm_v1(
        config,
        resolved_config_path=resolved_config_path,
        panel_builder=panel_builder,
        require_adjacent_month=False,
        categorical_feature_names=list(CATEGORICAL_PERSONA_FEATURES),
    )


def build_lightgbm_v2_panel(source_path: str | Path, destination_path: str | Path, **kwargs: Any) -> Path:
    """Use shared persona/history checkpoints, then create V2 labels only."""
    declared_feature_store = kwargs.pop("feature_store_path", None)
    force_process = bool(kwargs.pop("force_process", False))
    memory_limit = kwargs.pop("memory_limit", None)
    temp_directory = kwargs.pop("temp_directory", None)
    if kwargs:
        raise TypeError(f"Unexpected V2 panel arguments: {sorted(kwargs)}")
    source, destination = Path(source_path), Path(destination_path)
    feature_store = Path(declared_feature_store) if declared_feature_store else _feature_store_path(source, destination)
    if declared_feature_store and not feature_store.is_absolute():
        feature_store = source.parent.parent / feature_store
    ensure_customer_month_feature_store(
        source, feature_store, force_process=force_process,
        memory_limit=memory_limit, temp_directory=temp_directory,
    )
    return build_model_panel_from_feature_store(
        feature_store, destination, force_process=force_process,
        memory_limit=memory_limit, temp_directory=temp_directory,
    )


def _feature_store_path(source: Path, destination: Path) -> Path:
    """Keep one shared processed checkpoint per data root, including in tests."""
    if source.parent.name == "interim":
        return source.parent.parent / "processed" / "customer_month_features.parquet"
    return destination.parent / "customer_month_features.parquet"


def run_lightgbm_v2_competition_from_config(model_index_path: str | Path, config_path: str | Path | None = None) -> Path:
    """Create a submission using artifacts produced by this v2 pipeline."""
    return run_lightgbm_v1_competition_from_config(model_index_path, config_path)
