"""History-feature v2 of the 24 independent LightGBM pipeline.

Pipeline v1 remains an immutable baseline.  This module selects the v2 panel
and adjacent-month transition semantics while reusing the shared orchestration
and submission contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from src.features.model_panel import build_model_panel
from src.features.persona import CATEGORICAL_PERSONA_FEATURES
from src.features.history_features import (
    acquisitions_last_1m_sql,
    acquisitions_last_3m_sql,
    acquisitions_last_6m_sql,
    cumulative_acquisitions_sql,
    cumulative_drops_sql,
    customer_history_length_sql,
    months_since_last_acquisition_sql,
    never_acquired_before_sql,
    products_owned_count_sql,
    record_gap_months_sql,
)
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
    """Build the v2 feature panel and train on valid adjacent-month events."""
    return run_lightgbm_v1(
        config,
        resolved_config_path=resolved_config_path,
        panel_builder=build_lightgbm_v2_panel,
        require_adjacent_month=True,
        categorical_feature_names=list(CATEGORICAL_PERSONA_FEATURES),
    )


def build_lightgbm_v2_panel(source_path: str | Path, destination_path: str | Path, **kwargs: Any) -> Path:
    """Compose v2's individual feature builders, then materialize one panel."""
    products = _product_columns(source_path)
    history_features = [
        record_gap_months_sql(output=True),
        customer_history_length_sql(),
        products_owned_count_sql(products),
        cumulative_acquisitions_sql(),
        months_since_last_acquisition_sql(),
        never_acquired_before_sql(),
        acquisitions_last_1m_sql(),
        acquisitions_last_3m_sql(),
        acquisitions_last_6m_sql(),
        cumulative_drops_sql(),
    ]
    return build_model_panel(source_path, destination_path, history_feature_sql=history_features, **kwargs)


def _product_columns(source_path: str | Path) -> list[str]:
    """Read only the source schema to wire product-dependent feature builders."""
    import duckdb

    con = duckdb.connect(database=":memory:")
    try:
        return [row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(source_path)]).fetchall() if row[0].endswith("_ult1")]
    finally:
        con.close()


def run_lightgbm_v2_competition_from_config(model_index_path: str | Path, config_path: str | Path | None = None) -> Path:
    """Create a submission using artifacts produced by this v2 pipeline."""
    return run_lightgbm_v1_competition_from_config(model_index_path, config_path)
