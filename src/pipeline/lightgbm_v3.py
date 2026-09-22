"""Persona-and-history LightGBM acquisition pipeline, version 3.

Version 3 is an independently reproducible experiment that keeps the v2
acquisition target and adjacent-month semantics, while making the feature
contract explicit: canonical persona features at snapshot ``t`` are combined
with history-only portfolio and event features available strictly before ``t``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

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
from src.features.model_panel import build_model_panel
from src.features.persona import CATEGORICAL_PERSONA_FEATURES, PERSONA_FEATURES
from src.pipeline.lightgbm_v1 import (
    run_lightgbm_v1,
    run_lightgbm_v1_competition_from_config,
)


def run_lightgbm_v3_from_config(config_path: str | Path = "configs/baselines/lightgbm_v3.yaml") -> dict[str, Any]:
    """Load the v3 config and train the 24 acquisition classifiers."""
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("LightGBM v3 config must be a YAML mapping.")
    return run_lightgbm_v3(config, resolved_config_path=path)


def run_lightgbm_v3(config: Mapping[str, Any], *, resolved_config_path: str | Path) -> dict[str, Any]:
    """Build the v3 persona/history panel and train valid monthly transitions."""
    return run_lightgbm_v1(
        config,
        resolved_config_path=resolved_config_path,
        panel_builder=build_lightgbm_v3_panel,
        require_adjacent_month=True,
        categorical_feature_names=list(CATEGORICAL_PERSONA_FEATURES),
    )


def build_lightgbm_v3_panel(source_path: str | Path, destination_path: str | Path, **kwargs: Any) -> Path:
    """Materialise one panel with canonical persona and past-only history features.

    ``build_model_panel`` owns the feature join: it projects every value in
    :data:`PERSONA_FEATURES` for the current snapshot and evaluates the SQL
    expressions below over windows ending at the preceding customer record.
    """
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
    """Read only the Parquet schema to configure product-dependent features."""
    import duckdb

    con = duckdb.connect(database=":memory:")
    try:
        return [
            row[0]
            for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(source_path)]).fetchall()
            if row[0].endswith("_ult1")
        ]
    finally:
        con.close()


def run_lightgbm_v3_competition_from_config(
    model_index_path: str | Path, config_path: str | Path | None = None
) -> Path:
    """Create a submission using immutable artifacts from a v3 training run."""
    return run_lightgbm_v1_competition_from_config(model_index_path, config_path)
