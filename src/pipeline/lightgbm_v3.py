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
    selection_config = kwargs.pop("selection_config", None)
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
    selected_rows_sql, selection_columns = _selected_rows_sql(products, selection_config) if selection_config else (None, ())
    return build_model_panel(source_path, destination_path, history_feature_sql=history_features, selected_rows_sql=selected_rows_sql, selection_columns=selection_columns, **kwargs)


def _selected_rows_sql(products: list[str], config: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Select V3 train keys before feature materialisation; history remains context."""
    validation_date = str(config["validation_date"]).replace("'", "''")
    negative_limit = int(config["max_negative_rows_per_product"])
    seed = int(config["random_state"])
    values = ", ".join(f"('{product}', COALESCE(TRY_CAST(\"{product}\" AS TINYINT), 0), COALESCE(TRY_CAST(\"previous_{product}\" AS TINYINT), 0), {index})" for index, product in enumerate(products, start=1))
    flags = tuple("sampled_for_" + product for product in products)
    flag_sql = ", ".join(f"CAST(MAX(CASE WHEN product = '{product}' AND is_negative = 1 THEN 1 ELSE 0 END) AS TINYINT) AS \"sampled_for_{product}\"" for product in products)
    sql = f"""
        WITH ordered AS (
            SELECT ncodpers, fecha_dato, LAG(fecha_dato) OVER customer_time AS previous_date,
                   {', '.join(f'\"{product}\", LAG(\"{product}\") OVER customer_time AS \"previous_{product}\"' for product in products)}
            FROM read_parquet('{{source_path}}')
            WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato)
        ), candidate_events AS (
            SELECT ncodpers, fecha_dato, product,
                   CAST(current_value = 1 AND previous_value = 0 AS TINYINT) AS label,
                   product_ordinal
            FROM ordered CROSS JOIN LATERAL (VALUES {values}) AS v(product, current_value, previous_value, product_ordinal)
            WHERE date_diff('month', CAST(previous_date AS DATE), CAST(fecha_dato AS DATE)) = 1
              AND previous_value = 0 AND CAST(fecha_dato AS DATE) < CAST('{validation_date}' AS DATE)
        ), sampled AS (
            SELECT *, CASE WHEN label = 0 AND ROW_NUMBER() OVER (PARTITION BY product ORDER BY hash(ncodpers, fecha_dato, {seed} + product_ordinal), ncodpers, fecha_dato) <= {negative_limit} THEN 1 ELSE 0 END AS is_negative
            FROM candidate_events
        ), selected_train AS (
            SELECT ncodpers, fecha_dato, product, is_negative FROM sampled WHERE label = 1 OR is_negative = 1
        ), validation_keys AS (
            SELECT ncodpers, fecha_dato, NULL::VARCHAR AS product, 0 AS is_negative
            FROM ordered WHERE date_diff('month', CAST(previous_date AS DATE), CAST(fecha_dato AS DATE)) = 1 AND CAST(fecha_dato AS DATE) = CAST('{validation_date}' AS DATE)
        )
        SELECT ncodpers, fecha_dato, {flag_sql}
        FROM (SELECT * FROM selected_train UNION ALL SELECT * FROM validation_keys)
        GROUP BY ncodpers, fecha_dato
    """
    return sql, flags


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
