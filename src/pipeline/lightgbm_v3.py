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

from src.features.feature_store import ensure_customer_month_feature_store
from src.features.model_panel import build_model_panel_from_feature_store
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
    declared_feature_store = config.get("data", {}).get("feature_store")

    def panel_builder(source: str | Path, destination: str | Path, **kwargs: Any) -> Path:
        return build_lightgbm_v3_panel(source, destination, feature_store_path=declared_feature_store, **kwargs)

    return run_lightgbm_v1(
        config,
        resolved_config_path=resolved_config_path,
        panel_builder=panel_builder,
        require_adjacent_month=True,
        categorical_feature_names=list(CATEGORICAL_PERSONA_FEATURES),
    )


def build_lightgbm_v3_panel(source_path: str | Path, destination_path: str | Path, **kwargs: Any) -> Path:
    """Materialise one panel with canonical persona and past-only history features.

    Persona and history are first read from the common customer-month feature
    store.  This function owns only V3's model-specific selection flags.
    """
    source, destination = Path(source_path), Path(destination_path)
    selection_config = kwargs.pop("selection_config", None)
    declared_feature_store = kwargs.pop("feature_store_path", None)
    force_process = bool(kwargs.pop("force_process", False))
    memory_limit = kwargs.pop("memory_limit", None)
    temp_directory = kwargs.pop("temp_directory", None)
    if kwargs:
        raise TypeError(f"Unexpected V3 panel arguments: {sorted(kwargs)}")
    feature_store = Path(declared_feature_store) if declared_feature_store else _feature_store_path(source, destination)
    if declared_feature_store and not feature_store.is_absolute():
        feature_store = source.parent.parent / feature_store
    ensure_customer_month_feature_store(
        source, feature_store, force_process=force_process,
        memory_limit=memory_limit, temp_directory=temp_directory,
    )
    products = _product_columns(feature_store)
    selected_rows_sql, selection_columns = _selected_rows_sql(products, selection_config) if selection_config else (None, ())
    return build_model_panel_from_feature_store(
        feature_store, destination, selected_rows_sql=selected_rows_sql,
        selection_columns=selection_columns, force_process=force_process,
        memory_limit=memory_limit, temp_directory=temp_directory,
    )


def _selected_rows_sql(products: list[str], config: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Select V3 train keys before feature materialisation; history remains context."""
    validation_date = str(config["validation_date"]).replace("'", "''")
    negative_limit = int(config["max_negative_rows_per_product"])
    seed = int(config["random_state"])
    values = ", ".join(f"('{product}', COALESCE(TRY_CAST(\"{product}\" AS TINYINT), 0), COALESCE(TRY_CAST(\"prev_{product}\" AS TINYINT), 0), {index})" for index, product in enumerate(products, start=1))
    flags = tuple("sampled_for_" + product for product in products)
    flag_sql = ", ".join(f"CAST(MAX(CASE WHEN product = '{product}' AND is_negative = 1 THEN 1 ELSE 0 END) AS TINYINT) AS \"sampled_for_{product}\"" for product in products)
    sql = f"""
        WITH source_rows AS (
            SELECT * FROM read_parquet('{{source_path}}')
        ), candidate_events AS (
            SELECT ncodpers, fecha_dato, product,
                   CAST(current_value = 1 AND previous_value = 0 AS TINYINT) AS label,
                   product_ordinal
            FROM source_rows CROSS JOIN LATERAL (VALUES {values}) AS v(product, current_value, previous_value, product_ordinal)
            WHERE record_gap_months = 1
              AND previous_value = 0 AND CAST(fecha_dato AS DATE) < CAST('{validation_date}' AS DATE)
        ), sampled AS (
            SELECT *, CASE WHEN label = 0 AND ROW_NUMBER() OVER (PARTITION BY product ORDER BY hash(ncodpers, fecha_dato, {seed} + product_ordinal), ncodpers, fecha_dato) <= {negative_limit} THEN 1 ELSE 0 END AS is_negative
            FROM candidate_events
        ), selected_train AS (
            SELECT ncodpers, fecha_dato, product, is_negative FROM sampled WHERE label = 1 OR is_negative = 1
        ), validation_keys AS (
            SELECT ncodpers, fecha_dato, NULL::VARCHAR AS product, 0 AS is_negative
            FROM source_rows WHERE record_gap_months = 1 AND CAST(fecha_dato AS DATE) = CAST('{validation_date}' AS DATE)
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
            if row[0].endswith("_ult1") and not row[0].startswith(("prev_", "acq_"))
        ]
    finally:
        con.close()


def _feature_store_path(source: Path, destination: Path) -> Path:
    if source.parent.name == "interim":
        return source.parent.parent / "processed" / "customer_month_features.parquet"
    return destination.parent / "customer_month_features.parquet"


def run_lightgbm_v3_competition_from_config(
    model_index_path: str | Path, config_path: str | Path | None = None
) -> Path:
    """Create a submission using immutable artifacts from a v3 training run."""
    return run_lightgbm_v1_competition_from_config(model_index_path, config_path)
