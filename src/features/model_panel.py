"""Leakage-safe panel inputs for the 24 product-acquisition classifiers."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import duckdb

from src.features.customer_history import acquisition_label_sql, product_event_count_sql, quote
from src.features.history_features import customer_history_length_window_sql, record_gap_months_sql
from src.features.persona import PERSONA_FEATURES, persona_feature_projection_sql, raw_persona_projection_sql


def build_model_panel(source_path: str | Path, destination_path: str | Path, *, history_feature_sql: Sequence[str], selected_rows_sql: str | None = None, selection_columns: Sequence[str] = (), force_process: bool = False, memory_limit: str | None = None, temp_directory: str | Path | None = None) -> Path:
    """Materialise sample ``t`` using profiles at ``t`` and history before it.

    Acquisition/drop events and derived RFM features are valid only where the
    preceding customer record is exactly one month earlier. Gapped records stay
    observable (via ``record_gap_months``), but are not labelled transitions.
    """
    source, destination = Path(source_path), Path(destination_path)
    if not history_feature_sql:
        raise ValueError("A pipeline must supply the selected history feature builders.")
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source}")
    if destination.is_file() and not force_process:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        if memory_limit:
            con.execute(f"SET memory_limit = '{memory_limit}'")
        if temp_directory:
            tmp = Path(temp_directory); tmp.mkdir(parents=True, exist_ok=True)
            con.execute(f"SET temp_directory = '{str(tmp).replace(chr(39), chr(39) * 2)}'")
        columns = [r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]).fetchall()]
        required = {"ncodpers", "fecha_dato"}
        if missing := required.difference(columns):
            raise ValueError(f"Source is missing required columns: {sorted(missing)}")
        products = [column for column in columns if column.endswith("_ult1")]
        if not products:
            raise ValueError("Source has no product columns ending in '_ult1'.")
        profile_sql = ", ".join(quote(column) for column in PERSONA_FEATURES)
        raw_persona_sql = raw_persona_projection_sql("source", columns)
        persona_sql = persona_feature_projection_sql()
        product_sql = ", ".join(f"source.{quote(product)}" for product in products)
        previous_states = ", ".join(f"CAST(COALESCE({quote('previous_' + product)}, 0) AS TINYINT) AS {quote('prev_' + product)}" for product in products)
        previous_raw = ", ".join(f"LAG({quote(product)}) OVER customer_time AS {quote('previous_' + product)}" for product in products)
        labels = ", ".join(f"{acquisition_label_sql(product)} AS {quote('acq_' + product)}" for product in products)
        acquisition_events = product_event_count_sql(products, event="acquisition")
        drop_events = product_event_count_sql(products, event="drop")
        selected_history_features = ",\n                       ".join(history_feature_sql)
        source_sql = str(source).replace("'", "''")
        selection_projection = ", ".join(f"selection.{quote(name)}" for name in selection_columns)
        selection_ctes = ""
        source_join = ""
        final_join = ""
        if selected_rows_sql:
            selection_sql = selected_rows_sql.replace("{source_path}", source_sql)
            selection_ctes = f"selection AS ({selection_sql}), selected_customers AS (SELECT DISTINCT ncodpers FROM selection),"
            source_join = "INNER JOIN selected_customers USING (ncodpers)"
            final_join = "INNER JOIN selection USING (ncodpers, fecha_dato)"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        temporary_sql = str(temporary).replace("'", "''")
        con.execute(f"""
            COPY (
                WITH {selection_ctes} source_rows AS (
                    SELECT source.ncodpers, source.fecha_dato, {raw_persona_sql}, {product_sql}
                    FROM read_parquet('{source_sql}') AS source {source_join}
                ), ordered AS (
                    SELECT *, {persona_sql}, LAG(fecha_dato) OVER customer_time AS previous_date,
                           {customer_history_length_window_sql()},
                           {previous_raw}
                    FROM source_rows
                    WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato),
                           prior_customer_rows AS (PARTITION BY ncodpers ORDER BY fecha_dato ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                ), transitions AS (
                    SELECT *, {record_gap_months_sql()}
                    FROM ordered
                ), events AS (
                    SELECT *, {acquisition_events} AS acquisition_events, {drop_events} AS drop_events FROM transitions
                )
                SELECT ncodpers, fecha_dato{', ' if profile_sql else ''}{profile_sql},
                       CAST(previous_date IS NOT NULL AS TINYINT) AS previous_observation,
                       {previous_states},
                       {selected_history_features},
                       {labels}{', ' if selection_projection else ''}{selection_projection}
                FROM events {final_join}
                WINDOW prior_rows AS (PARTITION BY ncodpers ORDER BY fecha_dato ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                       recent_1m AS (PARTITION BY ncodpers ORDER BY CAST(fecha_dato AS DATE) RANGE BETWEEN INTERVAL 1 MONTH PRECEDING AND INTERVAL 1 DAY PRECEDING),
                       recent_3m AS (PARTITION BY ncodpers ORDER BY CAST(fecha_dato AS DATE) RANGE BETWEEN INTERVAL 3 MONTH PRECEDING AND INTERVAL 1 DAY PRECEDING),
                       recent_6m AS (PARTITION BY ncodpers ORDER BY CAST(fecha_dato AS DATE) RANGE BETWEEN INTERVAL 6 MONTH PRECEDING AND INTERVAL 1 DAY PRECEDING)
            ) TO '{temporary_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        temporary.replace(destination)
        return destination
    finally:
        con.close()
