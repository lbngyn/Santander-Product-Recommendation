"""Leakage-safe panel inputs for the 24 product-acquisition classifiers."""
from __future__ import annotations

from pathlib import Path

import duckdb

from src.features.customer_history import acquisition_label_sql, product_event_count_sql, quote


def build_model_panel(source_path: str | Path, destination_path: str | Path, *, force_process: bool = False, memory_limit: str | None = None, temp_directory: str | Path | None = None) -> Path:
    """Materialise sample ``t`` using profiles at ``t`` and history before it.

    Acquisition/drop events and derived RFM features are valid only where the
    preceding customer record is exactly one month earlier. Gapped records stay
    observable (via ``record_gap_months``), but are not labelled transitions.
    """
    source, destination = Path(source_path), Path(destination_path)
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
        profiles = [column for column in columns if column not in required and column not in products]
        profile_sql = ", ".join(quote(column) for column in profiles)
        previous_states = ", ".join(f"CAST(COALESCE({quote('previous_' + product)}, 0) AS TINYINT) AS {quote('prev_' + product)}" for product in products)
        previous_raw = ", ".join(f"LAG({quote(product)}) OVER customer_time AS {quote('previous_' + product)}" for product in products)
        labels = ", ".join(f"{acquisition_label_sql(product)} AS {quote('acq_' + product)}" for product in products)
        acquisition_events = product_event_count_sql(products, event="acquisition")
        drop_events = product_event_count_sql(products, event="drop")
        owned_count = " + ".join(f"COALESCE({quote('previous_' + product)}, 0)" for product in products)
        source_sql = str(source).replace("'", "''")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        temporary_sql = str(temporary).replace("'", "''")
        con.execute(f"""
            COPY (
                WITH ordered AS (
                    SELECT *, LAG(fecha_dato) OVER customer_time AS previous_date,
                           COUNT(*) OVER (customer_time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS customer_history_length,
                           {previous_raw}
                    FROM read_parquet('{source_sql}')
                    WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato)
                ), transitions AS (
                    SELECT *, CASE WHEN previous_date IS NULL THEN NULL ELSE date_diff('month', CAST(previous_date AS DATE), CAST(fecha_dato AS DATE)) END AS record_gap_months
                    FROM ordered
                ), events AS (
                    SELECT *, {acquisition_events} AS acquisition_events, {drop_events} AS drop_events FROM transitions
                )
                SELECT ncodpers, fecha_dato{', ' if profile_sql else ''}{profile_sql},
                       CAST(previous_date IS NOT NULL AS TINYINT) AS previous_observation,
                       CAST(record_gap_months AS INTEGER) AS record_gap_months,
                       CAST(customer_history_length AS INTEGER) AS customer_history_length,
                       CAST({owned_count} AS TINYINT) AS products_owned_count,
                       {previous_states},
                       CAST(COALESCE(SUM(acquisition_events) OVER prior_rows, 0) AS INTEGER) AS cumulative_acquisitions,
                       CAST(COALESCE(SUM(drop_events) OVER prior_rows, 0) AS INTEGER) AS cumulative_drops,
                       CAST(date_diff('month', MAX(CASE WHEN acquisition_events > 0 THEN CAST(fecha_dato AS DATE) END) OVER prior_rows, CAST(fecha_dato AS DATE)) AS INTEGER) AS months_since_last_acquisition,
                       CAST(COALESCE(MAX(CASE WHEN acquisition_events > 0 THEN 1 ELSE 0 END) OVER prior_rows, 0) = 0 AS TINYINT) AS never_acquired_before,
                       CAST(COALESCE(SUM(acquisition_events) OVER recent_1m, 0) AS INTEGER) AS acquisitions_last_1m,
                       CAST(COALESCE(SUM(acquisition_events) OVER recent_3m, 0) AS INTEGER) AS acquisitions_last_3m,
                       CAST(COALESCE(SUM(acquisition_events) OVER recent_6m, 0) AS INTEGER) AS acquisitions_last_6m,
                       {labels}
                FROM events
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
