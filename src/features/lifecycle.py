"""Minimal offline customer-lifecycle feature materialisation.

Lifecycle churn is a delayed, analytical label: confirming a six-month
absence necessarily requires waiting through those six months.  This module
therefore writes a separate lifecycle artifact and must not be joined into a
point-in-time acquisition model as an input feature.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from src.features.customer_history import quote


LIFECYCLE_FEATURE_NAMES: tuple[str, ...] = (
    "record_type",
    "lifecycle_tier",
    "relationship_exit_signal",
    "churn_confirmed_month",
    "first_winback_date",
)


def materialize_customer_lifecycle(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    churn_absence_months: int = 6,
    force_process: bool = False,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
) -> Path:
    """Write observed lifecycle rows plus one synthetic churn event per spell.

    The final lifecycle rules from the EDA are intentionally minimal:

    * ``New``: observed ``ind_nuevo = 1`` before a confirmed churn episode;
    * ``At-Risk``: an observed relationship-exit marker;
    * ``Churned``: absence for ``churn_absence_months`` (sticky until recovery);
    * ``Win-Back``: first post-churn row with behavioural recovery or a cleared
      relationship-exit marker;
    * ``Active``: all remaining observed rows.

    Churn is emitted only after the future absence horizon has been observed,
    and is represented by one synthetic row at ``last_seen + K months``.  This
    output is suitable for lifecycle reporting and delayed-label analysis, not
    as a same-month predictive-model feature.
    """
    if churn_absence_months < 1:
        raise ValueError("churn_absence_months must be positive.")
    source, destination = Path(source_path), Path(destination_path)
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    con = _connect(memory_limit, temp_directory)
    try:
        columns = _columns(con, source)
        if missing := {"ncodpers", "fecha_dato"}.difference(columns):
            raise ValueError(f"Source is missing lifecycle keys: {sorted(missing)}")
        products = [name for name in columns if name.endswith("_ult1") and not name.startswith(("prev_", "acq_"))]
        if not products:
            raise ValueError("Source has no product columns ending in '_ult1'.")
        required_output = {"ncodpers", "fecha_dato", *LIFECYCLE_FEATURE_NAMES}
        if destination.is_file() and not force_process and required_output.issubset(_columns(con, destination)):
            return destination

        raw = lambda name, sql_type: f"TRY_CAST({quote(name)} AS {sql_type})" if name in columns else f"NULL::{sql_type}"
        text = lambda name: f"NULLIF(TRIM(CAST({quote(name)} AS VARCHAR)), '')" if name in columns else "NULL::VARCHAR"
        product_current = ", ".join(f"TRY_CAST({quote(product)} AS INTEGER) AS {quote(product)}" for product in products)
        product_lag = ", ".join(f"LAG({quote(product)}) OVER customer_time AS {quote('previous_' + product)}" for product in products)
        portfolio = " + ".join(f"CASE WHEN {quote(product)} = 1 THEN 1 ELSE 0 END" for product in products)
        acquisitions = " + ".join(
            f"CASE WHEN previous_date IS NOT NULL AND COALESCE({quote('previous_' + product)}, 0) = 0 "
            f"AND COALESCE({quote(product)}, 0) = 1 THEN 1 ELSE 0 END"
            for product in products
        )
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        con.execute(f"""
            COPY (
                WITH source_rows AS (
                    SELECT CAST(ncodpers AS BIGINT) AS ncodpers,
                           CAST(fecha_dato AS DATE) AS fecha_dato,
                           {raw('ind_nuevo', 'INTEGER')} AS is_new_customer,
                           {raw('ind_actividad_cliente', 'INTEGER')} AS active,
                           {raw('indrel', 'INTEGER')} AS indrel,
                           {text('tiprel_1mes')} AS tiprel_1mes,
                           {text('indrel_1mes')} AS indrel_1mes,
                           {raw('ult_fec_cli_1t', 'DATE')} AS ult_fec_cli_1t,
                           {text('indfall')} AS indfall,
                           {product_current}
                    FROM read_parquet('{_path(source)}')
                ), ordered AS (
                    SELECT *, LAG(fecha_dato) OVER customer_time AS previous_date,
                           LAG(active) OVER customer_time AS previous_active,
                           LAG(({portfolio})) OVER customer_time AS previous_portfolio_size,
                           LEAD(fecha_dato) OVER customer_time AS next_date,
                           {product_lag}
                    FROM source_rows
                    WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato)
                ), events AS (
                    SELECT *, date_diff('month', previous_date, fecha_dato) AS record_gap_months,
                           ({portfolio})::INTEGER AS portfolio_size
                    FROM ordered
                ), lifecycle_signals AS (
                    SELECT *, ({acquisitions})::INTEGER AS n_acquisition,
                           CASE WHEN TRY_CAST(indrel_1mes AS DOUBLE) IS NOT NULL
                                THEN CAST(CAST(TRY_CAST(indrel_1mes AS DOUBLE) AS INTEGER) AS VARCHAR)
                                ELSE UPPER(indrel_1mes) END AS indrel_1mes_norm
                    FROM events
                ), observed AS (
                    SELECT *, CASE WHEN tiprel_1mes = 'P' OR indrel = 99
                                             OR indrel_1mes_norm IN ('3', '4')
                                             OR ult_fec_cli_1t IS NOT NULL THEN 1 ELSE 0 END AS relationship_exit_signal
                    FROM lifecycle_signals
                ), bounds AS (
                    SELECT MAX(fecha_dato) AS max_date FROM observed
                ), qualifying_churn_events AS (
                    SELECT o.ncodpers, o.fecha_dato AS last_seen_before_churn,
                           CAST(o.fecha_dato + INTERVAL '{churn_absence_months} months' AS DATE) AS churn_confirmed_month,
                           o.active AS active_before_churn, o.portfolio_size AS portfolio_before_churn
                    FROM observed o CROSS JOIN bounds b
                    WHERE COALESCE(o.indfall, 'N') <> 'S'
                      AND ((o.next_date IS NOT NULL AND date_diff('month', o.fecha_dato, o.next_date) - 1 >= {churn_absence_months})
                           OR (o.next_date IS NULL AND date_diff('month', o.fecha_dato, b.max_date) >= {churn_absence_months}))
                ), observed_with_episode AS (
                    SELECT o.*, e.churn_confirmed_month, e.last_seen_before_churn,
                           e.active_before_churn, e.portfolio_before_churn,
                           CASE WHEN e.churn_confirmed_month IS NOT NULL AND
                                     (o.n_acquisition > 0 OR (e.active_before_churn = 0 AND o.active = 1)
                                      OR o.portfolio_size > e.portfolio_before_churn) THEN 1 ELSE 0 END AS has_behavioral_reactivation,
                           CASE WHEN e.churn_confirmed_month IS NOT NULL AND o.relationship_exit_signal = 0 THEN 1 ELSE 0 END AS has_relationship_recovery
                    FROM observed o
                    LEFT JOIN LATERAL (
                        SELECT * FROM qualifying_churn_events e
                        WHERE e.ncodpers = o.ncodpers AND e.churn_confirmed_month < o.fecha_dato
                        ORDER BY e.churn_confirmed_month DESC LIMIT 1
                    ) e ON TRUE
                ), observed_with_winback AS (
                    SELECT *, MIN(CASE WHEN churn_confirmed_month IS NOT NULL
                                             AND (has_behavioral_reactivation = 1 OR has_relationship_recovery = 1)
                                        THEN fecha_dato END) OVER (PARTITION BY ncodpers, churn_confirmed_month) AS first_winback_date
                    FROM observed_with_episode
                ), lifecycle_rows AS (
                    SELECT ncodpers, fecha_dato, 'observed_snapshot'::VARCHAR AS record_type,
                           relationship_exit_signal, churn_confirmed_month, first_winback_date,
                           CASE WHEN churn_confirmed_month IS NOT NULL
                                      AND (first_winback_date IS NULL OR fecha_dato < first_winback_date) THEN 'Churned'
                                WHEN churn_confirmed_month IS NOT NULL AND fecha_dato = first_winback_date THEN 'Win-Back'
                                WHEN churn_confirmed_month IS NULL AND COALESCE(is_new_customer, 0) = 1 THEN 'New'
                                WHEN relationship_exit_signal = 1 THEN 'At-Risk'
                                ELSE 'Active' END AS lifecycle_tier
                    FROM observed_with_winback
                ), synthetic_churn_rows AS (
                    SELECT ncodpers, churn_confirmed_month AS fecha_dato,
                           'synthetic_churn_event'::VARCHAR AS record_type,
                           NULL::INTEGER AS relationship_exit_signal, churn_confirmed_month,
                           NULL::DATE AS first_winback_date, 'Churned'::VARCHAR AS lifecycle_tier
                    FROM qualifying_churn_events
                )
                SELECT * FROM lifecycle_rows
                UNION ALL SELECT * FROM synthetic_churn_rows
            ) TO '{_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        duplicate_keys = con.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT ncodpers, fecha_dato FROM read_parquet('{_path(temporary)}')
                GROUP BY 1, 2 HAVING COUNT(*) > 1
            )
        """).fetchone()[0]
        if duplicate_keys:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Lifecycle output contains {duplicate_keys} duplicate customer-month keys.")
        temporary.replace(destination)
        return destination
    finally:
        con.close()


def _connect(memory_limit: str | None, temp_directory: str | Path | None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    if memory_limit:
        con.execute(f"SET memory_limit = '{str(memory_limit).replace(chr(39), chr(39) * 2)}'")
    if temp_directory:
        directory = Path(temp_directory); directory.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{_path(directory)}'")
    return con


def _columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    return {row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()}


def _path(path: Path) -> str:
    return str(path).replace("'", "''")
