"""Leakage-safe preparation of Santander's June-2016 competition test input."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import duckdb

from src.features.customer_history import HISTORY_FEATURE_NAMES, product_event_count_sql


def materialize_competition_test_input(
    test_checkpoint_path: str | Path,
    train_history_path: str | Path,
    destination_path: str | Path,
    *,
    product_names: Sequence[str],
    feature_names: Sequence[str],
    history_date: str = "2016-05-28",
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
) -> Path:
    """Write model input with June profiles and ownership state at May 2016.

    ``test_ver2.csv`` has customer/profile values for the prediction month but
    no product flags.  Product state is therefore joined only from the latest
    train observation on or before ``history_date``; no test-month product
    state can enter the input.
    """
    test, history, destination = Path(test_checkpoint_path), Path(train_history_path), Path(destination_path)
    if not test.is_file() or not history.is_file():
        missing = [str(path) for path in (test, history) if not path.is_file()]
        raise FileNotFoundError(f"Missing competition input source(s): {missing}")
    if len(product_names) != 24 or len(set(product_names)) != 24:
        raise ValueError("Competition inference requires exactly 24 unique product names.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        _configure_duckdb(con, memory_limit, temp_directory)
        test_columns = _columns(con, test)
        history_columns = _columns(con, history)
        required_test = {"ncodpers", "fecha_dato"}
        missing_test = required_test.difference(test_columns)
        missing_history = {"ncodpers", "fecha_dato", *product_names}.difference(history_columns)
        if missing_test or missing_history:
            raise ValueError(f"Input schema missing: test={sorted(missing_test)}, history={sorted(missing_history)}")
        history_features = set(HISTORY_FEATURE_NAMES)
        profile_features = [name for name in feature_names if not name.startswith("prev_") and name not in history_features]
        missing_features = set(profile_features).difference(test_columns)
        if missing_features:
            raise ValueError(f"Test checkpoint lacks model feature(s): {sorted(missing_features)}")
        if not bool(con.execute("SELECT count(*) = count(DISTINCT ncodpers) FROM read_parquet(?)", [str(test)]).fetchone()[0]):
            raise ValueError("Competition test must contain one row per ncodpers.")
        q = _quote
        profiles = ", ".join(f"test.{q(name)}" for name in profile_features)
        previous = ", ".join(
            f"CAST(COALESCE(history.{q(product)}, 0) AS TINYINT) AS {q('prev_' + product)}"
            for product in product_names
        )
        owned_count = " + ".join(f"COALESCE(history.{q(product)}, 0)" for product in product_names)
        previous_raw = ", ".join(
            f"LAG({q(product)}) OVER customer_time AS {q('previous_' + product)}" for product in product_names
        )
        acquisition_events = product_event_count_sql(product_names, event="acquisition")
        drop_events = product_event_count_sql(product_names, event="drop")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        history_sql = str(history).replace("'", "''")
        test_sql = str(test).replace("'", "''")
        temporary_sql = str(temporary).replace("'", "''")
        history_date_sql = history_date.replace("'", "''")
        con.execute(
            f"""
            COPY (
                WITH history_base AS (
                    SELECT * FROM read_parquet('{history_sql}')
                    WHERE CAST(fecha_dato AS DATE) <= CAST('{history_date_sql}' AS DATE)
                ), ordered AS (
                    SELECT *, LAG(fecha_dato) OVER customer_time AS previous_date,
                           {previous_raw}
                    FROM history_base
                    WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato)
                ), transitions AS (
                    SELECT *, CASE WHEN previous_date IS NULL THEN NULL
                                   ELSE date_diff('month', CAST(previous_date AS DATE), CAST(fecha_dato AS DATE)) END AS record_gap_months
                    FROM ordered
                ), events AS (
                    SELECT *, {acquisition_events} AS acquisition_events, {drop_events} AS drop_events FROM transitions
                ), latest_history AS (
                    SELECT * EXCLUDE (row_number) FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY ncodpers ORDER BY fecha_dato DESC) AS row_number FROM events
                    ) WHERE row_number = 1
                ), history_summary AS (
                    SELECT ncodpers, COUNT(*) AS customer_history_length,
                           COALESCE(SUM(acquisition_events), 0) AS cumulative_acquisitions,
                           COALESCE(SUM(drop_events), 0) AS cumulative_drops,
                           MAX(CASE WHEN acquisition_events > 0 THEN CAST(fecha_dato AS DATE) END) AS last_acquisition_date
                    FROM events GROUP BY ncodpers
                ), recent_acquisitions AS (
                    SELECT test.ncodpers,
                           COALESCE(SUM(CASE WHEN CAST(events.fecha_dato AS DATE) >= CAST(test.fecha_dato AS DATE) - INTERVAL 1 MONTH THEN events.acquisition_events ELSE 0 END), 0) AS acquisitions_last_1m,
                           COALESCE(SUM(CASE WHEN CAST(events.fecha_dato AS DATE) >= CAST(test.fecha_dato AS DATE) - INTERVAL 3 MONTH THEN events.acquisition_events ELSE 0 END), 0) AS acquisitions_last_3m,
                           COALESCE(SUM(CASE WHEN CAST(events.fecha_dato AS DATE) >= CAST(test.fecha_dato AS DATE) - INTERVAL 6 MONTH THEN events.acquisition_events ELSE 0 END), 0) AS acquisitions_last_6m
                    FROM read_parquet('{test_sql}') AS test
                    LEFT JOIN events ON test.ncodpers = events.ncodpers
                        AND CAST(events.fecha_dato AS DATE) < CAST(test.fecha_dato AS DATE)
                    GROUP BY test.ncodpers
                )
                SELECT test.ncodpers, test.fecha_dato{', ' if profiles else ''}{profiles}, {previous}
                       , CAST(date_diff('month', CAST(history.fecha_dato AS DATE), CAST(test.fecha_dato AS DATE)) AS INTEGER) AS record_gap_months
                       , CAST(COALESCE(summary.customer_history_length, 0) AS INTEGER) AS customer_history_length
                       , CAST(COALESCE({owned_count}, 0) AS TINYINT) AS products_owned_count
                       , CAST(COALESCE(summary.cumulative_acquisitions, 0) AS INTEGER) AS cumulative_acquisitions
                       , CAST(COALESCE(summary.cumulative_drops, 0) AS INTEGER) AS cumulative_drops
                       , CAST(date_diff('month', summary.last_acquisition_date, CAST(test.fecha_dato AS DATE)) AS INTEGER) AS months_since_last_acquisition
                       , CAST(summary.last_acquisition_date IS NULL AS TINYINT) AS never_acquired_before
                       , CAST(COALESCE(recent.acquisitions_last_1m, 0) AS INTEGER) AS acquisitions_last_1m
                       , CAST(COALESCE(recent.acquisitions_last_3m, 0) AS INTEGER) AS acquisitions_last_3m
                       , CAST(COALESCE(recent.acquisitions_last_6m, 0) AS INTEGER) AS acquisitions_last_6m
                FROM read_parquet('{test_sql}') AS test
                LEFT JOIN latest_history AS history USING (ncodpers)
                LEFT JOIN history_summary AS summary USING (ncodpers)
                LEFT JOIN recent_acquisitions AS recent USING (ncodpers)
            ) TO '{temporary_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        temporary.replace(destination)
        return destination
    finally:
        con.close()


def load_competition_input(path: str | Path, *, columns: Sequence[str], dtypes: dict[str, str] | None = None) -> object:
    """Read only the artifact-required columns to a pandas frame for scoring."""
    con = duckdb.connect(database=":memory:")
    try:
        available = _columns(con, Path(path))
        missing = set(columns).difference(available)
        if missing:
            raise ValueError(f"Prepared competition input lacks columns: {sorted(missing)}")
        frame = con.execute(f"SELECT {', '.join(_quote(column) for column in columns)} FROM read_parquet(?)", [str(path)]).fetchdf()
        for column, dtype in (dtypes or {}).items():
            frame[column] = frame[column].astype(dtype)
        return frame
    finally:
        con.close()


def _columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    return [row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()]


def _configure_duckdb(con: duckdb.DuckDBPyConnection, memory_limit: str | None, temp_directory: str | Path | None) -> None:
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    if temp_directory:
        directory = Path(temp_directory); directory.mkdir(parents=True, exist_ok=True)
        escaped_directory = str(directory).replace("'", "''")
        con.execute(f"SET temp_directory = '{escaped_directory}'")


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
