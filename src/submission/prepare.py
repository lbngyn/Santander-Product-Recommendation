"""Leakage-safe preparation of Santander's June-2016 competition test input."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import duckdb

from src.features.customer_history import HISTORY_FEATURE_NAMES, product_event_count_sql
from src.features.rfm import rfm_acquisition_count_sql
from src.features.persona import (
    PERSONA_FEATURES,
    persona_feature_projection_sql,
    raw_persona_projection_sql,
)


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
        unknown_features = set(profile_features).difference(PERSONA_FEATURES)
        if unknown_features:
            raise ValueError(f"Unknown canonical persona model feature(s): {sorted(unknown_features)}")
        if not bool(con.execute("SELECT count(*) = count(DISTINCT ncodpers) FROM read_parquet(?)", [str(test)]).fetchone()[0]):
            raise ValueError("Competition test must contain one row per ncodpers.")
        q = _quote
        persona_sql = persona_feature_projection_sql(features=profile_features)
        test_raw_persona_sql = raw_persona_projection_sql("test", test_columns)
        history_raw_persona_sql = raw_persona_projection_sql("history", history_columns)
        profiles = ", ".join(f"test.{q(name)}" for name in profile_features)
        previous = ", ".join(
            f"CAST(COALESCE(history.{q(product)}, 0) AS TINYINT) AS {q('prev_' + product)}"
            for product in product_names
        )
        monetary_known = " + ".join(
            f"CASE WHEN TRY_CAST(history.{q(product)} AS INTEGER) IN (0, 1) THEN 1 ELSE 0 END"
            for product in product_names
        )
        monetary_owned = " + ".join(
            f"CASE WHEN TRY_CAST(history.{q(product)} AS INTEGER) = 1 THEN 1 ELSE 0 END"
            for product in product_names
        )
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
                WITH profile_history AS (
                    SELECT history.ncodpers, history.fecha_dato, 0 AS source_rank, {history_raw_persona_sql}
                    FROM read_parquet('{history_sql}') AS history
                    WHERE CAST(history.fecha_dato AS DATE) <= CAST('{history_date_sql}' AS DATE)
                ), profile_test AS (
                    SELECT test.ncodpers, test.fecha_dato, 1 AS source_rank, {test_raw_persona_sql}
                    FROM read_parquet('{test_sql}') AS test
                ), profile_ordered AS (
                    SELECT *, {persona_sql}
                    FROM (SELECT * FROM profile_history UNION ALL BY NAME SELECT * FROM profile_test)
                    WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato, source_rank),
                           prior_customer_rows AS (PARTITION BY ncodpers ORDER BY fecha_dato, source_rank ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                ), test_rows AS (
                    SELECT ncodpers, fecha_dato{', ' if profiles else ''}{profiles}
                    FROM profile_ordered AS test WHERE source_rank = 1
                ), history_base AS (
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
                ), rfm_events AS (
                    SELECT *, {rfm_acquisition_count_sql(product_names)} AS rfm_n_acquisition FROM events
                ), latest_history AS (
                    SELECT * EXCLUDE (row_number) FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY ncodpers ORDER BY fecha_dato DESC) AS row_number FROM rfm_events
                    ) WHERE row_number = 1
                ), history_summary AS (
                    SELECT ncodpers, COUNT(*) AS rfm_observed_history_records,
                           COALESCE(SUM(COALESCE(rfm_n_acquisition, 0)), 0) AS rfm_frequency,
                           COALESCE(SUM(drop_events), 0) AS cumulative_drops,
                           MAX(CASE WHEN rfm_n_acquisition > 0 THEN CAST(fecha_dato AS DATE) END) AS rfm_last_acquisition_date
                    FROM rfm_events GROUP BY ncodpers
                ), recent_acquisitions AS (
                    SELECT test.ncodpers,
                           COALESCE(SUM(CASE WHEN CAST(events.fecha_dato AS DATE) >= CAST(test.fecha_dato AS DATE) - INTERVAL 1 MONTH THEN events.acquisition_events ELSE 0 END), 0) AS acquisitions_last_1m,
                           COALESCE(SUM(CASE WHEN CAST(events.fecha_dato AS DATE) >= CAST(test.fecha_dato AS DATE) - INTERVAL 3 MONTH THEN events.acquisition_events ELSE 0 END), 0) AS acquisitions_last_3m,
                           COALESCE(SUM(CASE WHEN CAST(events.fecha_dato AS DATE) >= CAST(test.fecha_dato AS DATE) - INTERVAL 6 MONTH THEN events.acquisition_events ELSE 0 END), 0) AS acquisitions_last_6m
                    FROM test_rows AS test
                    LEFT JOIN rfm_events AS events ON test.ncodpers = events.ncodpers
                        AND CAST(events.fecha_dato AS DATE) < CAST(test.fecha_dato AS DATE)
                    GROUP BY test.ncodpers
                )
                SELECT test.ncodpers, test.fecha_dato{', ' if profiles else ''}{profiles}, {previous}
                       , CAST(date_diff('month', CAST(history.fecha_dato AS DATE), CAST(test.fecha_dato AS DATE)) AS INTEGER) AS record_gap_months
                       , CAST(history.ncodpers IS NOT NULL AS TINYINT) AS rfm_has_previous_record
                       , CAST(date_diff('month', summary.rfm_last_acquisition_date, CAST(test.fecha_dato AS DATE)) AS INTEGER) AS rfm_recency_months
                       , CAST(summary.rfm_last_acquisition_date IS NULL AS TINYINT) AS rfm_never_acquired_before
                       , CAST(COALESCE(summary.rfm_frequency, 0) AS INTEGER) AS rfm_frequency
                       , CAST(CASE WHEN history.ncodpers IS NOT NULL AND ({monetary_known}) > 0 THEN ({monetary_owned}) ELSE NULL END AS INTEGER) AS rfm_monetary
                       , CAST(COALESCE(summary.rfm_observed_history_records, 0) AS INTEGER) AS rfm_observed_history_records
                       , CAST(COALESCE(summary.cumulative_drops, 0) AS INTEGER) AS cumulative_drops
                       , CAST(COALESCE(recent.acquisitions_last_1m, 0) AS INTEGER) AS acquisitions_last_1m
                       , CAST(COALESCE(recent.acquisitions_last_3m, 0) AS INTEGER) AS acquisitions_last_3m
                       , CAST(COALESCE(recent.acquisitions_last_6m, 0) AS INTEGER) AS acquisitions_last_6m
                FROM test_rows AS test
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


def materialize_acquisition_only_test_input(test, history, destination, runtime, work):
    """Clean June profiles using historical residuals and the shared income rule.

    Canonical history is preserved. Only observations before June enter profile
    fills and previous ownership. Unknown customers have zero previous ownership.
    June income uses the same un-clipped RFM rule as canonical preprocessing.
    """
    from src.features.acquisition_only import _copy
    from src.features.acquisition_only_contract import DATES, KEYS, NUMERIC, PROFILES, parquet, quote
    from src.products import PRODUCT_COLUMNS
    from src.utils.modeling_runtime import connection
    from src.preprocessing.customer_profile import (
        fit_baseline_profile_preprocessing, transform_customer_profiles_past_only,
    )
    from src.preprocessing.renta_monetary import preprocess_renta_monetary

    test, history, destination, work = map(Path, (test, history, destination, work))
    work.mkdir(parents=True, exist_ok=True)
    historical = work / "history_before_june.parquet"
    raw_test = work / "june_profiles_raw.parquet"
    clean_test = work / "june_profiles_clean.parquet"
    income_source = work / "income_source.parquet"
    income = work / "income_filled.parquet"
    with connection(runtime, work) as con:
        for source, required in ((test, [*KEYS, *PROFILES]),
                                 (history, [*KEYS, *PROFILES, *PRODUCT_COLUMNS])):
            columns = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {parquet(source)}").fetchall()}
            if missing := set(required) - columns:
                raise ValueError(f"Missing inference source columns in {source}: {sorted(missing)}")
        invalid = con.execute(f"SELECT count(*) = 0 OR count(*) != count(DISTINCT ncodpers) "
                              f"OR count(*) != count(ncodpers) OR count(*) != count(fecha_dato) "
                              f"OR count(*) FILTER (WHERE CAST(fecha_dato AS DATE) < DATE '2016-06-01' "
                              f"OR CAST(fecha_dato AS DATE) >= DATE '2016-07-01') > 0 FROM {parquet(test)}").fetchone()[0]
        if invalid:
            raise ValueError("Test input requires unique non-null customers and June-2016 dates")
        history_columns = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {parquet(history)}").fetchall()}
        names = [*KEYS, *PROFILES, *PRODUCT_COLUMNS]
        if "renta_raw" in history_columns:
            names.append("renta_raw")
        _copy(con, f"SELECT {', '.join(map(quote, names))} FROM {parquet(history)} "
                   "WHERE CAST(fecha_dato AS DATE) < DATE '2016-06-01'", historical)
        _copy(con, f"SELECT {', '.join(map(quote, [*KEYS, *PROFILES]))} FROM {parquet(test)}", raw_test)
    # Categories/dates use the shared past-only cleaning. Numeric residuals are
    # fitted on the pinned canonical history, never on the June population.
    stats = fit_baseline_profile_preprocessing(historical)
    transform_customer_profiles_past_only(
        raw_test, clean_test, stats, history_path=historical,
        memory_limit=runtime.get("memory_limit"), temp_directory=work / "profile_spill",
    )
    with connection(runtime, work) as con:
        renta = "renta_raw" if "renta_raw" in history_columns else "renta"
        # The income rule reads original income, rather than a residual filled
        # by the profile cleaner; other profile columns remain cleaned.
        _copy(con, f"SELECT ncodpers, fecha_dato, {quote(renta)} AS renta, cod_prov, pais_residencia "
                   f"FROM {parquet(historical)} UNION ALL BY NAME "
                   f"SELECT ncodpers, fecha_dato, renta, cod_prov, pais_residencia FROM {parquet(raw_test)}", income_source)
    preprocess_renta_monetary(income_source, income, memory_limit=runtime.get("memory_limit"),
                             temp_directory=work / "income_spill")
    profiles = []
    for name in PROFILES:
        value = "i.renta_filled" if name == "renta" else f"t.{quote(name)}"
        if name in NUMERIC:
            expression = f"TRY_CAST({value} AS FLOAT)"
        elif name in DATES:
            expression = f"CAST(date_diff('day', DATE '1970-01-01', TRY_CAST({value} AS DATE)) AS FLOAT)"
        else:
            expression = f"COALESCE(CAST({value} AS VARCHAR), '__MISSING__')"
        profiles.append(f"{expression} AS {quote(name)}")
    previous = [f"COALESCE(TRY_CAST(h.{quote(p)} AS TINYINT), 0) AS {quote('prev_' + p)}"
                for p in PRODUCT_COLUMNS]
    with connection(runtime, work) as con:
        _copy(con, f"WITH latest AS (SELECT * FROM {parquet(historical)} "
                   "QUALIFY ROW_NUMBER() OVER (PARTITION BY ncodpers ORDER BY fecha_dato DESC) = 1) "
                   f"SELECT t.ncodpers, t.fecha_dato, {', '.join([*profiles, *previous])} "
                   f"FROM {parquet(clean_test)} t JOIN {parquet(income)} i USING (ncodpers, fecha_dato) "
                   "LEFT JOIN latest h USING (ncodpers) ORDER BY t.ncodpers", destination)
    return destination
