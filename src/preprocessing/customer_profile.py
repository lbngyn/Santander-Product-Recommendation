"""Customer-profile cleaning used by the first reproducible baseline.

This module deliberately preserves raw profile columns and categorical levels:
it performs neither encoding nor feature engineering.  Its train transform can
use a value after a missing row for *offline data repair*.  That is useful for
the requested baseline dataset, but is not valid for time-based validation or
online scoring.  Set ``allow_future_values=False`` for those workflows.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb


MISSING_CATEGORY = "__MISSING__"
IDENTIFIER_COLUMNS = {"ncodpers", "fecha_dato"}
NUMERIC_PROFILE_COLUMNS = ("age", "antiguedad", "renta")
DATE_PROFILE_COLUMNS = ("fecha_alta", "ult_fec_cli_1t")


@dataclass(frozen=True)
class BaselineProfilePreprocessingStats:
    """Numeric values fitted from the baseline training input only."""

    age_median: float
    antiguedad_median: float
    renta_median: float
    renta_clip_upper: float
    renta_clip_upper_quantile: float


def fit_baseline_profile_preprocessing(
    train_path: str | Path,
    *,
    renta_clip_upper_quantile: float = 0.99,
) -> BaselineProfilePreprocessingStats:
    """Fit residual numeric medians and the income upper clipping threshold."""
    if not 0 < renta_clip_upper_quantile <= 1:
        raise ValueError("renta_clip_upper_quantile must be in (0, 1].")
    source = Path(train_path)
    if not source.is_file():
        raise FileNotFoundError(f"Training Parquet not found: {source}")

    con = duckdb.connect(database=":memory:")
    try:
        _require_columns(_parquet_columns(con, source), set(NUMERIC_PROFILE_COLUMNS), source)
        row = con.execute(
            """
            WITH cleaned AS (
                SELECT
                    TRY_CAST(age AS DOUBLE) AS age_value,
                    CASE WHEN TRY_CAST(antiguedad AS DOUBLE) < 0 THEN NULL
                         ELSE TRY_CAST(antiguedad AS DOUBLE) END AS antiguedad_value,
                    CASE WHEN TRY_CAST(renta AS DOUBLE) < 0 THEN NULL
                         ELSE TRY_CAST(renta AS DOUBLE) END AS renta_value
                FROM read_parquet(?)
            )
            SELECT
                quantile_cont(age_value, 0.5),
                quantile_cont(antiguedad_value, 0.5),
                quantile_cont(renta_value, 0.5),
                quantile_cont(renta_value, ?)
            FROM cleaned
            """,
            [str(source), renta_clip_upper_quantile],
        ).fetchone()
    finally:
        con.close()
    if row is None or any(value is None for value in row):
        raise ValueError("Training data has insufficient valid numeric values for baseline preprocessing.")
    return BaselineProfilePreprocessingStats(
        age_median=float(row[0]),
        antiguedad_median=float(row[1]),
        renta_median=float(row[2]),
        renta_clip_upper=float(row[3]),
        renta_clip_upper_quantile=renta_clip_upper_quantile,
    )


def transform_customer_profiles_bidirectional(
    source_path: str | Path,
    destination_path: str | Path,
    stats: BaselineProfilePreprocessingStats,
    *,
    history_path: str | Path | None = None,
    allow_future_values: bool = False,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
) -> Path:
    """Fill profile fields from the same customer and write a Parquet artifact.

    Numeric values between two observations are the arithmetic mean requested
    for baseline v0.  A one-sided observation supplies the nearest value.
    Categorical/date values use the closest observed record; categorical
    residual nulls become ``__MISSING__`` and date residuals remain null.
    """
    source, destination = Path(source_path), Path(destination_path)
    history = Path(history_path) if history_path else None
    if not source.is_file():
        raise FileNotFoundError(f"Source Parquet not found: {source}")
    if history is not None and not history.is_file():
        raise FileNotFoundError(f"History Parquet not found: {history}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        _configure_duckdb(con, memory_limit, temp_directory)
        source_columns = _parquet_columns(con, source)
        _require_columns(source_columns, IDENTIFIER_COLUMNS, source)
        profile_columns = _profile_columns(source_columns)
        if history is not None:
            _require_columns(_parquet_columns(con, history), set(profile_columns) | IDENTIFIER_COLUMNS, history)
        query = _transform_query(
            source,
            history,
            source_columns,
            profile_columns,
            stats,
            allow_future_values=allow_future_values,
        )
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        con.execute(
            f"COPY ({query}) TO '{_quote_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        temporary.replace(destination)
        return destination
    finally:
        con.close()


def preprocess_customer_profile_baseline(
    train_path: str | Path,
    test_path: str | Path | None,
    output_dir: str | Path,
    *,
    force_process: bool = False,
    allow_future_values_within_train: bool = True,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
    renta_clip_upper_quantile: float = 0.99,
) -> dict[str, object]:
    """Build baseline-v0 train/test profile artifacts and fitted-stat metadata."""
    train, test, directory = Path(train_path), Path(test_path) if test_path else None, Path(output_dir)
    output_train, output_test = directory / "train.parquet", directory / "test.parquet"
    stats_path = directory / "preprocessing_stats.json"
    if not train.is_file():
        raise FileNotFoundError(f"Train Parquet not found: {train}")
    if test is not None and not test.is_file():
        raise FileNotFoundError(f"Test Parquet not found: {test}")
    directory.mkdir(parents=True, exist_ok=True)
    if output_train.is_file() and (test is None or output_test.is_file()) and not force_process:
        return {"train": str(output_train), "test": str(output_test) if test else None, "stats": str(stats_path), "status": "reused"}

    stats = fit_baseline_profile_preprocessing(train, renta_clip_upper_quantile=renta_clip_upper_quantile)
    transform_customer_profiles_bidirectional(
        train, output_train, stats, allow_future_values=allow_future_values_within_train,
        memory_limit=memory_limit, temp_directory=temp_directory,
    )
    if test is not None:
        # A future test row is unavailable at scoring time; test is always past-only.
        transform_customer_profiles_bidirectional(
            test, output_test, stats, history_path=train, allow_future_values=False,
            memory_limit=memory_limit, temp_directory=temp_directory,
        )
    stats_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
    return {"train": str(output_train), "test": str(output_test) if test else None, "stats": str(stats_path), "status": "rebuilt"}


def _profile_columns(columns: list[str]) -> list[str]:
    return [
        column for column in columns
        if column not in IDENTIFIER_COLUMNS and not column.endswith("_ult1") and not column.startswith("acq_")
    ]


def _transform_query(source: Path, history: Path | None, source_columns: list[str], profile_columns: list[str], stats: BaselineProfilePreprocessingStats, *, allow_future_values: bool) -> str:
    cleaned = {column: _clean_expression(column) for column in profile_columns}
    projection = ", ".join(f"{expression} AS {_quote(column)}" for column, expression in cleaned.items())
    history_cte = ""
    if history is not None:
        history_cte = f"SELECT ncodpers, fecha_dato, 0 AS source_rank, {projection} FROM read_parquet('{_quote_path(history)}') UNION ALL"
    previous_values = ",\n                ".join(
        f"LAST_VALUE({_quote(column)} IGNORE NULLS) OVER previous_{index} AS prev_{index}, "
        f"LAST_VALUE(CASE WHEN {_quote(column)} IS NOT NULL THEN fecha_dato END IGNORE NULLS) OVER previous_{index} AS prev_date_{index}"
        for index, column in enumerate(profile_columns)
    )
    next_values = ",\n                ".join(
        f"FIRST_VALUE({_quote(column)} IGNORE NULLS) OVER following_{index} AS next_{index}, "
        f"FIRST_VALUE(CASE WHEN {_quote(column)} IS NOT NULL THEN fecha_dato END IGNORE NULLS) OVER following_{index} AS next_date_{index}"
        for index, column in enumerate(profile_columns)
    ) if allow_future_values else ""
    filled = ",\n                ".join(
        _fill_expression(column, index, stats, allow_future_values) + f" AS {_quote('filled_' + column)}"
        for index, column in enumerate(profile_columns)
    )
    excluded = ", ".join(_quote(column) for column in profile_columns)
    base_select = f"base.* EXCLUDE ({excluded})" if excluded else "base.*"
    output = ",\n            ".join([base_select, *(f"filled.{_quote('filled_' + c)} AS {_quote(c)}" for c in profile_columns)])
    following_window = ""
    if allow_future_values:
        following_window = ",\n                ".join(
            f"following_{index} AS (PARTITION BY ncodpers ORDER BY fecha_dato, source_rank ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)"
            for index, _ in enumerate(profile_columns)
        )
    previous_window = ",\n                ".join(
        f"previous_{index} AS (PARTITION BY ncodpers ORDER BY fecha_dato, source_rank ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)"
        for index, _ in enumerate(profile_columns)
    )
    window_clause = previous_window + (",\n                " + following_window if following_window else "")
    next_projection = (",\n                " + next_values) if next_values else ""
    return f"""
        WITH target_base AS (SELECT * FROM read_parquet('{_quote_path(source)}')),
        combined AS (
            {history_cte}
            SELECT ncodpers, fecha_dato, 1 AS source_rank, {projection} FROM target_base
        ),
        neighbour_values AS (
            SELECT *,
                {previous_values}{next_projection}
            FROM combined
            WINDOW {window_clause}
        ),
        filled AS (
            SELECT ncodpers, fecha_dato,
                {filled}
            FROM neighbour_values WHERE source_rank = 1
        )
        SELECT {output}
        FROM target_base AS base JOIN filled USING (ncodpers, fecha_dato)
    """


def _clean_expression(column: str) -> str:
    quoted = _quote(column)
    if column in {"antiguedad", "renta"}:
        return f"CASE WHEN TRY_CAST({quoted} AS DOUBLE) < 0 THEN NULL ELSE TRY_CAST({quoted} AS DOUBLE) END"
    if column == "age":
        return f"TRY_CAST({quoted} AS DOUBLE)"
    return quoted


def _fill_expression(column: str, index: int, stats: BaselineProfilePreprocessingStats, allow_future_values: bool) -> str:
    current, previous = _quote(column), f"prev_{index}"
    following = f"next_{index}" if allow_future_values else "NULL"
    if column in NUMERIC_PROFILE_COLUMNS:
        fallback = getattr(stats, f"{column}_median")
        candidate = (
            f"CASE WHEN {current} IS NOT NULL THEN {current} "
            f"WHEN {previous} IS NOT NULL AND {following} IS NOT NULL THEN ({previous} + {following}) / 2.0 "
            f"ELSE COALESCE({previous}, {following}, {fallback}) END"
        )
        return f"LEAST({candidate}, {stats.renta_clip_upper})" if column == "renta" else candidate
    if column in DATE_PROFILE_COLUMNS:
        return _nearest_expression(current, previous, following, index, allow_future_values, residual="NULL")
    result = _nearest_expression(current, previous, following, index, allow_future_values, residual=f"'{MISSING_CATEGORY}'")
    if column == "indrel_1mes":
        return f"CASE WHEN {result} IN ('1', '1.0') THEN '1' ELSE {result} END"
    return result


def _nearest_expression(current: str, previous: str, following: str, index: int, allow_future_values: bool, *, residual: str) -> str:
    if not allow_future_values:
        value = f"COALESCE({current}, {previous})"
    else:
        value = (
            f"CASE WHEN {current} IS NOT NULL THEN {current} "
            f"WHEN {previous} IS NULL THEN {following} WHEN {following} IS NULL THEN {previous} "
            f"WHEN date_diff('day', TRY_CAST(prev_date_{index} AS DATE), TRY_CAST(fecha_dato AS DATE)) "
            f"<= date_diff('day', TRY_CAST(fecha_dato AS DATE), TRY_CAST(next_date_{index} AS DATE)) THEN {previous} "
            f"ELSE {following} END"
        )
    if residual == "NULL":
        return value
    return f"COALESCE(NULLIF(TRIM(CAST({value} AS VARCHAR)), ''), {residual})"


def _configure_duckdb(con: duckdb.DuckDBPyConnection, memory_limit: str | None, temp_directory: str | Path | None) -> None:
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    if temp_directory:
        path = Path(temp_directory); path.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{_quote_path(path)}'")


def _parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    return [row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()]


def _require_columns(columns: list[str], required: set[str], path: Path) -> None:
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Parquet file {path} is missing columns: {sorted(missing)}")


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_path(path: str | Path) -> str:
    return str(path).replace("'", "''")
