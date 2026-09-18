"""Leakage-aware, disk-backed preprocessing for Santander customer profiles.

This module intentionally stops before feature engineering: it does not create
missingness flags, bins, encodings, interactions, lags, or product features.
It only applies the cleaning decisions documented in
``docs/customer_profile_feature_deep_eda.md``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb


MISSING_CATEGORY = "__MISSING__"
PROFILE_HISTORY_COLUMNS = ("age", "antiguedad", "renta", "sexo", "pais_residencia", "cod_prov", "fecha_alta")
CATEGORICAL_COLUMNS = (
    "ind_nuevo",
    "indrel",
    "indrel_1mes",
    "tiprel_1mes",
    "ind_actividad_cliente",
    "ind_empleado",
    "indfall",
    "sexo",
    "pais_residencia",
    "cod_prov",
    "canal_entrada",
    "segmento",
    "indresi",
    "indext",
)
BASELINE_DROP_COLUMNS = ("conyuemp", "nomprov", "ult_fec_cli_1t", "tipodom")


@dataclass(frozen=True)
class ProfilePreprocessingStats:
    """Statistics fitted only on the training partition used for a transform."""

    age_median: float
    antiguedad_median: float
    renta_median: float
    renta_clip_lower: float
    renta_clip_upper: float
    renta_clip_lower_quantile: float
    renta_clip_upper_quantile: float


def fit_profile_preprocessing(
    train_path: str | Path,
    *,
    renta_clip_lower_quantile: float = 0.0,
    renta_clip_upper_quantile: float = 0.99,
) -> ProfilePreprocessingStats:
    """Fit numeric imputation and clipping values from one training partition.

    Call this separately inside every time split. Passing the full historical
    train file is suitable only for preparing a final train/test artifact, not
    for evaluating an earlier validation month.
    """
    if not 0 <= renta_clip_lower_quantile < renta_clip_upper_quantile <= 1:
        raise ValueError("Renta clipping quantiles must satisfy 0 <= lower < upper <= 1.")
    source = Path(train_path)
    if not source.is_file():
        raise FileNotFoundError(f"Training Parquet not found: {source}")

    con = duckdb.connect(database=":memory:")
    try:
        columns = _parquet_columns(con, source)
        _require_columns(columns, {"age", "antiguedad", "renta"}, source)
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
                quantile_cont(age_value, 0.5) AS age_median,
                quantile_cont(antiguedad_value, 0.5) AS antiguedad_median,
                quantile_cont(renta_value, 0.5) AS renta_median,
                quantile_cont(renta_value, ?) AS renta_clip_lower,
                quantile_cont(renta_value, ?) AS renta_clip_upper
            FROM cleaned
            """,
            [str(source), renta_clip_lower_quantile, renta_clip_upper_quantile],
        ).fetchone()
    finally:
        con.close()

    if row is None or any(value is None for value in row):
        raise ValueError("Training data does not contain enough valid numeric profile values to fit preprocessing.")
    return ProfilePreprocessingStats(
        age_median=float(row[0]),
        antiguedad_median=float(row[1]),
        renta_median=float(row[2]),
        renta_clip_lower=float(row[3]),
        renta_clip_upper=float(row[4]),
        renta_clip_lower_quantile=renta_clip_lower_quantile,
        renta_clip_upper_quantile=renta_clip_upper_quantile,
    )


def transform_customer_profiles(
    source_path: str | Path,
    destination_path: str | Path,
    stats: ProfilePreprocessingStats,
    *,
    history_path: str | Path | None = None,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
) -> Path:
    """Transform one Parquet input using only prior customer observations.

    ``history_path`` may be the preceding train partition when transforming a
    later test partition. It contributes only rows before a target row in the
    ``ncodpers, fecha_dato`` ordering; it is never copied to the output.
    """
    source = Path(source_path)
    destination = Path(destination_path)
    history = Path(history_path) if history_path is not None else None
    if not source.is_file():
        raise FileNotFoundError(f"Source Parquet not found: {source}")
    if history is not None and not history.is_file():
        raise FileNotFoundError(f"History Parquet not found: {history}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        _configure_duckdb(con, memory_limit, temp_directory)
        source_columns = _parquet_columns(con, source)
        _require_columns(source_columns, {"ncodpers", "fecha_dato"}, source)
        replacement_columns = [column for column in PROFILE_HISTORY_COLUMNS if column in source_columns]
        categorical_columns = [column for column in CATEGORICAL_COLUMNS if column in source_columns]
        drop_columns = [column for column in BASELINE_DROP_COLUMNS if column in source_columns]

        if history is not None:
            history_columns = _parquet_columns(con, history)
            _require_columns(history_columns, {"ncodpers", "fecha_dato"}, history)
            missing_history = set(replacement_columns).difference(history_columns)
            if missing_history:
                raise ValueError(f"History Parquet is missing profile columns: {sorted(missing_history)}")

        query = _transform_query(
            source=source,
            history=history,
            source_columns=source_columns,
            replacement_columns=replacement_columns,
            categorical_columns=categorical_columns,
            drop_columns=drop_columns,
            stats=stats,
        )
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        con.execute(f"COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(temporary)])
        temporary.replace(destination)
        return destination
    finally:
        con.close()


def preprocess_customer_profiles(
    train_path: str | Path,
    test_path: str | Path | None,
    processed_dir: str | Path,
    *,
    force_process: bool = False,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
    renta_clip_lower_quantile: float = 0.0,
    renta_clip_upper_quantile: float = 0.99,
) -> dict[str, object]:
    """Write cleaned train/test profile checkpoints to the canonical processed path.

    The first call fits statistics on ``train_path`` and writes
    ``processed/train.parquet`` and ``processed/test.parquet``. Test can use
    prior observations from train, but no future test observation is used for a
    row. Existing outputs are reused unless ``force_process`` is true.
    """
    raw_train = Path(train_path)
    raw_test = Path(test_path) if test_path is not None else None
    destination_dir = Path(processed_dir)
    processed_train = destination_dir / "train.parquet"
    processed_test = destination_dir / "test.parquet"
    stats_path = destination_dir / "profile_preprocessing_stats.json"

    if not raw_train.is_file():
        raise FileNotFoundError(f"Train Parquet not found: {raw_train}")
    if raw_test is not None and not raw_test.is_file():
        raise FileNotFoundError(f"Test Parquet not found: {raw_test}")
    destination_dir.mkdir(parents=True, exist_ok=True)

    if processed_train.is_file() and not force_process:
        return {
            "train": str(processed_train),
            "test": str(processed_test) if processed_test.is_file() else None,
            "stats": str(stats_path) if stats_path.is_file() else None,
            "status": "reused",
        }

    # Preserve any unrelated existing columns, e.g. targets created by a later
    # pipeline step, when a forced refresh is explicitly requested.
    train_source = processed_train if processed_train.is_file() else raw_train
    test_source = processed_test if processed_test.is_file() else raw_test
    stats = fit_profile_preprocessing(
        train_source,
        renta_clip_lower_quantile=renta_clip_lower_quantile,
        renta_clip_upper_quantile=renta_clip_upper_quantile,
    )
    transform_customer_profiles(
        train_source,
        processed_train,
        stats,
        memory_limit=memory_limit,
        temp_directory=temp_directory,
    )
    if test_source is not None:
        transform_customer_profiles(
            test_source,
            processed_test,
            stats,
            history_path=train_source,
            memory_limit=memory_limit,
            temp_directory=temp_directory,
        )
    stats_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
    return {
        "train": str(processed_train),
        "test": str(processed_test) if test_source is not None else None,
        "stats": str(stats_path),
        "status": "rebuilt",
    }


def _transform_query(
    *,
    source: Path,
    history: Path | None,
    source_columns: list[str],
    replacement_columns: list[str],
    categorical_columns: list[str],
    drop_columns: list[str],
    stats: ProfilePreprocessingStats,
) -> str:
    history_projection = ", ".join(_clean_expression(column) + f" AS {_quote(column)}" for column in replacement_columns)
    target_projection = ", ".join(_clean_expression(column) + f" AS {_quote(column)}" for column in replacement_columns)
    history_cte = ""
    if history is not None:
        history_cte = f"""
            SELECT ncodpers, fecha_dato, 0 AS source_rank, {history_projection}
            FROM read_parquet('{_quote_path(history)}')
            UNION ALL
        """

    prior_expressions = ",\n                ".join(
        f"LAST_VALUE({_quote(column)} IGNORE NULLS) OVER customer_time AS prior_{column}"
        for column in replacement_columns
    )
    filled_expressions = ",\n                ".join(
        _filled_expression(column, stats) + f" AS {_quote('filled_' + column)}" for column in replacement_columns
    )
    # Exclude every transformed column before adding its cleaned replacement;
    # otherwise DuckDB would retain the raw column and rename the replacement.
    excluded = sorted(set(replacement_columns) | set(categorical_columns) | set(drop_columns))
    select_base = "base.*" if not excluded else f"base.* EXCLUDE ({', '.join(_quote(column) for column in excluded)})"
    output_expressions = [select_base]
    output_expressions.extend(f"filled.{_quote('filled_' + column)} AS {_quote(column)}" for column in replacement_columns)
    output_expressions.extend(_categorical_expression(column) + f" AS {_quote(column)}" for column in categorical_columns if column not in replacement_columns)
    output_sql = ",\n            ".join(output_expressions)

    return f"""
        WITH target_base AS (
            SELECT * FROM read_parquet('{_quote_path(source)}')
        ),
        combined AS (
            {history_cte}
            SELECT ncodpers, fecha_dato, 1 AS source_rank, {target_projection}
            FROM target_base
        ),
        previous_values AS (
            SELECT
                *,
                {prior_expressions}
            FROM combined
            WINDOW customer_time AS (
                PARTITION BY ncodpers
                ORDER BY fecha_dato, source_rank
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            )
        ),
        filled AS (
            SELECT
                ncodpers,
                fecha_dato,
                {filled_expressions}
            FROM previous_values
            WHERE source_rank = 1
        )
        SELECT
            {output_sql}
        FROM target_base AS base
        JOIN filled USING (ncodpers, fecha_dato)
    """


def _clean_expression(column: str) -> str:
    quoted = _quote(column)
    if column == "antiguedad":
        return f"CASE WHEN TRY_CAST({quoted} AS DOUBLE) < 0 THEN NULL ELSE TRY_CAST({quoted} AS DOUBLE) END"
    if column == "renta":
        return f"CASE WHEN TRY_CAST({quoted} AS DOUBLE) < 0 THEN NULL ELSE TRY_CAST({quoted} AS DOUBLE) END"
    if column == "age":
        return f"TRY_CAST({quoted} AS DOUBLE)"
    return quoted


def _filled_expression(column: str, stats: ProfilePreprocessingStats) -> str:
    clean = _quote(column)
    prior = f"prior_{column}"
    if column == "age":
        return f"COALESCE({clean}, {prior}, {stats.age_median})"
    if column == "antiguedad":
        return f"COALESCE({clean}, {prior}, {stats.antiguedad_median})"
    if column == "renta":
        value = f"COALESCE({clean}, {prior}, {stats.renta_median})"
        return f"LEAST(GREATEST({value}, {stats.renta_clip_lower}), {stats.renta_clip_upper})"
    value = f"COALESCE({clean}, {prior})"
    if column in {"sexo", "pais_residencia", "cod_prov"}:
        return f"COALESCE(NULLIF(TRIM(CAST({value} AS VARCHAR)), ''), '{MISSING_CATEGORY}')"
    # fecha_alta is a date: retain a residual null rather than inventing a date.
    return value


def _categorical_expression(column: str) -> str:
    quoted = _quote(column)
    if column == "indrel_1mes":
        return (
            f"CASE WHEN {quoted} IS NULL THEN '{MISSING_CATEGORY}' "
            f"WHEN TRIM(CAST({quoted} AS VARCHAR)) IN ('1', '1.0') THEN '1' "
            f"ELSE TRIM(CAST({quoted} AS VARCHAR)) END"
        )
    return f"COALESCE(NULLIF(TRIM(CAST({quoted} AS VARCHAR)), ''), '{MISSING_CATEGORY}')"


def _configure_duckdb(con: duckdb.DuckDBPyConnection, memory_limit: str | None, temp_directory: str | Path | None) -> None:
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    if temp_directory:
        path = Path(temp_directory)
        path.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{_quote_path(path)}'")


def _parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    return [row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()]


def _require_columns(columns: list[str], required: set[str], path: Path) -> None:
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Parquet file {path} is missing required columns: {sorted(missing)}")


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_path(path: str | Path) -> str:
    return str(path).replace("'", "''")
