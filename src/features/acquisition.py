"""Disk-backed acquisition-target feature engineering for Santander products."""
from __future__ import annotations

from pathlib import Path

import duckdb


ACQUISITION_PREFIX = "acq_"


def acquisition_columns(product_columns: list[str]) -> list[str]:
    """Return the acquisition-target column name for every product column."""
    return [f"{ACQUISITION_PREFIX}{column}" for column in product_columns]


def build_acquisition_features(
    train_path: str | Path,
    test_path: str | Path | None,
    processed_dir: str | Path,
    *,
    force_process: bool = False,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
) -> dict[str, object]:
    """Persist 0->1 product transitions as TINYINT acquisition targets.

    The previous row is the chronologically nearest earlier record for the same
    customer.  A customer's first record, and every transition other than 0->1,
    receives 0.  Processing uses DuckDB windows and Parquet COPY so panel data
    is never materialised in pandas.
    """
    train_source = Path(train_path)
    test_source = Path(test_path) if test_path is not None else None
    destination_dir = Path(processed_dir)
    processed_train = destination_dir / "train.parquet"
    processed_test = destination_dir / "test.parquet"

    if not train_source.is_file():
        raise FileNotFoundError(f"Train checkpoint not found: {train_source}")
    if test_source is not None and not test_source.is_file():
        raise FileNotFoundError(f"Test checkpoint not found: {test_source}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        if memory_limit:
            con.execute(f"SET memory_limit = '{memory_limit}'")
        if temp_directory:
            temporary_dir = Path(temp_directory)
            temporary_dir.mkdir(parents=True, exist_ok=True)
            escaped_temp_dir = str(temporary_dir).replace("'", "''")
            con.execute(f"SET temp_directory = '{escaped_temp_dir}'")

        train_columns = _parquet_columns(con, train_source)
        required = {"ncodpers", "fecha_dato"}
        missing_required = required.difference(train_columns)
        if missing_required:
            raise ValueError(f"Train checkpoint is missing required columns: {sorted(missing_required)}")

        product_columns = [column for column in train_columns if column.endswith("_ult1")]
        if not product_columns:
            raise ValueError("Train checkpoint has no product columns ending in '_ult1'.")
        target_columns = acquisition_columns(product_columns)
        processed_columns = _parquet_columns(con, processed_train) if processed_train.is_file() else []

        if not force_process and set(target_columns).issubset(processed_columns):
            train_status = "reused"
        else:
            source_train = processed_train if processed_train.is_file() else train_source
            _write_acquisition_train(con, source_train, processed_train, product_columns, target_columns)
            train_status = "rebuilt"

        # Test has no product-state columns, so it cannot have acquisition labels.
        # Copy it once to the canonical processed location while preserving its schema.
        test_status = "unavailable"
        if test_source is not None:
            if processed_test.is_file():
                test_status = "reused"
            else:
                _copy_parquet(con, test_source, processed_test)
                test_status = "copied"

        return {
            "train": str(processed_train),
            "test": str(processed_test) if test_source is not None else None,
            "product_columns": product_columns,
            "acquisition_columns": target_columns,
            "train_status": train_status,
            "test_status": test_status,
        }
    finally:
        con.close()


def _parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    return [row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()]


def _write_acquisition_train(
    con: duckdb.DuckDBPyConnection,
    source_path: Path,
    destination_path: Path,
    product_columns: list[str],
    target_columns: list[str],
) -> None:
    source_columns = _parquet_columns(con, source_path)
    source_target_columns = [column for column in target_columns if column in source_columns]
    source_projection = "*" if not source_target_columns else f"* EXCLUDE ({', '.join(_quote(column) for column in source_target_columns)})"
    target_expressions = ",\n            ".join(
        f"CAST(CASE WHEN LAG(fecha_dato) OVER customer_time IS NOT NULL "
        f"AND COALESCE(TRY_CAST({_quote(product)} AS TINYINT), 0) = 1 "
        f"AND COALESCE(TRY_CAST(LAG({_quote(product)}) OVER customer_time AS TINYINT), 0) = 0 "
        f"THEN 1 ELSE 0 END AS TINYINT) AS {_quote(target)}"
        for product, target in zip(product_columns, target_columns)
    )
    source_sql = _quote_path(source_path)
    temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    copy_sql = f"""
        COPY (
            SELECT {source_projection},
                   {target_expressions}
            FROM read_parquet('{source_sql}')
            WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato)
        ) TO '{_quote_path(temporary_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(copy_sql)
    temporary_path.replace(destination_path)


def _copy_parquet(con: duckdb.DuckDBPyConnection, source_path: Path, destination_path: Path) -> None:
    temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{_quote_path(source_path)}')) "
        f"TO '{_quote_path(temporary_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    temporary_path.replace(destination_path)


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_path(path: Path) -> str:
    return str(path).replace("'", "''")
