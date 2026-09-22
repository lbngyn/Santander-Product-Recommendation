"""Cached temporal split artifacts for product recommendation experiments."""
from __future__ import annotations

from pathlib import Path

import duckdb


def build_validation_split(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    validation_date: str,
    force_split: bool = False,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
) -> dict[str, object]:
    """Cache historical train, validation input, and validation target Parquet.

    The input contains profile values at X plus only `prev_<product>` product
    states from X-1. The target contains the 24 product flags at X. Therefore
    consumers cannot accidentally read the target product flags as features.
    """
    source, destination = Path(source_path), Path(output_dir)
    train, validation_input, validation_target = destination / "train.parquet", destination / "validation_input.parquet", destination / "validation_target.parquet"
    if not source.is_file(): raise FileNotFoundError(f"Source checkpoint not found: {source}")
    if not force_split and all(path.is_file() for path in (train, validation_input, validation_target)):
        return {"train": str(train), "validation_input": str(validation_input), "validation_target": str(validation_target), "status": "reused"}
    destination.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        if memory_limit: con.execute(f"SET memory_limit = '{memory_limit}'")
        if temp_directory:
            temp = Path(temp_directory); temp.mkdir(parents=True, exist_ok=True)
            escaped_temp = str(temp).replace("'", "''")
            con.execute(f"SET temp_directory = '{escaped_temp}'")
        columns = [row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]).fetchall()]
        products = [column for column in columns if column.endswith("_ult1")]
        if not products: raise ValueError("Source checkpoint has no product columns.")
        profiles = [column for column in columns if column not in {"ncodpers", "fecha_dato", *products}]
        previous = ", ".join(f"COALESCE(TRY_CAST(\"previous_{p}\" AS TINYINT), 0) AS \"prev_{p}\"" for p in products)
        profile_sql = ", ".join(f"\"{column}\"" for column in profiles)
        target_sql = ", ".join(f"\"{column}\"" for column in products)
        escaped_source = str(source).replace("'", "''")
        ordered = f"""WITH ordered AS (
            SELECT *, {', '.join(f'LAG("{p}") OVER customer_time AS "previous_{p}"' for p in products)}
            FROM read_parquet('{escaped_source}')
            WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato)
        )"""
        _copy(con, f"SELECT * FROM read_parquet('{escaped_source}') WHERE CAST(fecha_dato AS DATE) < CAST('{validation_date}' AS DATE)", train)
        _copy(con, ordered + f" SELECT ncodpers, fecha_dato{', ' if profile_sql else ''}{profile_sql}, {previous} FROM ordered WHERE CAST(fecha_dato AS DATE) = CAST('{validation_date}' AS DATE)", validation_input)
        _copy(con, f"SELECT ncodpers, fecha_dato, {target_sql} FROM read_parquet('{escaped_source}') WHERE CAST(fecha_dato AS DATE) = CAST('{validation_date}' AS DATE)", validation_target)
    finally:
        con.close()
    return {"train": str(train), "validation_input": str(validation_input), "validation_target": str(validation_target), "status": "rebuilt"}


def _copy(con: duckdb.DuckDBPyConnection, query: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists(): temporary.unlink()
    escaped_temporary = str(temporary).replace("'", "''")
    con.execute(f"COPY ({query}) TO '{escaped_temporary}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    temporary.replace(destination)
