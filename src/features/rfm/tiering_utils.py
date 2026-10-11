from pathlib import Path
import duckdb
from collections.abc import Sequence
from numbers import Real
from math import isfinite
from src.features.customer_history import quote

def tier_case_sql(column: str, boundaries: Sequence[Real], *, lower_is_better: bool) -> str:
    values = [float(value) for value in boundaries]
    if values != sorted(values) or any(not isfinite(value) for value in values):
        raise ValueError(f"Tier boundaries for {column} must be sorted finite values.")
    tier_count = len(values) + 1
    clauses = []
    for index, value in enumerate(values, start=1):
        score = tier_count - index + 1 if lower_is_better else index
        clauses.append(f"WHEN {quote(column)} <= {value:.17g} THEN {score}")
    final = 1 if lower_is_better else tier_count
    return "CASE " + " ".join(clauses) + f" ELSE {final} END"

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
