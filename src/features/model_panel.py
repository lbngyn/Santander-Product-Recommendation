"""Leakage-safe panel inputs for product-acquisition classifiers.

The output deliberately contains no current-month product flag as a feature.
Current flags are only used to write ``acq_*`` labels.
"""
from __future__ import annotations

from pathlib import Path

import duckdb


def build_model_panel(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    force_process: bool = False,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
) -> Path:
    """Materialise X_t with profiles at t and product history through t-1.

    ``previous_observation`` is retained for training eligibility: a first
    observed customer row has no meaningful t-1 state and is never trained on.
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
            tmp = Path(temp_directory)
            tmp.mkdir(parents=True, exist_ok=True)
            escaped_tmp = str(tmp).replace("'", "''")
            con.execute(f"SET temp_directory = '{escaped_tmp}'")
        columns = [r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]).fetchall()]
        required = {"ncodpers", "fecha_dato"}
        if missing := required.difference(columns):
            raise ValueError(f"Source is missing required columns: {sorted(missing)}")
        products = [c for c in columns if c.endswith("_ult1")]
        if not products:
            raise ValueError("Source has no product columns ending in '_ult1'.")
        profiles = [c for c in columns if c not in required and c not in products]
        q = lambda value: '"' + value.replace('"', '""') + '"'
        profile_sql = ", ".join(q(c) for c in profiles)
        previous_sql = ", ".join(
            f"CAST(COALESCE(LAG({q(p)}) OVER customer_time, 0) AS TINYINT) AS {q('prev_' + p)}"
            for p in products
        )
        label_sql = ", ".join(
            f"CAST(CASE WHEN LAG(fecha_dato) OVER customer_time IS NOT NULL "
            f"AND COALESCE(LAG({q(p)}) OVER customer_time, 0) = 0 AND COALESCE({q(p)}, 0) = 1 "
            f"THEN 1 ELSE 0 END AS TINYINT) AS {q('acq_' + p)}"
            for p in products
        )
        source_sql = str(source).replace("'", "''")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        temporary_sql = str(temporary).replace("'", "''")
        con.execute(f"""
            COPY (
                SELECT ncodpers, fecha_dato{', ' if profile_sql else ''}{profile_sql},
                       CAST(LAG(fecha_dato) OVER customer_time IS NOT NULL AS TINYINT) AS previous_observation,
                       {previous_sql}, {label_sql}
                FROM read_parquet('{source_sql}')
                WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato)
            ) TO '{temporary_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        temporary.replace(destination)
        return destination
    finally:
        con.close()
