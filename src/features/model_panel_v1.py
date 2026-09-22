"""Original v1 panel: profiles at t and nearest prior product states only."""
from __future__ import annotations

from pathlib import Path

import duckdb

from src.features.persona import PERSONA_FEATURES, persona_feature_projection_sql, raw_persona_projection_sql


def build_model_panel_v1(source_path: str | Path, destination_path: str | Path, *, force_process: bool = False, memory_limit: str | None = None, temp_directory: str | Path | None = None) -> Path:
    """Materialise the immutable pre-history-feature v1 model panel."""
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
            temp = Path(temp_directory); temp.mkdir(parents=True, exist_ok=True)
            con.execute(f"SET temp_directory = '{str(temp).replace(chr(39), chr(39) * 2)}'")
        columns = [row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]).fetchall()]
        required = {"ncodpers", "fecha_dato"}
        if missing := required.difference(columns):
            raise ValueError(f"Source is missing required columns: {sorted(missing)}")
        products = [column for column in columns if column.endswith("_ult1")]
        if not products:
            raise ValueError("Source has no product columns ending in '_ult1'.")
        q = lambda value: '"' + value.replace('"', '""') + '"'
        profile_sql = ", ".join(q(column) for column in PERSONA_FEATURES)
        raw_persona_sql = raw_persona_projection_sql("source", columns)
        persona_sql = persona_feature_projection_sql()
        previous_sql = ", ".join(f"CAST(COALESCE(LAG({q(product)}) OVER customer_time, 0) AS TINYINT) AS {q('prev_' + product)}" for product in products)
        label_sql = ", ".join(
            f"CAST(CASE WHEN LAG(fecha_dato) OVER customer_time IS NOT NULL AND COALESCE(LAG({q(product)}) OVER customer_time, 0) = 0 AND COALESCE({q(product)}, 0) = 1 THEN 1 ELSE 0 END AS TINYINT) AS {q('acq_' + product)}"
            for product in products
        )
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        con.execute(f"""
            COPY (
                WITH source_rows AS (
                    SELECT source.ncodpers, source.fecha_dato, {raw_persona_sql},
                           {', '.join('source.' + q(product) for product in products)}
                    FROM read_parquet('{str(source).replace("'", "''")}') AS source
                )
                SELECT ncodpers, fecha_dato{', ' if profile_sql else ''}{profile_sql},
                       CAST(LAG(fecha_dato) OVER customer_time IS NOT NULL AS TINYINT) AS previous_observation,
                       {previous_sql}, {label_sql}
                FROM source_rows
                WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato),
                       prior_customer_rows AS (PARTITION BY ncodpers ORDER BY fecha_dato ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
            ) TO '{str(temporary).replace("'", "''")}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        temporary.replace(destination)
        return destination
    finally:
        con.close()
