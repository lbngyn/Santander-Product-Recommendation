"""Approved RFM income preprocessing, shared by canonical RFM and CLV.

This is an offline imputation rule: guarded backward fill and full-history
geographic statistics can use future observations. No clipping is applied.
"""
from pathlib import Path
import duckdb

RENTA_OUTPUTS = ("renta_raw", "renta_filled", "renta_imputation_method")

def preprocess_renta_monetary(source_path: str | Path, destination_path: str | Path, *,
                             force_process: bool = False, memory_limit: str | None = None,
                             temp_directory: str | Path | None = None) -> Path:
    """Materialize the exact customer_history_geo_median_v1 RFM rule.

    Declared outputs: RENTA_OUTPUTS. Existing source outputs skip processing.
    Preserve every unrelated source column and customer-month key.
    """
    source, destination = Path(source_path), Path(destination_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with duckdb.connect() as con:
        if memory_limit:
            con.execute("SET memory_limit = ?", [memory_limit])
        if temp_directory:
            Path(temp_directory).mkdir(parents=True, exist_ok=True)
            con.execute("SET temp_directory = ?", [str(temp_directory)])
        columns = [r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]).fetchall()]
        missing = {"ncodpers", "fecha_dato", "renta", "cod_prov", "pais_residencia"}.difference(columns)
        if missing:
            raise ValueError(f"Renta preprocessing missing inputs: {sorted(missing)}")
        if not force_process and set(RENTA_OUTPUTS).issubset(columns):
            print("[processing] SKIP renta_monetary: all declared outputs exist")
            return source
        if source.resolve() == destination.resolve():
            raise ValueError("Use a new candidate path; never overwrite the source checkpoint.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_sql = str(source).replace("'", "''")
        # Recompute from preserved upstream values, not an already-filled renta.
        renta_input = 'renta_raw' if 'renta_raw' in columns else 'renta'
        query = f"""
            WITH rows AS (
                SELECT ncodpers, fecha_dato,
                       TRY_CAST({renta_input} AS DOUBLE) AS renta_raw,
                       TRY_CAST(cod_prov AS INTEGER) AS cod_prov,
                       NULLIF(TRIM(CAST(pais_residencia AS VARCHAR)), '') AS pais_residencia
                FROM read_parquet('{source_sql}')
            ), customer_stats AS (
                SELECT ncodpers,
                       COUNT(DISTINCT ROUND(renta_raw, 2)) FILTER (WHERE renta_raw IS NOT NULL)::INTEGER AS renta_n_unique,
                       MEDIAN(renta_raw) FILTER (WHERE renta_raw >= 0)::DOUBLE AS renta_customer,
                       CASE WHEN ARG_MAX(pais_residencia, fecha_dato) FILTER (WHERE pais_residencia IS NOT NULL) = 'ES'
                                 AND ARG_MAX(cod_prov, fecha_dato) FILTER (WHERE cod_prov IS NOT NULL) IS NOT NULL
                            THEN 'PROV:' || CAST(ARG_MAX(cod_prov, fecha_dato) FILTER (WHERE cod_prov IS NOT NULL) AS VARCHAR)
                            WHEN ARG_MAX(pais_residencia, fecha_dato) FILTER (WHERE pais_residencia IS NOT NULL) <> 'ES'
                            THEN 'COUNTRY:' || ARG_MAX(pais_residencia, fecha_dato) FILTER (WHERE pais_residencia IS NOT NULL)
                       END AS geo_group
                FROM rows GROUP BY ncodpers
            ), persistence AS (
                SELECT COALESCE(AVG(CAST(n_unique = 1 AS DOUBLE)), 0) >= 0.99 AS allow_bfill
                FROM (
                    SELECT ncodpers, COUNT(*) AS n_observed, COUNT(DISTINCT ROUND(renta_raw, 2)) AS n_unique
                    FROM rows WHERE renta_raw >= 0 GROUP BY ncodpers HAVING COUNT(*) >= 2
                )
            ), profile AS (
                SELECT * FROM customer_stats
            ),
            geo_stats AS (
                SELECT
                    geo_group,
                    MEDIAN(renta_customer)::DOUBLE AS geo_median
                FROM profile
                WHERE renta_customer IS NOT NULL
                  AND renta_customer >= 0
                  AND geo_group IS NOT NULL
                GROUP BY 1
            ),
            global_stats AS (
                SELECT MEDIAN(renta_customer)::DOUBLE AS global_median
                FROM profile
                WHERE renta_customer IS NOT NULL
                  AND renta_customer >= 0
            ),
            history AS (
                SELECT
                    r.*,
                    p.renta_n_unique,
                    p.geo_group AS customer_geo_group,
                    LAST_VALUE(r.renta_raw IGNORE NULLS) OVER (
                        PARTITION BY r.ncodpers
                        ORDER BY r.fecha_dato
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS renta_past_fill,
                    FIRST_VALUE(r.renta_raw IGNORE NULLS) OVER (
                        PARTITION BY r.ncodpers
                        ORDER BY r.fecha_dato
                        ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
                    ) AS renta_backward_fill,
                    CASE
                        WHEN r.pais_residencia = 'ES' AND r.cod_prov IS NOT NULL
                            THEN 'PROV:' || CAST(r.cod_prov AS VARCHAR)
                        WHEN r.pais_residencia IS NOT NULL AND r.pais_residencia <> 'ES'
                            THEN 'COUNTRY:' || r.pais_residencia
                        ELSE NULL
                    END AS row_geo_group
                FROM rows r
                LEFT JOIN profile p USING (ncodpers)
            ),
            enriched AS (
                SELECT
                    h.*,
                    COALESCE(h.row_geo_group, h.customer_geo_group) AS geo_group,
                    g.geo_median,
                    gs.global_median
                FROM history h
                LEFT JOIN geo_stats g
                  ON COALESCE(h.row_geo_group, h.customer_geo_group) = g.geo_group
                CROSS JOIN global_stats gs
            )
            SELECT
                ncodpers,
                fecha_dato,
                renta_raw,
                cod_prov,
                pais_residencia,
                geo_group,
                COALESCE(
                    renta_raw,
                    renta_past_fill,
                    CASE
                        WHEN (SELECT allow_bfill FROM persistence)
                         AND COALESCE(renta_n_unique, 0) <= 1
                        THEN renta_backward_fill
                    END,
                    geo_median,
                    global_median
                )::DOUBLE AS renta_filled,
                CASE
                    WHEN renta_raw IS NOT NULL THEN 'observed'
                    WHEN renta_past_fill IS NOT NULL THEN 'past_fill'
                    WHEN (SELECT allow_bfill FROM persistence)
                     AND COALESCE(renta_n_unique, 0) <= 1
                     AND renta_backward_fill IS NOT NULL THEN 'backward_fill'
                    WHEN geo_median IS NOT NULL THEN 'geography_median'
                    ELSE 'global_median'
                END AS renta_imputation_method
            FROM enriched

        """
        keep = ', '.join('s."' + c.replace('"', '""') + '"' for c in columns if c not in RENTA_OUTPUTS)
        output = ', '.join('f."' + c + '"' for c in RENTA_OUTPUTS)
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        destination_sql = str(temporary).replace("'", "''")
        con.execute(f"COPY (WITH income AS ({query}) SELECT {keep}, {output} "
                    f"FROM read_parquet('{source_sql}') s JOIN income f USING (ncodpers, fecha_dato)) "
                    f"TO '{destination_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        valid = con.execute("SELECT COUNT(*) = COUNT(DISTINCT (ncodpers, fecha_dato)) "
                            "AND COUNT(*) = COUNT(ncodpers) AND COUNT(*) = COUNT(fecha_dato) "
                            "FROM read_parquet(?)", [str(temporary)]).fetchone()[0]
        if not valid:
            temporary.unlink(missing_ok=True)
            raise ValueError("Renta output violates unique, non-null customer-month grain")
        temporary.replace(destination)
        print("[processing] RUN renta_monetary | declared-output validation=PASS")
    return destination
