"""Materialise dual-Monetary RFM tiers for offline segmentation.

The EDA notebook evaluates several ways to select tier boundaries.  Production
code deliberately does not refit or select those boundaries: callers must
provide the frozen boundaries selected by an experiment.  This keeps model
features reproducible and prevents validation-period outcomes from affecting
the training representation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from numbers import Real
from math import isfinite

import duckdb

from src.features.customer_history import quote


RFM_TIERING_FEATURE_NAMES: tuple[str, ...] = (
    "renta_filled",
    "renta_imputation_method",
    "R_score",
    "F_score",
    "M_portfolio_score",
    "M_renta_score",
    "RFM_portfolio_score",
    "RFM_renta_score",
    "RFM_portfolio_tier",
    "RFM_renta_tier",
)


def build_renta_monetary_proxy(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    force_process: bool = False,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
) -> Path:
    """Materialise an offline income Monetary proxy for RFM segmentation.

    The priority is observed income, bidirectional customer fill, province
    median for Spanish residents (or country median elsewhere), then a global
    median. This is intentionally an offline segmentation artifact: its
    bidirectional fill must not be used as a temporal-model input.
    """
    source, destination = Path(source_path), Path(destination_path)
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source}")
    required = {"ncodpers", "fecha_dato", "renta"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    con = _connect(memory_limit, temp_directory)
    try:
        columns = _columns(con, source)
        if missing := required.difference(columns):
            raise ValueError(f"Source is missing renta-proxy columns: {sorted(missing)}")
        output_columns = {"ncodpers", "fecha_dato", "renta_filled", "renta_imputation_method"}
        if destination.is_file() and not force_process and output_columns.issubset(_columns(con, destination)):
            return destination
        province = "TRY_CAST(" + quote("cod_prov") + " AS INTEGER)" if "cod_prov" in columns else "NULL::INTEGER"
        country = "NULLIF(TRIM(CAST(" + quote("pais_residencia") + " AS VARCHAR)), '')" if "pais_residencia" in columns else "NULL::VARCHAR"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        con.execute(f"""
            COPY (
                WITH rows AS (
                    SELECT CAST(ncodpers AS BIGINT) AS ncodpers,
                           CAST(fecha_dato AS DATE) AS fecha_dato,
                           CASE WHEN TRY_CAST(renta AS DOUBLE) >= 0 THEN TRY_CAST(renta AS DOUBLE) END AS renta_raw,
                           {province} AS cod_prov, {country} AS pais_residencia
                    FROM read_parquet('{_path(source)}')
                ), history AS (
                    SELECT *, LAST_VALUE(renta_raw IGNORE NULLS) OVER (
                        PARTITION BY ncodpers ORDER BY fecha_dato
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS renta_past_fill,
                    FIRST_VALUE(renta_raw IGNORE NULLS) OVER (
                        PARTITION BY ncodpers ORDER BY fecha_dato
                        ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
                    ) AS renta_future_fill,
                    CASE WHEN pais_residencia = 'ES' AND cod_prov IS NOT NULL THEN 'PROV:' || CAST(cod_prov AS VARCHAR)
                         WHEN pais_residencia IS NOT NULL AND pais_residencia <> 'ES' THEN 'COUNTRY:' || pais_residencia END AS geo_group
                    FROM rows
                ), geographic AS (
                    SELECT *, MEDIAN(renta_raw) FILTER (WHERE renta_raw IS NOT NULL) OVER (
                        PARTITION BY fecha_dato, geo_group
                    ) AS geo_median,
                    MEDIAN(renta_raw) FILTER (WHERE renta_raw IS NOT NULL) OVER (
                        PARTITION BY fecha_dato
                    ) AS global_median
                    FROM history
                )
                SELECT ncodpers, fecha_dato,
                       CAST(COALESCE(renta_raw, renta_past_fill, renta_future_fill, geo_median, global_median) AS DOUBLE) AS renta_filled,
                       CASE WHEN renta_raw IS NOT NULL THEN 'observed'
                            WHEN renta_past_fill IS NOT NULL THEN 'past_fill'
                            WHEN renta_future_fill IS NOT NULL THEN 'future_fill'
                            WHEN geo_median IS NOT NULL THEN 'geography_median'
                            ELSE 'global_median' END AS renta_imputation_method
                FROM geographic
            ) TO '{_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        temporary.replace(destination)
        return destination
    finally:
        con.close()


def materialize_rfm_tiering_features(
    feature_store_path: str | Path,
    renta_proxy_path: str | Path,
    destination_path: str | Path,
    *,
    boundaries: Mapping[str, Sequence[Real]],
    force_process: bool = False,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
) -> Path:
    """Create RFM scores and profiles from frozen tier boundaries.

    ``boundaries`` must contain ``recency``, ``frequency``,
    ``monetary_portfolio`` and ``monetary_renta``.  Recency tiers are scored
    with lower values as better; the remaining dimensions use higher-is-better
    scores.  A customer with no prior acquisition receives ``R_score = 0``.
    """
    required_boundaries = {"recency", "frequency", "monetary_portfolio", "monetary_renta"}
    if missing := required_boundaries.difference(boundaries):
        raise ValueError(f"Missing frozen RFM tier boundaries: {sorted(missing)}")
    feature_store, renta_proxy, destination = Path(feature_store_path), Path(renta_proxy_path), Path(destination_path)
    if not feature_store.is_file() or not renta_proxy.is_file():
        missing = [str(path) for path in (feature_store, renta_proxy) if not path.is_file()]
        raise FileNotFoundError(f"Missing RFM tiering source(s): {missing}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    con = _connect(memory_limit, temp_directory)
    try:
        feature_columns, renta_columns = _columns(con, feature_store), _columns(con, renta_proxy)
        required_features = {"ncodpers", "fecha_dato", "rfm_recency_months", "rfm_never_acquired_before", "rfm_frequency", "rfm_monetary"}
        if missing := required_features.difference(feature_columns):
            raise ValueError(f"Feature store is missing RFM inputs: {sorted(missing)}")
        if missing := {"ncodpers", "fecha_dato", "renta_filled"}.difference(renta_columns):
            raise ValueError(f"Renta proxy is missing columns: {sorted(missing)}")
        if destination.is_file() and not force_process and set(RFM_TIERING_FEATURE_NAMES).issubset(_columns(con, destination)):
            return destination
        r_case = _tier_case_sql("recency", boundaries["recency"], lower_is_better=True)
        f_case = _tier_case_sql("frequency", boundaries["frequency"], lower_is_better=False)
        mp_case = _tier_case_sql("monetary_portfolio", boundaries["monetary_portfolio"], lower_is_better=False)
        mr_case = _tier_case_sql("monetary_renta_log", boundaries["monetary_renta"], lower_is_better=False)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        con.execute(f"""
            COPY (
                WITH base AS (
                    SELECT f.ncodpers, f.fecha_dato, f.rfm_recency_months AS recency,
                           f.rfm_never_acquired_before AS never_acquired_before,
                           f.rfm_frequency AS frequency, f.rfm_monetary AS monetary_portfolio,
                           r.renta_filled, r.renta_imputation_method,
                           LN(1 + r.renta_filled) AS monetary_renta_log
                    FROM read_parquet('{_path(feature_store)}') f
                    LEFT JOIN read_parquet('{_path(renta_proxy)}') r USING (ncodpers, fecha_dato)
                ), scored AS (
                    SELECT *, CASE WHEN recency IS NULL OR never_acquired_before = 1 THEN 0 ELSE ({r_case}) END AS R_score,
                           CASE WHEN frequency IS NULL THEN NULL ELSE ({f_case}) END AS F_score,
                           CASE WHEN monetary_portfolio IS NULL THEN NULL ELSE ({mp_case}) END AS M_portfolio_score,
                           CASE WHEN monetary_renta_log IS NULL THEN NULL ELSE ({mr_case}) END AS M_renta_score
                    FROM base
                )
                SELECT *,
                       CASE WHEN R_score IS NOT NULL AND F_score IS NOT NULL AND M_portfolio_score IS NOT NULL
                            THEN CAST(R_score AS VARCHAR) || CAST(F_score AS VARCHAR) || CAST(M_portfolio_score AS VARCHAR) END AS RFM_portfolio_score,
                       CASE WHEN R_score IS NOT NULL AND F_score IS NOT NULL AND M_renta_score IS NOT NULL
                            THEN CAST(R_score AS VARCHAR) || CAST(F_score AS VARCHAR) || CAST(M_renta_score AS VARCHAR) END AS RFM_renta_score,
                       CASE WHEN R_score = 0 THEN 'NEVER' ELSE 'R' || CAST(R_score AS VARCHAR) END || '-F' || CAST(F_score AS VARCHAR) || '-M' || CAST(M_portfolio_score AS VARCHAR) AS RFM_portfolio_tier,
                       CASE WHEN R_score = 0 THEN 'NEVER' ELSE 'R' || CAST(R_score AS VARCHAR) END || '-F' || CAST(F_score AS VARCHAR) || '-M' || CAST(M_renta_score AS VARCHAR) AS RFM_renta_tier
                FROM scored
            ) TO '{_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        temporary.replace(destination)
        return destination
    finally:
        con.close()


def _tier_case_sql(column: str, boundaries: Sequence[Real], *, lower_is_better: bool) -> str:
    values = [float(value) for value in boundaries]
    if values != sorted(values) or any(not isfinite(value) for value in values):
        raise ValueError(f"Tier boundaries for {column} must be sorted finite values.")
    tier_count = len(values) + 1
    clauses = []
    for index, value in enumerate(values, start=1):
        score = tier_count - index + 1 if lower_is_better else index
        clauses.append(f"WHEN {quote(column)} <= {value:.12g} THEN {score}")
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
