"""Materialise frozen canonical and dual-Monetary RFM tiers.

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



# Frozen selection approved from RFM_Tiering_EDA; monetary is portfolio size.
CANONICAL_RFM_BOUNDARIES = {
    "recency": (1.7063406529255913, 3.5948354200815933, 6.349777031935352),
    "frequency": (0.5, 1.5, 2.931542461005199),
    "monetary": (0.0, 1.0, 2.0, 6.0),
}
CANONICAL_RFM_METHODS = {
    "recency": "kmeans_k4", "frequency": "kmeans_k4", "monetary": "rule_k5",
}
CANONICAL_RFM_TIER_FEATURE_NAMES = (
    "R_score", "F_score", "M_score", "R_tier", "F_tier", "M_tier",
    "RFM_score", "RFM_tier",
)
CANONICAL_RFM_TIER_INPUTS = (
    "ncodpers", "fecha_dato", "rfm_recency_months", "rfm_never_acquired_before",
    "rfm_frequency", "rfm_monetary",
)


def materialize_canonical_rfm_tiers(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    force_process: bool = False,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
) -> Path:
    """Add frozen portfolio RFM scores/tiers, preserving all unrelated columns.

    No fitting occurs. R=0 denotes NEVER; absent F/M remain NULL scores
    and receive MISSING tier labels. Raw features are retained unchanged.
    """
    source, destination = Path(source_path), Path(destination_path)
    con = _connect(memory_limit, temp_directory)
    try:
        columns = _columns(con, source)
        if not force_process and set(CANONICAL_RFM_TIER_FEATURE_NAMES).issubset(columns):
            return source
        if missing := set(CANONICAL_RFM_TIER_INPUTS).difference(columns):
            raise ValueError(f"Canonical RFM inputs missing: {sorted(missing)}")
        r = _tier_case_sql("rfm_recency_months", CANONICAL_RFM_BOUNDARIES["recency"], lower_is_better=True)
        f = _tier_case_sql("rfm_frequency", CANONICAL_RFM_BOUNDARIES["frequency"], lower_is_better=False)
        m = _tier_case_sql("rfm_monetary", CANONICAL_RFM_BOUNDARIES["monetary"], lower_is_better=False)
        excluded = columns.intersection(CANONICAL_RFM_TIER_FEATURE_NAMES)
        projection = "*" + (" EXCLUDE (" + ", ".join(quote(c) for c in sorted(excluded)) + ")" if excluded else "")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        con.execute(f"""
            COPY (
                WITH scored AS (
                    SELECT {projection},
                        CAST(CASE WHEN rfm_recency_months IS NULL OR rfm_never_acquired_before = 1
                             THEN 0 ELSE ({r}) END AS TINYINT) AS R_score,
                        CAST(CASE WHEN rfm_frequency IS NULL THEN NULL ELSE ({f}) END AS TINYINT) AS F_score,
                        CAST(CASE WHEN rfm_monetary IS NULL THEN NULL ELSE ({m}) END AS TINYINT) AS M_score
                    FROM read_parquet('{_path(source)}')
                ), tiered AS (
                    SELECT *,
                        CASE WHEN R_score = 0 THEN 'NEVER' ELSE 'R' || CAST(R_score AS VARCHAR) END AS R_tier,
                        CASE WHEN F_score IS NULL THEN 'MISSING' ELSE 'F' || CAST(F_score AS VARCHAR) END AS F_tier,
                        CASE WHEN M_score IS NULL THEN 'MISSING' ELSE 'M' || CAST(M_score AS VARCHAR) END AS M_tier
                    FROM scored
                )
                SELECT *,
                    CAST(R_score AS VARCHAR) || CAST(F_score AS VARCHAR) || CAST(M_score AS VARCHAR) AS RFM_score,
                    R_tier || '-' || F_tier || '-' || M_tier AS RFM_tier
                FROM tiered
            ) TO '{_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    finally:
        con.close()
    temporary.replace(destination)
    return destination


def build_renta_monetary_proxy(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    force_process: bool = False,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
) -> Path:
    """Project already-preprocessed canonical income; never impute again.

    The shared pipeline owns customer_history_geo_median_v1 preprocessing.
    Declared outputs: renta_filled, renta_imputation_method.
    """
    source, destination = Path(source_path), Path(destination_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    con = _connect(memory_limit, temp_directory)
    try:
        outputs = {"ncodpers", "fecha_dato", "renta_filled", "renta_imputation_method"}
        if missing := outputs.difference(_columns(con, source)):
            raise ValueError(f"Canonical income is not preprocessed: {sorted(missing)}. Run Data Pipeline first.")
        if source.resolve() == destination.resolve():
            return source
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        con.execute(f"COPY (SELECT ncodpers, fecha_dato, renta_filled, renta_imputation_method "
                    f"FROM read_parquet('{_path(source)}')) TO '{_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
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
