"""Monthly rarity/income weights using filled income and observed lagged ownership.

Income follows the canonical offline fill policy. Weight fitting excludes rows
at or after the target month; ownership always comes from a prior record.
"""
from __future__ import annotations
import math
import os
from datetime import date
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from src.products import PRODUCT_COLUMNS
from src.features.customer_history import quote
from .tiering_utils import _connect

FEATURE_NAME = "rfm_monetary_rarity_income"
FORMULA_VERSION = "v4_linear_mu_minus_2_pop_variance"
PRODUCTS = PRODUCT_COLUMNS


def validate_weight_sidecar(weights: pd.DataFrame) -> None:
    """Reject stale formulas, incomplete matrices and inconsistent V4 algebra."""
    required = {"formula_version", "income_column", "target_month", "product",
                "mean_percentile", "sd_percentile", "linear_location", "sd_penalty",
                "income_score", "rarity", "product_weight"}
    if missing := required - set(weights.columns):
        raise ValueError(f"Weight sidecar missing V4 columns: {sorted(missing)}; force a refit")
    if weights.empty or not weights.formula_version.eq(FORMULA_VERSION).all():
        raise ValueError("Weight sidecar formula version is incompatible; force a refit")
    if not weights.income_column.eq("renta_filled").all():
        raise ValueError("Weight sidecar must use renta_filled")
    if weights[["target_month", "product"]].isna().any().any() or weights.duplicated(["target_month", "product"]).any():
        raise ValueError("Invalid or duplicate target_month/product keys")
    for _, group in weights.groupby("target_month"):
        if set(group["product"]) != set(PRODUCTS):
            raise ValueError("Monthly product weights must contain exactly the canonical products")
    if not np.isfinite(weights[["rarity", "income_score", "product_weight"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite monthly product weights")
    if not weights.rarity.between(0, 1).all():
        raise ValueError("Rarity outside [0,1]")
    valid = weights.mean_percentile.notna() & weights.sd_percentile.notna()
    for row in weights.loc[valid].itertuples():
        linear, penalty, score = income_score(row.mean_percentile, row.sd_percentile)
        if not np.allclose([row.linear_location, row.sd_penalty, row.income_score],
                           [linear, penalty, score], rtol=1e-9, atol=1e-10):
            raise ValueError("Weight sidecar income score does not match V4")
    if not weights.loc[~valid, "income_score"].eq(0).all():
        raise ValueError("Missing holder-income moments require neutral C=0")
    if not np.allclose(weights.product_weight, 1 + weights.rarity * weights.income_score,
                       rtol=1e-9, atol=1e-10):
        raise ValueError("Product weights must satisfy w=1+R*C without an extra IDF factor")
    if not weights.product_weight.between(-1e-9, 2 + 1e-9).all():
        raise ValueError("Product weights outside [0,2]")

def lit(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"

def parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    return con.execute(f"DESCRIBE SELECT * FROM read_parquet({lit(path)})").fetchdf()["column_name"].tolist()

def validate_inputs(con: duckdb.DuckDBPyConnection, source: Path) -> dict:
    """Validate grain and lagged-only input contract before feature engineering."""
    columns = set(parquet_columns(con, source))
    necessary = {"ncodpers", "fecha_dato", *("prev_" + p for p in PRODUCTS)}
    missing = necessary - columns
    if missing:
        raise ValueError(f"Canonical checkpoint missing required columns: {sorted(missing)}")
    income = "renta_filled"
    if income not in columns:
        raise ValueError("renta_filled is required; run canonical income preprocessing first")
    n, duplicates, months, missing_keys = con.execute(f"""
        SELECT COUNT(*) AS n,
        (SELECT COUNT(*) FROM (
            SELECT ncodpers, CAST(fecha_dato AS DATE) d
            FROM read_parquet({lit(source)}) GROUP BY 1,2 HAVING COUNT(*)>1
        )) AS duplicates,
        COUNT(DISTINCT CAST(fecha_dato AS DATE)) AS months,
        COUNT(*) FILTER (WHERE ncodpers IS NULL OR fecha_dato IS NULL) AS missing_keys
        FROM read_parquet({lit(source)})
    """).fetchone()
    if not n or duplicates or missing_keys:
        raise ValueError(f"Invalid source: rows={n}, duplicate customer-month keys={duplicates}, missing keys={missing_keys}")
    return {"row_count": int(n), "month_count": int(months), "income_column": income,
            "source_path": str(source)}

def income_score(mu: float, sigma: float) -> tuple[float, float, float]:
    """Return (linear_location, sd_penalty, C) with C in [-1,1]."""
    if not np.isfinite(mu) or not np.isfinite(sigma):
        return (float("nan"), float("nan"), 0.0)  # neutral for unavailable holder income
    if not 0 <= mu <= 1 or not 0 <= sigma <= 0.5 + 1e-9:
        raise ValueError(f"Out-of-domain location/dispersion: {mu}, {sigma}")
    if sigma**2 > mu * (1 - mu) + 1e-10:
        raise ValueError(f"Infeasible percentile moments: mu={mu}, sd={sigma}")
    linear = 2.0 * mu - 1.0
    penalty = 2.0 * sigma**2
    score = float(linear - penalty)
    if not -1 - 1e-10 <= score <= 1 + 1e-10:
        raise AssertionError(f"Income score outside analytic bounds: {score}")
    return (linear, penalty, score)

def fit_product_weights_asof(con: duckdb.DuckDBPyConnection, source: Path, month: date,
                             income_col: str) -> pd.DataFrame:
    """As-of fit: observations tau < target month, persona(tau), product(prev_tau).

    Rarity denominator is historical rows with at least one observed lagged product;
    holder numerator is rows where the specific lagged product=1. The income CDF
    uses the same eligible history. Missing income
    is excluded ONLY from the percentile CDF and holder income moments.
    """
    if income_col != "renta_filled":
        raise ValueError("Rarity weights require renta_filled; raw income is not supported")
    target = pd.Timestamp(month).strftime("%Y-%m-%d")
    history = f"read_parquet({lit(source)})"
    where = f"CAST(fecha_dato AS DATE) < DATE {lit(target)}"
    eligible = " OR ".join(f"{quote('prev_' + p)} IN (0,1)" for p in PRODUCTS)
    counts = ["COUNT(*) FILTER (WHERE (" + eligible + ")) AS n_eligible"]
    for idx,p in enumerate(PRODUCTS):
        q = quote("prev_" + p)
        counts += [f"COUNT(*) FILTER (WHERE {q} IN (0,1)) AS obs_{idx}",
                   f"COUNT(*) FILTER (WHERE {q}=1) AS hold_{idx}"]
    count_sql = f"SELECT {', '.join(counts)} FROM {history} WHERE {where}"
    count_data = con.execute(count_sql).fetchone()
    eligible_n = int(count_data[0])
    observed = [int(count_data[1 + 2*i]) for i in range(len(PRODUCTS))]
    holders = [int(count_data[2 + 2*i]) for i in range(len(PRODUCTS))]

    # Global empirical midrank percentiles over eligible historical ownership rows.
    # rank-1 is the number strictly below the income value, count_ties/2 puts
    # tied values at the midpoint of their percentile mass.
    income_cast = f"TRY_CAST({quote(income_col)} AS DOUBLE)"
    exprs = ["COUNT(*) AS global_income_n"]
    for idx,p in enumerate(PRODUCTS):
        lag = quote("prev_" + p)
        exprs.extend([
            f"COUNT(*) FILTER (WHERE {lag}=1) AS income_holders_{idx}",
            f"AVG(u) FILTER (WHERE {lag}=1) AS mu_{idx}",
            f"STDDEV_POP(u) FILTER (WHERE {lag}=1) AS sigma_{idx}",
        ])
    income_sql = f"""
        WITH valid_income AS (
          SELECT {income_cast} AS income,
                 {', '.join(quote('prev_'+p) for p in PRODUCTS)}
          FROM {history}
          WHERE {where} AND ({eligible}) AND {income_cast} IS NOT NULL
            AND isfinite({income_cast}) AND {income_cast} >= 0
        ), ranked AS (
          SELECT *,
            (RANK() OVER (ORDER BY income)-1
             + COUNT(*) OVER (PARTITION BY income)/2.0)
            / NULLIF(COUNT(*) OVER (),0) AS u
          FROM valid_income
        )
        SELECT {', '.join(exprs)} FROM ranked
    """
    # No historical records/income: neutral weights and documented cold start.
    if eligible_n == 0:
        return pd.DataFrame([{
            "formula_version": FORMULA_VERSION, "income_column": "renta_filled",
            "target_month": target, "product": p, "n_eligible": 0,
            "n_observed": 0, "n_holders": 0, "n_income_holders": 0,
            "global_income_n": 0, "ownership_rate": float("nan"),
            "idf": float("nan"), "rarity": 0.0,
            "mean_percentile": float("nan"), "sd_percentile": float("nan"),
            "linear_location": float("nan"), "sd_penalty": float("nan"),
            "income_score": 0.0, "product_weight": 1.0,
            "status": "cold_start_no_lagged_history",
        } for p in PRODUCTS])
    row = con.execute(income_sql).fetchone()
    income_n = int(row[0])
    idfs = np.array([1.0 + math.log((eligible_n+1)/(h+1)) for h in holders], dtype=float)
    idf_span = float(idfs.max()-idfs.min())
    rarities = (idfs-idfs.min())/idf_span if idf_span > 0 else np.zeros_like(idfs)
    records = []
    for i,p in enumerate(PRODUCTS):
        income_holders = int(row[1+3*i])
        mu = float(row[2+3*i]) if row[2+3*i] is not None else float("nan")
        sd = float(row[3+3*i]) if row[3+3*i] is not None else float("nan")
        linear, penalty, C = income_score(mu,sd)
        R = float(rarities[i])
        w = float(1.0 + R * C)
        status = "ok" if income_holders > 0 else "no_valid_holder_income_neutral"
        records.append({
            "formula_version": FORMULA_VERSION, "income_column": "renta_filled",
            "target_month": target, "product": p, "n_eligible": eligible_n,
            "n_observed": observed[i], "n_holders": holders[i],
            "n_income_holders": income_holders, "global_income_n": income_n,
            "ownership_rate": holders[i]/observed[i] if observed[i] else float("nan"),
            "idf": float(idfs[i]), "rarity": R,
            "mean_percentile": mu, "sd_percentile": sd,
            "linear_location": linear, "sd_penalty": penalty,
            "income_score": C, "product_weight": w, "status": status,
        })
    result = pd.DataFrame(records)
    validate_weight_sidecar(result)
    return result

def fit_monthly_product_weights(con: duckdb.DuckDBPyConnection, source: Path,
                               weights_path: Path, income_col: str,
                               force_process: bool = False,
                               target_months=None) -> pd.DataFrame:
    """Fit weights for every observed target month; skip complete sidecar by default."""
    if weights_path.is_file() and not force_process:
        cached = pd.read_parquet(weights_path)
        validate_weight_sidecar(cached)
        cached_months = set(pd.to_datetime(cached["target_month"]).dt.date)
        expected_months = ({pd.Timestamp(m).date() for m in target_months} if target_months is not None
                           else {pd.Timestamp(r[0]).date() for r in con.execute(
                               f"SELECT DISTINCT CAST(fecha_dato AS DATE) FROM read_parquet({lit(source)})"
                           ).fetchall()})
        if cached_months != expected_months:
            raise ValueError("Monthly weight cache has different target months; force a refit")
        print(f"[SKIP] Verified {FORMULA_VERSION} product-weight sidecar: {weights_path}")
        return cached
    months = [pd.Timestamp(v[0]).date() for v in con.execute(f"""
        SELECT DISTINCT CAST(fecha_dato AS DATE) AS d
        FROM read_parquet({lit(source)}) ORDER BY d
    """).fetchall()]
    if target_months is not None:
        months = sorted({pd.Timestamp(m).date() for m in target_months})
    rows = []
    for k, month in enumerate(months, 1):
        print(f"[FIT] {k}/{len(months)} target month = {month}")
        rows.append(fit_product_weights_asof(con, source, month, income_col))
    weights = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["target_month", "product", "product_weight"])
    if len(weights) != len(PRODUCTS) * len(months):
        raise AssertionError("Sidecar must have exactly one row per product / target month")
    validate_weight_sidecar(weights)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = weights_path.with_name(weights_path.stem + ".pending.parquet")
    weights.to_parquet(tmp, index=False)
    os.replace(tmp, weights_path)
    return weights

def materialize_monetary(con: duckdb.DuckDBPyConnection, source: Path,
                         weights_path: Path, output: Path, force_process: bool = False) -> Path:
    """Append one feature via DuckDB COPY, preserving source columns unchanged.

    If ALL prev states are missing, output NULL (unknown portfolio) rather than
    misrepresenting missing history as 0. Partial unknown product flags contribute
    0 but remain auditable through original prev columns.
    """
    if output.is_file() and not force_process:
        available = set(parquet_columns(con, output))
        if FEATURE_NAME in available:
            print(f"[SKIP] Feature already materialized: {output}")
            return output
    validate_weight_sidecar(pd.read_parquet(weights_path))
    missing_weights = con.execute(f"""SELECT COUNT(*) FROM (
        SELECT DISTINCT CAST(fecha_dato AS DATE) AS target_month FROM read_parquet({lit(source)})
    ) m CROSS JOIN (SELECT UNNEST(?) AS product) p
    LEFT JOIN read_parquet({lit(weights_path)}) w USING (target_month, product)
    WHERE w.product_weight IS NULL OR NOT isfinite(w.product_weight)
       OR w.product_weight < -1e-9 OR w.product_weight > 2 + 1e-9""", [list(PRODUCTS)]).fetchone()[0]
    duplicate_weights = con.execute(f"""SELECT COUNT(*) FROM (
        SELECT target_month, product FROM read_parquet({lit(weights_path)})
        GROUP BY 1,2 HAVING COUNT(*) <> 1)""").fetchone()[0]
    if missing_weights or duplicate_weights:
        raise ValueError("Incomplete or invalid monthly product weights")
    con.execute("DROP TABLE IF EXISTS _weight_matrix_tmp")
    wide_cols = ",\n".join(f"MAX(CASE WHEN product={lit(p)} THEN product_weight END) AS {quote('w'+str(i))}"
                            for i,p in enumerate(PRODUCTS))
    con.execute(f"""CREATE TEMP TABLE _weight_matrix_tmp AS
        SELECT CAST(target_month AS DATE) AS target_month, {wide_cols}
        FROM read_parquet({lit(weights_path)}) GROUP BY 1""")
    input_cols = set(parquet_columns(con, source))
    from_cols = 's.* EXCLUDE (' + quote(FEATURE_NAME) + ')' if FEATURE_NAME in input_cols else 's.*'
    has_previous = " OR ".join(f"s.{quote('prev_'+p)} IN (0,1)" for p in PRODUCTS)
    terms = [f"CASE WHEN s.{quote('prev_'+p)}=1 THEN w.{quote('w'+str(i))} ELSE 0.0 END"
             for i,p in enumerate(PRODUCTS)]
    score = " +\n                 ".join(terms)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + ".pending.parquet")
    try:
        con.execute(f"""COPY (
            SELECT {from_cols},
                   CASE WHEN ({has_previous}) THEN CAST(({score}) AS DOUBLE)
                        ELSE NULL::DOUBLE END AS {quote(FEATURE_NAME)}
            FROM read_parquet({lit(source)}) s
            LEFT JOIN _weight_matrix_tmp w
              ON CAST(s.fecha_dato AS DATE) = w.target_month
        ) TO {lit(tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
        input_n = con.execute(f"SELECT COUNT(*) FROM read_parquet({lit(source)})").fetchone()[0]
        output_n, invalid = con.execute(f"""
            SELECT COUNT(*), COUNT(*) FILTER (
                WHERE {quote(FEATURE_NAME)} < -1e-9 OR {quote(FEATURE_NAME)} > {2 * len(PRODUCTS)} + 1e-9
                   OR (NOT isfinite({quote(FEATURE_NAME)}) AND {quote(FEATURE_NAME)} IS NOT NULL))
            FROM read_parquet({lit(tmp)})""").fetchone()
        if input_n != output_n or invalid:
            raise AssertionError(f"Output failed validation: input={input_n}, output={output_n}, invalid={invalid}")
        os.replace(tmp, output)
    finally:
        con.execute("DROP TABLE IF EXISTS _weight_matrix_tmp")
        if tmp.is_file():
            tmp.unlink()
    return output


def monetary_tier_sql(column, boundaries):
    """Reserved until rarity tier boundaries and a method are approved by EDA."""
    raise NotImplementedError("rarity_rental tiering is a placeholder; run EDA to select a tiering method first")


def build_rarity_monetary_proxy(
    source_path, destination_path, *, force_process=False, memory_limit=None,
    threads=2, temp_directory=None, weights_path=None,
):
    """Append rarity monetary to a full snapshot, keeping existing columns intact.

    Reconstruct nullable lagged states from original products instead of using
    legacy zero-filled prev_* columns. Intermediates are private to this call.
    The caller owns force-feature/function resolution and checkpoint publication.
    """
    source, destination = Path(source_path), Path(destination_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Use a new candidate path; never overwrite the source checkpoint")
    con = _connect(memory_limit, temp_directory)
    con.execute("SET threads = ?", [threads])
    con.execute("SET preserve_insertion_order = false")
    ownership = destination.with_suffix(".ownership.pending.parquet")
    scored = destination.with_suffix(".scored.pending.parquet")
    pending = destination.with_suffix(".pending.parquet")
    try:
        columns = set(parquet_columns(con, source))
        if FEATURE_NAME in columns and not force_process:
            return source
        required = {"ncodpers", "fecha_dato", "renta_filled", *PRODUCTS}
        if missing := required - columns:
            raise ValueError(f"Rarity monetary missing inputs: {sorted(missing)}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        lagged = ", ".join(
            f"LAG(TRY_CAST({quote(p)} AS TINYINT)) OVER customer_time AS {quote('prev_' + p)}"
            for p in PRODUCTS
        )
        con.execute(f"""COPY (
            SELECT ncodpers, fecha_dato, renta_filled, {lagged}
            FROM read_parquet({lit(source)})
            WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato)
        ) TO {lit(ownership)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
        validate_inputs(con, ownership)
        weights = Path(weights_path) if weights_path else destination.with_suffix(".weights.parquet")
        # A RUN always fits the current input, never an unrelated sidecar.
        fit_monthly_product_weights(con, ownership, weights, "renta_filled", force_process=True)
        materialize_monetary(con, ownership, weights, scored, force_process=True)
        projection = "s.*" + (f" EXCLUDE ({quote(FEATURE_NAME)})" if FEATURE_NAME in columns else "")
        con.execute(f"""COPY (
            SELECT {projection}, r.{quote(FEATURE_NAME)}
            FROM read_parquet({lit(source)}) s
            JOIN read_parquet({lit(scored)}) r USING (ncodpers, fecha_dato)
        ) TO {lit(pending)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
        os.replace(pending, destination)
        return destination
    finally:
        con.close()
        for path in (ownership, scored, pending):
            path.unlink(missing_ok=True)
