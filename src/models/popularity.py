"""Leakage-safe popularity baseline for Santander product recommendation."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


def run_popularity_baseline(
    train_path: str | Path,
    output_dir: str | Path,
    *,
    validation_date: str,
    top_k: int = 7,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
    threads: int | None = None,
) -> dict[str, object]:
    """Score a temporal holdout using only product acquisitions before it.

    For every customer present on ``validation_date``, the model receives that
    date's non-product profile fields and the customer's immediately preceding
    product state.  It ranks products by historical 0->1 acquisition count,
    removes products already owned in that preceding state, and emits the
    first ``top_k`` remaining products.  The 24 product flags on the holdout
    date are written separately as the target, so they cannot enter scoring.
    """
    source, destination = Path(train_path), Path(output_dir)
    scratch_dir = Path(temp_directory) if temp_directory else destination
    if not source.is_file():
        raise FileNotFoundError(f"Train checkpoint not found: {source}")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    destination.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        _configure(con, memory_limit, temp_directory, threads)
        columns = _columns(con, source)
        products = [column for column in columns if column.endswith("_ult1")]
        if not products:
            raise ValueError("Train checkpoint has no product columns ending in '_ult1'.")
        _require(columns, {"ncodpers", "fecha_dato"})
        validation_rows = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?) WHERE CAST(fecha_dato AS DATE) = CAST(? AS DATE)",
            [str(source), validation_date],
        ).fetchone()[0]
        if not validation_rows:
            raise ValueError(f"No rows found for validation_date={validation_date}.")

        current_product_values = ",\n                    ".join(
            f"('{product}', TRY_CAST(\"{product}\" AS TINYINT), TRY_CAST(\"previous_{product}\" AS TINYINT))" for product in products
        )
        ownership_case = "CASE p.product " + " ".join(
            f"WHEN '{product}' THEN TRY_CAST(h.\"previous_{product}\" AS TINYINT)" for product in products
        ) + " ELSE 0 END"
        common = f"""
            WITH ordered AS (
                SELECT *, LAG(fecha_dato) OVER customer_time AS previous_date,
                    {', '.join(f'LAG("{p}") OVER customer_time AS "previous_{p}"' for p in products)}
                FROM read_parquet('{_path(source)}')
                WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato)
            ),
            historical_acquisitions AS (
                SELECT product, SUM(CASE WHEN value = 1 AND COALESCE(previous_value, 0) = 0 THEN 1 ELSE 0 END)::BIGINT AS new_purchase_count
                FROM ordered
                CROSS JOIN LATERAL (VALUES {current_product_values}) AS product_values(product, value, previous_value)
                WHERE CAST(fecha_dato AS DATE) < CAST('{validation_date}' AS DATE)
                  AND previous_date IS NOT NULL
                GROUP BY product
            ),
            popularity AS (
                SELECT product, new_purchase_count, ROW_NUMBER() OVER (ORDER BY new_purchase_count DESC, product) AS popularity_rank
                FROM historical_acquisitions
            ),
            holdout AS (
                SELECT * FROM ordered WHERE CAST(fecha_dato AS DATE) = CAST('{validation_date}' AS DATE)
            )
        """
        _copy(con, common + "SELECT product, new_purchase_count, popularity_rank FROM popularity ORDER BY popularity_rank", destination / "popularity_ranking.parquet")
        actual_path = scratch_dir / "validation_actual_additions.parquet"
        _copy(con, common + f"""
            , eligible AS (
                SELECT h.ncodpers, h.fecha_dato, p.product, p.popularity_rank,
                       ROW_NUMBER() OVER (PARTITION BY h.ncodpers ORDER BY p.popularity_rank) AS recommendation_rank
                FROM holdout AS h
                CROSS JOIN popularity AS p
                WHERE COALESCE({ownership_case}, 0) = 0
            )
            SELECT ncodpers, fecha_dato,
                   string_agg(p.product, ' ' ORDER BY p.popularity_rank) AS added_products
            FROM eligible AS p WHERE recommendation_rank <= {top_k}
            GROUP BY ncodpers, fecha_dato
        """, destination / "validation_predictions.parquet")
        _copy(con, common + f"""
            SELECT h.ncodpers, h.fecha_dato,
                   COALESCE(string_agg(product_values.product, ' ' ORDER BY product_values.product), '') AS actual_added_products
            FROM holdout AS h
            CROSS JOIN LATERAL (VALUES {current_product_values}) AS product_values(product, value, previous_value)
            WHERE product_values.value = 1 AND COALESCE(product_values.previous_value, 0) = 0
            GROUP BY h.ncodpers, h.fecha_dato
        """, actual_path)
        metrics = _map_at_k(con, destination / "validation_predictions.parquet", actual_path, validation_rows, top_k)
        actual_path.unlink(missing_ok=True)
        metrics.update({"validation_date": validation_date, "top_k": top_k, "validation_customers": validation_rows})
        metrics_path = destination / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return {"output_dir": str(destination), "products": products, "metrics": metrics, "metrics_path": str(metrics_path)}
    finally:
        con.close()


def run_popularity_competition(
    history_path: str | Path,
    test_path: str | Path,
    sample_submission_path: str | Path,
    output_path: str | Path,
    *,
    history_date: str,
    top_k: int = 7,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
) -> Path:
    """Create a competition-format Top-K CSV from historical product popularity."""
    history, test, template_path, destination = map(Path, (history_path, test_path, sample_submission_path, output_path))
    if not history.is_file() or not test.is_file() or not template_path.is_file():
        raise FileNotFoundError("History, test and sample submission files are all required.")
    con = duckdb.connect(database=":memory:")
    try:
        _configure(con, memory_limit, temp_directory, None)
        products = [column for column in _columns(con, history) if column.endswith("_ult1")]
        if not products:
            raise ValueError("History checkpoint has no product columns ending in '_ult1'.")
        values = ", ".join(f"('{product}', TRY_CAST(\"{product}\" AS TINYINT), TRY_CAST(\"previous_{product}\" AS TINYINT))" for product in products)
        ownership = "CASE popularity.product " + " ".join(f"WHEN '{product}' THEN COALESCE(latest.\"{product}\", 0)" for product in products) + " ELSE 0 END"
        history_sql, test_sql = _path(history), _path(test)
        query = f"""
            WITH history_rows AS (
                SELECT * FROM read_parquet('{history_sql}') WHERE CAST(fecha_dato AS DATE) <= CAST(? AS DATE)
            ), ordered AS (
                SELECT *, LAG(fecha_dato) OVER customer_time AS previous_date,
                       {', '.join(f'LAG("{product}") OVER customer_time AS "previous_{product}"' for product in products)}
                FROM history_rows WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato)
            ), popularity AS (
                SELECT product, SUM(CASE WHEN value = 1 AND COALESCE(previous_value, 0) = 0 THEN 1 ELSE 0 END) AS purchases
                FROM ordered CROSS JOIN LATERAL (VALUES {values}) AS v(product, value, previous_value)
                WHERE previous_date IS NOT NULL GROUP BY product
            ), latest AS (
                SELECT * EXCLUDE (row_number) FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY ncodpers ORDER BY fecha_dato DESC) AS row_number FROM history_rows
                ) WHERE row_number = 1
            ), candidates AS (
                SELECT test.ncodpers, popularity.product, popularity.purchases,
                       ROW_NUMBER() OVER (PARTITION BY test.ncodpers ORDER BY popularity.purchases DESC, popularity.product) AS rank
                FROM read_parquet('{test_sql}') AS test CROSS JOIN popularity
                LEFT JOIN latest USING (ncodpers)
                WHERE {ownership} = 0
            )
            SELECT ncodpers, COALESCE(string_agg(product, ' ' ORDER BY rank), '') AS added_products
            FROM candidates WHERE rank <= {int(top_k)} GROUP BY ncodpers
        """
        recommendations = con.execute(query, [history_date]).fetchdf()
    finally:
        con.close()
    template = pd.read_csv(template_path)
    if list(template.columns) != ["ncodpers", "added_products"] or template["ncodpers"].duplicated().any():
        raise ValueError("sample_submission must contain unique ncodpers and added_products columns.")
    output = template[["ncodpers"]].merge(recommendations, on="ncodpers", how="left", validate="one_to_one")
    if output["added_products"].isna().any():
        raise ValueError("sample_submission contains customers absent from test data.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return destination


def _map_at_k(con: duckdb.DuckDBPyConnection, predictions: Path, actual: Path, customers: int, top_k: int) -> dict[str, float]:
    row = con.execute("""
        WITH joined AS (
            SELECT p.ncodpers, p.added_products, COALESCE(a.actual_added_products, '') AS actual_added_products
            FROM read_parquet(?) p LEFT JOIN read_parquet(?) a USING (ncodpers, fecha_dato)
        ), exploded AS (
            SELECT ncodpers, actual_added_products, prediction, rank,
                   CASE WHEN list_contains(string_split(actual_added_products, ' '), prediction) THEN 1 ELSE 0 END AS is_hit
            FROM joined, UNNEST(string_split(added_products, ' ')) WITH ORDINALITY AS t(prediction, rank)
        ), ranked AS (
            SELECT *, SUM(is_hit) OVER (PARTITION BY ncodpers ORDER BY rank) AS hits FROM exploded
        ), per_customer AS (
            SELECT ncodpers, CASE WHEN max(actual_added_products) = '' THEN 0.0 ELSE
                SUM(CASE WHEN is_hit = 1 THEN hits::DOUBLE / rank ELSE 0 END)
                / LEAST(list_count(string_split(max(actual_added_products), ' ')), ?) END AS average_precision
            FROM ranked GROUP BY ncodpers
        ) SELECT COALESCE(AVG(average_precision), 0.0) FROM per_customer
    """, [str(predictions), str(actual), top_k]).fetchone()
    return {f"map_at_{top_k}": float(row[0]), "scored_customers": customers}


def _copy(con: duckdb.DuckDBPyConnection, query: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists(): temporary.unlink()
    con.execute(f"COPY ({query}) TO '{_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    temporary.replace(destination)


def _configure(con: duckdb.DuckDBPyConnection, memory_limit: str | None, temp_directory: str | Path | None, threads: int | None) -> None:
    if memory_limit: con.execute(f"SET memory_limit = '{memory_limit}'")
    if threads: con.execute(f"SET threads = {threads}")
    con.execute("SET preserve_insertion_order = false")
    if temp_directory:
        path = Path(temp_directory); path.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{_path(path)}'")


def _columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    return [row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()]


def _require(columns: list[str], required: set[str]) -> None:
    missing = required.difference(columns)
    if missing: raise ValueError(f"Train checkpoint is missing required columns: {sorted(missing)}")


def _path(path: Path) -> str: return str(path).replace("'", "''")
