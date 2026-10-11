"""Filled income proxy; monetary tiers use log1p income."""
from pathlib import Path
import duckdb
from .tiering_utils import _connect, _columns, _path, tier_case_sql

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

def monetary_value_sql(column="renta_filled"):
    from src.features.customer_history import quote
    return f"LN(1 + {quote(column)})"

def monetary_tier_sql(column, boundaries):
    """Column must already be on the log1p scale."""
    return tier_case_sql(column, boundaries, lower_is_better=False)
