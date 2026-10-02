"""Materialised, leakage-safe customer-month feature checkpoints.

The feature store is deliberately model agnostic.  It contains attributes
known at a customer snapshot and never labels, sampling flags, or validation
decisions.  Those model-specific fields are added only by ``model_panel``.
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import duckdb

from src.features.customer_history import HISTORY_FEATURE_NAMES, LEGACY_HISTORY_FEATURE_NAMES, product_event_count_sql, quote
from src.features.history_features import (
    acquisitions_last_1m_sql,
    acquisitions_last_3m_sql,
    acquisitions_last_6m_sql,
    cumulative_drops_sql,
    customer_history_length_window_sql,
    record_gap_months_sql,
)
from src.features.persona import PERSONA_FEATURES, persona_feature_projection_sql, raw_persona_projection_sql
from src.features.rfm import (
    rfm_acquisition_count_sql,
    rfm_frequency_sql,
    rfm_history_coverage_sql,
    rfm_monetary_sql,
    rfm_never_acquired_before_sql,
    rfm_recency_months_sql,
)


KEY_COLUMNS: tuple[str, str] = ("ncodpers", "fecha_dato")


def acquisition_feature_names(product_columns: Sequence[str]) -> tuple[str, ...]:
    """Names of EDA-only observed-record acquisition features.

    Their value compares a product state with the nearest earlier observed
    record for the same customer.  Unlike model labels, this intentionally
    does not require consecutive calendar months.
    """
    return tuple("acq_" + product for product in product_columns)


def ensure_customer_month_feature_store(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    force_process: bool = False,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
) -> Path:
    """Create or extend the reusable customer-month feature checkpoint.

    Persona and history groups are materialised independently.  A later code
    change that adds a feature to one group rebuilds only that group, then
    rewrites the shared checkpoint with the additional columns.  Existing
    columns remain untouched when ``force_process`` is false.
    """
    source, destination = Path(source_path), Path(destination_path)
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    persona_path = destination.with_name(destination.stem + "_persona.parquet")
    history_path = destination.with_name(destination.stem + "_history.parquet")
    _materialize_persona_group(source, persona_path, force_process=force_process, memory_limit=memory_limit, temp_directory=temp_directory)
    _materialize_history_group(source, history_path, force_process=force_process, memory_limit=memory_limit, temp_directory=temp_directory)

    source_columns = _columns(source)
    products = [name for name in source_columns if name.endswith("_ult1") and not name.startswith(("prev_", "acq_"))]
    required = {
        *KEY_COLUMNS, *PERSONA_FEATURES, *HISTORY_FEATURE_NAMES,
        *acquisition_feature_names(products),
    }
    if destination.is_file() and not force_process and required.issubset(_columns(destination)) and not _has_legacy_history_columns(destination):
        return destination
    _merge_groups(history_path, persona_path, destination, memory_limit=memory_limit, temp_directory=temp_directory)
    return destination


def _materialize_persona_group(source: Path, destination: Path, *, force_process: bool, memory_limit: str | None, temp_directory: str | Path | None, output_features: Sequence[str] = PERSONA_FEATURES) -> None:
    required = {*KEY_COLUMNS, *output_features}
    if destination.is_file() and not force_process and required.issubset(_columns(destination)):
        return
    con = _connect(memory_limit, temp_directory)
    try:
        columns = _columns(source, con)
        _require_keys(columns, source)
        raw_persona = raw_persona_projection_sql("source", columns)
        persona = persona_feature_projection_sql(features=output_features)
        source_sql = _path(source)
        _copy_query(con, f"""
            WITH source_rows AS (
                SELECT source.ncodpers, source.fecha_dato, {raw_persona}
                FROM read_parquet('{source_sql}') AS source
            ), ordered AS (
                SELECT *, {persona}
                FROM source_rows
                WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato),
                       prior_customer_rows AS (PARTITION BY ncodpers ORDER BY fecha_dato ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
            )
            SELECT ncodpers, fecha_dato, {', '.join(quote(name) for name in output_features)}
            FROM ordered
        """, destination)
    finally:
        con.close()


def _materialize_history_group(source: Path, destination: Path, *, force_process: bool, memory_limit: str | None, temp_directory: str | Path | None, output_features: Sequence[str] | None = None) -> None:
    con = _connect(memory_limit, temp_directory)
    try:
        columns = _columns(source, con)
        _require_keys(columns, source)
        products = [name for name in columns if name.endswith("_ult1") and not name.startswith(("prev_", "acq_"))]
        if not products:
            raise ValueError("Source has no product columns ending in '_ult1'.")
        all_outputs = (*HISTORY_FEATURE_NAMES, *("prev_" + name for name in products), *acquisition_feature_names(products))
        selected = tuple(output_features) if output_features is not None else all_outputs
        if missing := set(selected).difference(all_outputs):
            raise ValueError(f"Unknown history outputs: {sorted(missing)}")
        required = {*KEY_COLUMNS, *selected}
        if destination.is_file() and not force_process and required.issubset(_columns(destination, con)) and not _has_legacy_history_columns(destination):
            return
        product_sql = ", ".join(f"source.{quote(name)}" for name in products)
        previous_raw = ", ".join(f"LAG({quote(name)}) OVER customer_time AS {quote('previous_' + name)}" for name in products)
        previous_states = ", ".join(f"CAST(COALESCE({quote('previous_' + name)}, 0) AS TINYINT) AS {quote('prev_' + name)}" for name in products if 'prev_' + name in selected)
        # EDA acquisition features compare the current state with the nearest
        # prior *record*, including across calendar gaps.  The first observed
        # customer record remains NULL because no comparison is possible.
        acquisitions = ", ".join(
            f"CAST(CASE WHEN previous_date IS NULL THEN NULL "
            f"WHEN COALESCE({quote('previous_' + name)}, 0) = 0 "
            f"AND COALESCE({quote(name)}, 0) = 1 THEN 1 ELSE 0 END AS TINYINT) "
            f"AS {quote('acq_' + name)}"
            for name in products if 'acq_' + name in selected
        )
        builders = {
            "record_gap_months": lambda: record_gap_months_sql(output=True),
            "acquisitions_last_1m": acquisitions_last_1m_sql,
            "acquisitions_last_3m": acquisitions_last_3m_sql,
            "acquisitions_last_6m": acquisitions_last_6m_sql,
            "cumulative_drops": cumulative_drops_sql,
            "rfm_has_previous_record": lambda: "CAST(previous_date IS NOT NULL AS TINYINT) AS rfm_has_previous_record",
            "rfm_recency_months": rfm_recency_months_sql,
            "rfm_never_acquired_before": rfm_never_acquired_before_sql,
            "rfm_frequency": rfm_frequency_sql,
            "rfm_monetary": lambda: rfm_monetary_sql(products),
            "rfm_observed_history_records": rfm_history_coverage_sql,
        }
        projection = ", ".join(filter(None, [
            previous_states, acquisitions,
            *(builders[name]() for name in selected if name in builders),
        ]))
        raw_products = product_sql.replace('source.', '') + ', ' if output_features is None else ''
        source_sql = _path(source)
        _copy_query(con, f"""
            WITH source_rows AS (
                SELECT source.ncodpers, source.fecha_dato, {product_sql}
                FROM read_parquet('{source_sql}') AS source
            ), ordered AS (
                SELECT *, LAG(fecha_dato) OVER customer_time AS previous_date,
                       {customer_history_length_window_sql()}, {previous_raw}
                FROM source_rows
                WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato),
                       prior_customer_rows AS (PARTITION BY ncodpers ORDER BY fecha_dato ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
            ), transitions AS (
                SELECT *, {record_gap_months_sql()} FROM ordered
            ), events AS (
                SELECT *, {product_event_count_sql(products, event='acquisition')} AS acquisition_events,
                       {product_event_count_sql(products, event='drop')} AS drop_events
                FROM transitions
            ), rfm_events AS (
                SELECT *, {rfm_acquisition_count_sql(products)} AS rfm_n_acquisition
                FROM events
            )
            SELECT ncodpers, fecha_dato, {raw_products}{projection}
            FROM rfm_events
            WINDOW prior_rows AS (PARTITION BY ncodpers ORDER BY fecha_dato ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                   recent_1m AS (PARTITION BY ncodpers ORDER BY CAST(fecha_dato AS DATE) RANGE BETWEEN INTERVAL 1 MONTH PRECEDING AND INTERVAL 1 DAY PRECEDING),
                   recent_3m AS (PARTITION BY ncodpers ORDER BY CAST(fecha_dato AS DATE) RANGE BETWEEN INTERVAL 3 MONTH PRECEDING AND INTERVAL 1 DAY PRECEDING),
                   recent_6m AS (PARTITION BY ncodpers ORDER BY CAST(fecha_dato AS DATE) RANGE BETWEEN INTERVAL 6 MONTH PRECEDING AND INTERVAL 1 DAY PRECEDING)
        """, destination)
    finally:
        con.close()


def materialize_customer_month_feature_group(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    outputs: Sequence[str],
    force_process: bool = False,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
) -> Path:
    """Materialize only one declared output contract from the working snapshot.

    The canonical pipeline merges this narrow result into a new full candidate.
    Unselected output builders are never called. Common SQL intermediates
    remain internal; they are not persisted as replacements for other features.
    """
    source, destination = Path(source_path), Path(destination_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not outputs:
        raise ValueError("Feature group must declare at least one output")
    if set(outputs).issubset(_columns(source)) and not force_process:
        return source
    if source.resolve() == destination.resolve():
        raise ValueError("Feature-group destination must differ from source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if set(outputs).issubset(PERSONA_FEATURES):
        _materialize_persona_group(source, destination, output_features=outputs,
                                  force_process=True, memory_limit=memory_limit, temp_directory=temp_directory)
    else:
        _materialize_history_group(source, destination, output_features=outputs,
                                  force_process=True, memory_limit=memory_limit, temp_directory=temp_directory)
    if missing := set(outputs).difference(_columns(destination)):
        raise RuntimeError(f"Feature group did not materialize outputs: {sorted(missing)}")
    return destination


def _merge_groups(history_path: Path, persona_path: Path, destination: Path, *, memory_limit: str | None, temp_directory: str | Path | None) -> None:
    con = _connect(memory_limit, temp_directory)
    try:
        _copy_query(con, f"""
            SELECT history.*, {', '.join('persona.' + quote(name) for name in PERSONA_FEATURES)}
            FROM read_parquet('{_path(history_path)}') AS history
            INNER JOIN read_parquet('{_path(persona_path)}') AS persona USING (ncodpers, fecha_dato)
        """, destination)
    finally:
        con.close()


def _copy_query(con: duckdb.DuckDBPyConnection, query: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    con.execute(f"COPY ({query}) TO '{_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    temporary.replace(destination)


def _connect(memory_limit: str | None, temp_directory: str | Path | None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    if memory_limit:
        con.execute(f"SET memory_limit = '{str(memory_limit).replace(chr(39), chr(39) * 2)}'")
    if temp_directory:
        path = Path(temp_directory); path.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{_path(path)}'")
    return con


def _columns(path: Path, con: duckdb.DuckDBPyConnection | None = None) -> set[str]:
    owns_connection = con is None
    con = con or duckdb.connect(database=":memory:")
    try:
        return {row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()}
    finally:
        if owns_connection:
            con.close()


def _has_legacy_history_columns(path: Path) -> bool:
    return bool(_columns(path).intersection(LEGACY_HISTORY_FEATURE_NAMES))


def _require_keys(columns: Sequence[str] | set[str], source: Path) -> None:
    if missing := set(KEY_COLUMNS).difference(columns):
        raise ValueError(f"Source {source} is missing required columns: {sorted(missing)}")


def _path(path: Path) -> str:
    return str(path).replace("'", "''")
