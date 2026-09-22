"""One SQL builder per leakage-safe customer-history feature."""
from __future__ import annotations

from collections.abc import Sequence

from src.features.customer_history import quote


def record_gap_months_sql(*, output: bool = False) -> str:
    if output:
        return "CAST(record_gap_months AS INTEGER) AS record_gap_months"
    return "CASE WHEN previous_date IS NULL THEN NULL ELSE date_diff('month', CAST(previous_date AS DATE), CAST(fecha_dato AS DATE)) END AS record_gap_months"


def customer_history_length_sql() -> str:
    return "CAST(customer_history_length AS INTEGER) AS customer_history_length"


def customer_history_length_window_sql() -> str:
    return "COUNT(*) OVER (customer_time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS customer_history_length"


def products_owned_count_sql(products: Sequence[str]) -> str:
    owned = " + ".join(f"COALESCE({quote('previous_' + product)}, 0)" for product in products)
    return f"CAST({owned} AS TINYINT) AS products_owned_count"


def cumulative_acquisitions_sql() -> str:
    return "CAST(COALESCE(SUM(acquisition_events) OVER prior_rows, 0) AS INTEGER) AS cumulative_acquisitions"


def months_since_last_acquisition_sql() -> str:
    return "CAST(date_diff('month', MAX(CASE WHEN acquisition_events > 0 THEN CAST(fecha_dato AS DATE) END) OVER prior_rows, CAST(fecha_dato AS DATE)) AS INTEGER) AS months_since_last_acquisition"


def never_acquired_before_sql() -> str:
    return "CAST(COALESCE(MAX(CASE WHEN acquisition_events > 0 THEN 1 ELSE 0 END) OVER prior_rows, 0) = 0 AS TINYINT) AS never_acquired_before"


def acquisitions_last_1m_sql() -> str:
    return "CAST(COALESCE(SUM(acquisition_events) OVER recent_1m, 0) AS INTEGER) AS acquisitions_last_1m"


def acquisitions_last_3m_sql() -> str:
    return "CAST(COALESCE(SUM(acquisition_events) OVER recent_3m, 0) AS INTEGER) AS acquisitions_last_3m"


def acquisitions_last_6m_sql() -> str:
    return "CAST(COALESCE(SUM(acquisition_events) OVER recent_6m, 0) AS INTEGER) AS acquisitions_last_6m"


def cumulative_drops_sql() -> str:
    return "CAST(COALESCE(SUM(drop_events) OVER prior_rows, 0) AS INTEGER) AS cumulative_drops"
