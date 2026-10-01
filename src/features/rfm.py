"""Leakage-safe RFM-proxy SQL builders for customer product snapshots.

Santander's data contains monthly product-ownership snapshots rather than
transactions.  These builders therefore use valid 0-to-1 product transitions
as acquisition events and calculate every RFM value from records strictly
before the snapshot being scored.
"""
from __future__ import annotations

from collections.abc import Sequence

from src.features.customer_history import quote


RFM_FEATURE_NAMES: tuple[str, ...] = (
    "rfm_has_previous_record",
    "rfm_recency_months",
    "rfm_never_acquired_before",
    "rfm_frequency",
    "rfm_monetary",
    "rfm_observed_history_records",
)


def rfm_acquisition_count_sql(
    product_columns: Sequence[str],
    *,
    previous_prefix: str = "previous_",
    gap_column: str = "record_gap_months",
) -> str:
    """Return the current row's count of valid product-acquisition events.

    A product is acquired only when the nearest observed predecessor is one
    calendar month earlier and the product state changes from 0 to 1.  Gapped
    observations are deliberately not inferred as product transitions.
    """
    events = [
        f"CASE WHEN {quote(gap_column)} = 1 "
        f"AND COALESCE({quote(previous_prefix + product)}, 0) = 0 "
        f"AND COALESCE({quote(product)}, 0) = 1 THEN 1 ELSE 0 END"
        for product in product_columns
    ]
    return " + ".join(events) if events else "0"


def rfm_recency_months_sql() -> str:
    """Return months since the latest prior valid acquisition, or ``NULL``."""
    return (
        "CAST(date_diff('month', "
        "MAX(CASE WHEN rfm_n_acquisition > 0 THEN CAST(fecha_dato AS DATE) END) "
        "OVER prior_rows, CAST(fecha_dato AS DATE)) AS INTEGER) "
        "AS rfm_recency_months"
    )


def rfm_never_acquired_before_sql() -> str:
    """Return 1 when no valid acquisition exists before the current snapshot."""
    return (
        "CAST(COALESCE(MAX(CASE WHEN rfm_n_acquisition > 0 THEN 1 ELSE 0 END) "
        "OVER prior_rows, 0) = 0 AS TINYINT) AS rfm_never_acquired_before"
    )


def rfm_frequency_sql() -> str:
    """Return the cumulative count of acquisitions before the current row."""
    return "CAST(COALESCE(SUM(rfm_n_acquisition) OVER prior_rows, 0) AS INTEGER) AS rfm_frequency"


def rfm_monetary_sql(product_columns: Sequence[str], *, previous_prefix: str = "previous_") -> str:
    """Return owned-product count at the nearest prior observed snapshot.

    ``NULL`` distinguishes a customer without a prior observed record from a
    customer whose prior portfolio was observed and empty.  Only product
    values that can be parsed as binary states contribute to the calculation.
    """
    known = " + ".join(
        f"CASE WHEN TRY_CAST({quote(previous_prefix + product)} AS INTEGER) IN (0, 1) THEN 1 ELSE 0 END"
        for product in product_columns
    ) or "0"
    owned = " + ".join(
        f"CASE WHEN TRY_CAST({quote(previous_prefix + product)} AS INTEGER) = 1 THEN 1 ELSE 0 END"
        for product in product_columns
    ) or "0"
    return (
        "CAST(CASE WHEN previous_date IS NOT NULL "
        f"AND ({known}) > 0 THEN ({owned}) ELSE NULL END AS INTEGER) AS rfm_monetary"
    )


def rfm_history_coverage_sql() -> str:
    """Return the number of observed customer records before this snapshot."""
    return "CAST(COUNT(*) OVER prior_rows AS INTEGER) AS rfm_observed_history_records"
