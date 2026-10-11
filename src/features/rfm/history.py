from __future__ import annotations

from collections.abc import Sequence
from src.features.customer_history import quote
from .tiering_utils import tier_case_sql

def rfm_acquisition_count_sql(
    product_columns: Sequence[str],
    *,
    previous_prefix: str = "previous_",
    gap_column: str = "record_gap_months",
) -> str:
    """Return the current row's count of valid product-acquisition events.

    Compare with the nearest observed predecessor, including calendar gaps.
    The first customer record has no predecessor and contributes no event.
    """
    events = [
        f"CASE WHEN {quote(gap_column)} IS NOT NULL "
        f"AND COALESCE({quote(previous_prefix + product)}, 0) = 0 "
        f"AND COALESCE({quote(product)}, 0) = 1 THEN 1 ELSE 0 END"
        for product in product_columns
    ]
    return " + ".join(events) if events else "0"

def rfm_history_coverage_sql() -> str:
    """Return the number of observed customer records before this snapshot."""
    return "CAST(COUNT(*) OVER prior_rows AS INTEGER) AS rfm_observed_history_records"
