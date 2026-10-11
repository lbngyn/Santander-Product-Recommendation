from __future__ import annotations

from collections.abc import Sequence
from src.features.customer_history import quote
from .tiering_utils import tier_case_sql

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

def monetary_tier_sql(column, boundaries):
    return tier_case_sql(column, boundaries, lower_is_better=False)
