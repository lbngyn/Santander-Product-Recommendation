from __future__ import annotations

from collections.abc import Sequence
from src.features.customer_history import quote
from .tiering_utils import tier_case_sql

def rfm_frequency_sql() -> str:
    """Return the cumulative count of acquisitions before the current row."""
    return "CAST(COALESCE(SUM(rfm_n_acquisition) OVER prior_rows, 0) AS INTEGER) AS rfm_frequency"

def frequency_tier_sql(column, boundaries):
    return tier_case_sql(column, boundaries, lower_is_better=False)
