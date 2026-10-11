from __future__ import annotations

from collections.abc import Sequence
from src.features.customer_history import quote
from .tiering_utils import tier_case_sql

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

def recency_tier_sql(column, boundaries):
    return tier_case_sql(column, boundaries, lower_is_better=True)
