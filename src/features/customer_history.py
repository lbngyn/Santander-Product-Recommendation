"""Shared, leakage-safe customer history feature definitions.

The functions in this module only generate SQL expressions.  Keeping the
event definition here makes the training panel and the competition-input
builder use exactly the same definition of a valid transition.
"""
from __future__ import annotations

from collections.abc import Sequence


HISTORY_FEATURE_NAMES: tuple[str, ...] = (
    "record_gap_months",
    "customer_history_length",
    "products_owned_count",
    "cumulative_acquisitions",
    "months_since_last_acquisition",
    "never_acquired_before",
    "acquisitions_last_1m",
    "acquisitions_last_3m",
    "acquisitions_last_6m",
    "cumulative_drops",
)


def quote(identifier: str) -> str:
    """Quote a DuckDB identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def product_event_count_sql(
    product_columns: Sequence[str],
    *,
    event: str,
    previous_prefix: str = "previous_",
    gap_column: str = "record_gap_months",
) -> str:
    """Return the per-row count of valid 0→1 or 1→0 product transitions."""
    if event not in {"acquisition", "drop"}:
        raise ValueError("event must be 'acquisition' or 'drop'.")
    before, after = (0, 1) if event == "acquisition" else (1, 0)
    parts = [
        f"CASE WHEN {quote(gap_column)} = 1 "
        f"AND COALESCE({quote(previous_prefix + product)}, 0) = {before} "
        f"AND COALESCE({quote(product)}, 0) = {after} THEN 1 ELSE 0 END"
        for product in product_columns
    ]
    return " + ".join(parts) if parts else "0"


def acquisition_label_sql(product: str, *, previous_prefix: str = "previous_", gap_column: str = "record_gap_months") -> str:
    """Return the label expression for one product's valid acquisition."""
    return (
        f"CAST(CASE WHEN {quote(gap_column)} = 1 "
        f"AND COALESCE({quote(previous_prefix + product)}, 0) = 0 "
        f"AND COALESCE({quote(product)}, 0) = 1 THEN 1 ELSE 0 END AS TINYINT)"
    )
