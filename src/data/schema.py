"""Canonical Santander schemas used by the ingestion checkpoint writer."""
from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa

from src.data.csv_reader import DATE_COLUMNS, FLOAT_COLUMNS, INTEGER_COLUMNS, PRODUCT_COLUMNS

CUSTOMER_COLUMN = "ncodpers"


def build_arrow_schema(columns: Sequence[str]) -> pa.Schema:
    """Build the semantic Arrow schema for the CSV columns in their source order.

    The train and test files do not contain the same set of columns, so callers
    provide the header read from the specific input file.  No columns are added
    or removed from that header.
    """
    fields: list[pa.Field] = []
    for column in columns:
        if column == CUSTOMER_COLUMN:
            arrow_type = pa.int64()
        elif column in DATE_COLUMNS:
            arrow_type = pa.timestamp("ns")
        elif column in PRODUCT_COLUMNS:
            arrow_type = pa.int8()
        elif column in FLOAT_COLUMNS:
            arrow_type = pa.float32()
        elif column in INTEGER_COLUMNS:
            arrow_type = pa.int64()
        else:
            arrow_type = pa.string()
        fields.append(pa.field(column, arrow_type, nullable=True))
    return pa.schema(fields)
