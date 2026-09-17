"""Streaming Parquet checkpoint writer."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet_checkpoint(
    chunks: Iterable[pd.DataFrame],
    output_path: str | Path,
    schema: pa.Schema,
    collect_every: int = 10,
) -> int:
    """Write chunks incrementally using one canonical Arrow schema."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for chunk_number, chunk in enumerate(chunks, start=1):
            table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False, safe=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, schema, compression="snappy")
            writer.write_table(table)
            rows += len(chunk)
            del table
            del chunk
            if chunk_number % collect_every == 0:
                gc.collect()
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Cannot create a checkpoint from an empty CSV.")
    temporary.replace(destination)
    actual_schema = pq.read_schema(destination).remove_metadata()
    expected_schema = schema.remove_metadata()
    if not actual_schema.equals(expected_schema):
        raise TypeError(f"Checkpoint schema does not match the canonical schema: {destination}")
    return rows
