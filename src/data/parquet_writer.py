"""Streaming Parquet checkpoint writer."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet_checkpoint(
    chunks: Iterable[pd.DataFrame], output_path: str | Path, collect_every: int = 10
) -> int:
    """Write chunks incrementally, releasing each DataFrame/Arrow table after use."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for chunk_number, chunk in enumerate(chunks, start=1):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="snappy")
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
    return rows
