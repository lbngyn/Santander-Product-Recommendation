"""Local raw-file to interim-Parquet stage for reproducible pipelines."""
from __future__ import annotations

from pathlib import Path

from src.data.csv_reader import read_csv_chunks, read_csv_columns
from src.data.parquet_writer import write_parquet_checkpoint
from src.data.schema import build_arrow_schema


def build_interim_checkpoint(
    raw_path: str | Path,
    interim_path: str | Path,
    *,
    chunksize: int,
    force_rebuild: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    """Stream a raw Santander CSV/ZIP into one canonical interim Parquet file."""
    raw, interim = Path(raw_path), Path(interim_path)
    if not raw.is_file():
        raise FileNotFoundError(f"Raw input not found: {raw}")
    if chunksize < 1:
        raise ValueError("chunksize must be positive.")
    if interim.is_file() and not force_rebuild:
        return {"raw": str(raw), "interim": str(interim), "rows": None, "status": "reused"}
    schema = build_arrow_schema(read_csv_columns(raw))
    rows = write_parquet_checkpoint(
        read_csv_chunks(raw, chunksize=chunksize, show_progress=show_progress),
        interim,
        schema=schema,
    )
    return {"raw": str(raw), "interim": str(interim), "rows": rows, "status": "rebuilt"}
