"""Orchestrate the full reusable Google Cloud Storage ingestion flow."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.data.csv_reader import read_csv_chunks
from src.data.gcs_storage import download_object_if_missing, object_name, upload_file
from src.data.parquet_writer import write_parquet_checkpoint


@dataclass(frozen=True)
class IngestConfig:
    bucket: str
    raw_prefix: str = "raw"
    checkpoint_prefix: str = "interim"
    project_id: str | None = None
    raw_dir: Path = Path("data/raw")
    checkpoint_dir: Path = Path("data/interim")
    chunksize: int = 250_000


def download_raw_data(config: IngestConfig, csv_filename: str, force: bool = False) -> Path | None:
    """Ensure the requested raw CSV exists locally, downloading only that object if needed."""
    relative_csv_path = Path(csv_filename)
    remote_object = object_name(config.raw_prefix, relative_csv_path)
    local_path = config.raw_dir / relative_csv_path
    return download_object_if_missing(config.bucket, remote_object, local_path, config.project_id, force)


def create_parquet_checkpoint(config: IngestConfig, csv_filename: str, parquet_filename: str) -> tuple[Path, int]:
    """Read local raw CSV, parse dtypes/encoding, and write a Parquet checkpoint."""
    input_path = config.raw_dir / csv_filename
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Raw CSV not found: {input_path}. Remove --skip-download or place the file in data/raw."
        )
    output_path = config.checkpoint_dir / parquet_filename
    rows = write_parquet_checkpoint(read_csv_chunks(input_path, config.chunksize), output_path)
    return output_path, rows


def upload_checkpoint(config: IngestConfig, checkpoint_path: str | Path) -> str:
    """Upload a checkpoint to the configured GCS prefix."""
    path = Path(checkpoint_path)
    return upload_file(path, config.bucket, object_name(config.checkpoint_prefix, path.name), config.project_id)


def run_full_ingest(config: IngestConfig, csv_filename: str, parquet_filename: str, download_raw: bool = True, upload_parquet: bool = True, force_download: bool = False) -> dict[str, object]:
    """Run GCS download → CSV parse → Parquet checkpoint → optional GCS upload."""
    downloaded_file = download_raw_data(config, csv_filename, force_download) if download_raw else None
    checkpoint_path, rows = create_parquet_checkpoint(config, csv_filename, parquet_filename)
    checkpoint_uri = upload_checkpoint(config, checkpoint_path) if upload_parquet else None
    return {"downloaded_file": str(downloaded_file) if downloaded_file else None, "rows": rows, "checkpoint": str(checkpoint_path), "checkpoint_uri": checkpoint_uri}


def config_from_environment(chunksize: int) -> IngestConfig:
    """Read non-secret GCS config from the environment or local .env file."""
    load_dotenv()
    bucket = os.getenv("GCS_BUCKET")
    if not bucket:
        raise ValueError("Set GCS_BUCKET in .env or the environment.")
    return IngestConfig(bucket=bucket, raw_prefix=os.getenv("GCS_RAW_PREFIX", "raw"), checkpoint_prefix=os.getenv("GCS_CHECKPOINT_PREFIX", "interim"), project_id=os.getenv("GOOGLE_CLOUD_PROJECT"), chunksize=chunksize)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Santander GCS ingestion flow.")
    parser.add_argument("csv_filename", nargs="?", default="train_ver2.csv", help="Filename under data/raw and the raw GCS prefix")
    parser.add_argument("parquet_filename", nargs="?", default="train.parquet", help="Output filename under data/interim")
    parser.add_argument("--chunksize", type=int, default=250_000)
    download_mode = parser.add_mutually_exclusive_group()
    download_mode.add_argument("--skip-download", action="store_true", help="Use existing local raw data without a GCS download")
    download_mode.add_argument("--force-download", action="store_true", help="Download from GCS even when data/raw already has files")
    parser.add_argument("--skip-upload", action="store_true", help="Keep the Parquet checkpoint local only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be positive.")
    try:
        config = config_from_environment(args.chunksize)
        result = run_full_ingest(config, args.csv_filename, args.parquet_filename, not args.skip_download, not args.skip_upload, args.force_download)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Ingest failed: {error}") from error
    print(result)


if __name__ == "__main__":
    main()
