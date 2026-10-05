"""Publish an edited test JSON as a new GCS annotation revision."""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from dotenv import load_dotenv
    from src.tracking.gcs_runs import publish_test_results, validate_test_results
    from src.utils.run_files import read_json
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(); parser.add_argument("file")
    parser.add_argument("--bucket"); parser.add_argument("--prefix", default="model-runs")
    parser.add_argument("--project-id", default="santander-product-recommendation")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(); validate_test_results(read_json(args.file))
    if args.check_only:
        print("Test annotation schema valid"); return
    bucket = args.bucket or os.getenv("GCS_BUCKET")
    if not bucket:
        parser.error("Set GCS_BUCKET or --bucket")
    print(publish_test_results(args.file, bucket, args.prefix, args.project_id))


if __name__ == "__main__":
    main()
