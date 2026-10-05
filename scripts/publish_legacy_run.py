"""Explicit migration; never changes original model output."""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from dotenv import load_dotenv
    from src.tracking.legacy_runs import export_legacy
    from src.tracking.gcs_runs import publish_bundle
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir"); parser.add_argument("destination")
    parser.add_argument("--kind", required=True, choices=["independent", "joint", "popularity"])
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--project-id", default="santander-product-recommendation")
    parser.add_argument("--source-status", default="unknown", choices=["unknown", "FINISHED", "FAILED", "KILLED"])
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--bucket", default=None); parser.add_argument("--prefix", default="model-runs")
    args = parser.parse_args()
    bundle = export_legacy(args.run_dir, args.destination, project_id=args.project_id, experiment_name=args.experiment,
                           kind=args.kind, source_status=args.source_status)
    print({"bundle": str(bundle)})
    if args.publish:
        bucket = args.bucket or os.getenv("GCS_BUCKET")
        if not bucket:
            parser.error("Set GCS_BUCKET or --bucket")
        print(publish_bundle(bundle, bucket, args.prefix))


if __name__ == "__main__":
    main()
