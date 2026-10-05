"""Pull/verify/import GCS runs before optionally hosting MLflow."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import yaml
    from dotenv import load_dotenv
    from src.tracking.gcs_sync import backend_uri, sync
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tracking/gcs_sync.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host", action="store_true", help="Start server only after successful sync")
    parser.add_argument("--allow-stale", action="store_true", help="Explicitly serve previous local snapshot if sync fails")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config["bucket"] = config.get("bucket") or os.getenv("GCS_BUCKET")
    if not config["bucket"]:
        parser.error("Set GCS_BUCKET or bucket in sync config")
    host = Path(config["host_root"]).expanduser().resolve()
    try:
        print(sync(config, dry_run=args.dry_run))
    except Exception as error:
        if not args.host or not args.allow_stale or not (host / "last_sync.json").is_file():
            raise
        print(f"STALE SNAPSHOT: sync failed: {error}; last successful sync: {(host / 'last_sync.json').read_text()}")
    if args.host and not args.dry_run:
        subprocess.run([sys.executable, "-m", "mlflow", "server", "--backend-store-uri", backend_uri(host),
                        "--default-artifact-root", (host / "artifacts").as_uri(), "--no-serve-artifacts",
                        "--host", "127.0.0.1", "--port", str(config.get("port", 5000))], check=True)


if __name__ == "__main__":
    main()
