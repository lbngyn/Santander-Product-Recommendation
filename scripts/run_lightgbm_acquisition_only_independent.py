"""Importable notebook entry points and CLI for the independent pipeline."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.pipeline.lightgbm_acquisition_only_independent import (
    DEFAULT_CONFIG, load_config, preview, prepare_dataset, train_run,
    validate_run, finalize_run, publish_run,
    run_lightgbm_acquisition_only_independent, run_lightgbm_acquisition_only_independent_from_config,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--comparison-id")
    parser.add_argument("--checkpoint-version", type=int)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, comparison_id=args.comparison_id, checkpoint_version=args.checkpoint_version)
    print(preview(config) if args.preview else run_lightgbm_acquisition_only_independent(config, publish=args.publish))

if __name__ == "__main__":
    main()
