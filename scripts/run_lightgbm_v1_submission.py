"""Create a Santander competition submission from one trained LightGBM run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from src.pipeline.lightgbm_v1 import run_lightgbm_v1_competition_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_index", help="Path to the training run's model_artifacts.json")
    parser.add_argument("--config", help="Optional override; defaults to config_resolved.yaml beside the model index.")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    output = run_lightgbm_v1_competition_from_config(args.model_index, args.config)
    print(output)


if __name__ == "__main__":
    main()
