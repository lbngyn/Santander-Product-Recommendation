"""Importable entry points for the history-feature LightGBM v2 experiment."""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.pipeline.lightgbm_v2 import (  # noqa: E402
    run_lightgbm_v2,
    run_lightgbm_v2_competition_from_config,
    run_lightgbm_v2_from_config,
)

__all__ = ["run_lightgbm_v2", "run_lightgbm_v2_from_config", "run_lightgbm_v2_competition_from_config"]
