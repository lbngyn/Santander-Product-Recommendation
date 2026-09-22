"""Importable entry points for the LightGBM v3 persona/history experiment."""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.pipeline.lightgbm_v3 import (  # noqa: E402
    run_lightgbm_v3,
    run_lightgbm_v3_competition_from_config,
    run_lightgbm_v3_from_config,
)

__all__ = [
    "run_lightgbm_v3",
    "run_lightgbm_v3_from_config",
    "run_lightgbm_v3_competition_from_config",
]
