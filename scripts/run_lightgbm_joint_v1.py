"""Notebook- and CLI-friendly entry point for the shared LightGBM experiment."""
from __future__ import annotations

from src.pipeline.lightgbm_joint_v1 import run_lightgbm_joint_v1, run_lightgbm_joint_v1_from_config

__all__ = ["run_lightgbm_joint_v1", "run_lightgbm_joint_v1_from_config"]


if __name__ == "__main__":
    result = run_lightgbm_joint_v1_from_config()
    print(result)
