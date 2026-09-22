"""Hydra entry point for the end-to-end popularity baseline v0."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

from src.pipeline.baseline_v0 import run_baseline_v0, run_baseline_v0_competition_from_config

load_dotenv(PROJECT_ROOT / ".env")


@hydra.main(version_base=None, config_path="../configs/baselines", config_name="v0")
def main(config: DictConfig) -> None:
    """Resolve environment-backed config, then execute all v0 stages."""
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("Baseline config must resolve to a mapping.")
    config_path = Path(__file__).resolve().parents[1] / "configs" / "baselines" / "v0.yaml"
    result = run_baseline_v0(resolved, resolved_config_path=config_path)
    print(OmegaConf.to_yaml(OmegaConf.create(result), resolve=True))


if __name__ == "__main__":
    main()
