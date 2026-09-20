"""Hydra entry point for the reproducible LightGBM acquisition baseline."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.pipeline.lightgbm_v1 import run_lightgbm_v1

load_dotenv(PROJECT_ROOT / ".env")


@hydra.main(version_base=None, config_path="../configs/baselines", config_name="lightgbm_v1")
def main(config: DictConfig) -> None:
    """Resolve the versioned config, then execute the complete pipeline."""
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("LightGBM config must resolve to a mapping.")
    config_path = PROJECT_ROOT / "configs" / "baselines" / "lightgbm_v1.yaml"
    result = run_lightgbm_v1(resolved, resolved_config_path=config_path)
    print(OmegaConf.to_yaml(OmegaConf.create(result), resolve=True))


if __name__ == "__main__":
    main()
