"""Independent acquisition-only pipeline orchestration."""
from __future__ import annotations
from src.pipeline.acquisition_only import (
    load_config as _load_config, preview, prepare_dataset, train_run as _train_run,
    validate_run, finalize_run, publish_run,
)

DEFAULT_CONFIG = "configs/baselines/lightgbm_acquisition_only_independent.yaml"

def load_config(config_path=DEFAULT_CONFIG, **overrides):
    config = _load_config(config_path, **overrides)
    if config["pipeline"]["approach"] != "independent":
        raise ValueError("This pipeline requires approach=independent")
    return config

def train_run(config, dataset):
    if config["pipeline"]["approach"] != "independent":
        raise ValueError("This pipeline requires approach=independent")
    return _train_run(config, dataset)

def run_lightgbm_acquisition_only_independent(config, *, publish=False):
    dataset = prepare_dataset(config)
    run = train_run(config, dataset)
    metrics = validate_run(config, dataset, run)
    bundle = finalize_run(run)
    result = {"dataset": dataset, "run": run, "metrics": metrics, "bundle": bundle}
    if publish:
        result["published"] = publish_run(bundle, config)
    return result

def run_lightgbm_acquisition_only_independent_from_config(config_path=DEFAULT_CONFIG, *, publish=False, **overrides):
    return run_lightgbm_acquisition_only_independent(load_config(config_path, **overrides), publish=publish)
