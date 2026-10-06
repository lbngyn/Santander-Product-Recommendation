"""Independent acquisition-only models with approved engineered features."""
from src.pipeline.acquisition_engineered import (
    load_config as _load_config, preview, prepare_dataset, train_run as _train_run,
    validate_run, prepare_test, create_submission, finalize_run, publish_run, run_pipeline,
)

DEFAULT_CONFIG = 'configs/baselines/lightgbm_acquisition_only_engineered_independent.yaml'


def load_config(config_path=DEFAULT_CONFIG, **overrides):
    config = _load_config(config_path, **overrides)
    if config['pipeline']['approach'] != 'independent':
        raise ValueError('This pipeline requires independent')
    return config


def train_run(config, dataset):
    if config['pipeline']['approach'] != 'independent':
        raise ValueError('This pipeline requires independent')
    return _train_run(config, dataset)


def run_lightgbm_acquisition_only_engineered_independent(config, *, publish=False):
    if config['pipeline']['approach'] != 'independent':
        raise ValueError('This pipeline requires independent')
    return run_pipeline(config, publish=publish)


def run_lightgbm_acquisition_only_engineered_independent_from_config(config_path=DEFAULT_CONFIG, *, publish=False, **overrides):
    return run_lightgbm_acquisition_only_engineered_independent(load_config(config_path, **overrides), publish=publish)
