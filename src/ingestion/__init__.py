"""Public ingestion namespace.

The implementation remains in ``src.data`` during the gradual migration so
existing notebooks importing ``src.data.ingest`` continue to work.
"""

from src.data.ingest import IngestConfig, config_from_environment, run_default_ingests, run_full_ingest
from src.ingestion.checkpoint import build_interim_checkpoint

__all__ = ["IngestConfig", "build_interim_checkpoint", "config_from_environment", "run_default_ingests", "run_full_ingest"]
