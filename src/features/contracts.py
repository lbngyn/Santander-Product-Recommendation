"""Feature ownership contracts shared by checkpoints and model panels."""
from __future__ import annotations

from collections.abc import Iterable


# These fields are useful for descriptive analysis but must never become model
# inputs.  ``acq_*`` at the checkpoint layer compares two observed product
# states; it is therefore unavailable for the competition test and is not a
# valid point-in-time predictor.
EDA_ONLY_FEATURE_PREFIXES: tuple[str, ...] = ("acq_",)


def is_eda_only_feature(name: str) -> bool:
    """Return whether a canonical field is prohibited as a model input."""
    return name.startswith(EDA_ONLY_FEATURE_PREFIXES)


def exclude_eda_only_features(columns: Iterable[str]) -> list[str]:
    """Return columns that are permitted to cross the model-panel boundary."""
    return [name for name in columns if not is_eda_only_feature(name)]
