"""Schema-validated, model-agnostic inference for product classifiers."""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class InputSchema:
    feature_names: list[str]
    dtypes: dict[str, str]
    version: str
    # Vocabulary is semantic data, not a user-managed numeric encoding.  It
    # lets pandas construct the same categorical schema at train and infer.
    category_values: dict[str, list[Any]] = field(default_factory=dict)

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_names if c not in frame.columns]
        extra = [c for c in frame.columns if c not in self.feature_names]
        if missing or extra:
            raise ValueError(f"Input schema mismatch; missing={missing}, unexpected={extra}")
        ordered = frame.loc[:, self.feature_names].copy()
        for column, categories in self.category_values.items():
            if column not in ordered:
                continue
            # Values unseen during training become NaN, which LightGBM handles
            # as its native missing branch.  Known categories retain precisely
            # the training vocabulary and unordered category semantics.
            ordered[column] = pd.Categorical(ordered[column], categories=categories)
        wrong = {c: (str(ordered[c].dtype), self.dtypes[c]) for c in self.feature_names if str(ordered[c].dtype) != self.dtypes[c]}
        if wrong:
            raise TypeError(f"Input datatype mismatch: {wrong}")
        return ordered


@dataclass(frozen=True)
class ModelArtifact:
    product: str
    model_version: str
    schema: InputSchema
    model_file: str = "model.pkl"


def load_artifact_metadata(artifact_dir: str | Path) -> ModelArtifact:
    """Load an artifact contract without materialising its model."""
    directory = Path(artifact_dir)
    metadata = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
    schema = InputSchema(**metadata["input_schema"])
    return ModelArtifact(product=metadata["product"], model_version=metadata["model_version"], schema=schema, model_file=metadata.get("model_file", "model.pkl"))


def load_artifact(artifact_dir: str | Path) -> tuple[ModelArtifact, Any]:
    """Load metadata plus a model exposing sklearn-style ``predict_proba``."""
    directory = Path(artifact_dir)
    artifact = load_artifact_metadata(directory)
    with (directory / artifact.model_file).open("rb") as handle:
        return artifact, pickle.load(handle)


def predict(prepared_input: pd.DataFrame, artifact: str | Path | ModelArtifact, model: Any | None = None) -> pd.Series:
    """Return P(y=1|X) after validating the artifact-declared input schema.

    Supplying ``ModelArtifact`` and ``model`` makes the function usable with a
    non-file model registry; a directory keeps local artifacts self-contained.
    """
    if isinstance(artifact, (str, Path)):
        artifact, model = load_artifact(artifact)
    if model is None:
        raise ValueError("model is required when artifact metadata is supplied directly.")
    features = artifact.schema.validate(prepared_input)
    probabilities = np.asarray(model.predict_proba(features))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Binary model predict_proba must return an (n_rows, 2) probability array.")
    return pd.Series(probabilities[:, 1], index=prepared_input.index, name=artifact.product, dtype="float64")


def write_artifact_metadata(directory: str | Path, artifact: ModelArtifact) -> Path:
    path = Path(directory) / "artifact.json"
    path.write_text(json.dumps({"product": artifact.product, "model_version": artifact.model_version, "model_file": artifact.model_file, "input_schema": asdict(artifact.schema)}, indent=2), encoding="utf-8")
    return path
