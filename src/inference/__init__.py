"""Model-agnostic inference contracts and implementations."""

from src.inference.predict import InputSchema, ModelArtifact, load_artifact, predict

__all__ = ["InputSchema", "ModelArtifact", "load_artifact", "predict"]
