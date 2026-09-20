"""LightGBM baseline: one acquisition classifier per Santander product."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import duckdb
import pandas as pd

from src.inference.predict import InputSchema, ModelArtifact, write_artifact_metadata


def train_product_classifiers(
    model_panel_path: str | Path,
    artifact_root: str | Path,
    *,
    feature_names: Sequence[str] | None = None,
    model_version: str = "lightgbm-binary-v1",
    max_rows_per_product: int | None = None,
    random_state: int = 42,
    n_estimators: int = 300,
    n_jobs: int = 1,
) -> dict[str, str]:
    """Fit 24 independent classifiers using only customers unowned at t-1.

    The panel must come from :func:`build_model_panel`; its current product
    states are absent, while ``acq_*`` is a label.  Models are trained one at a
    time, limiting peak memory to one product sample.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # clear installation failure instead of a cryptic import later
        raise ImportError("LightGBM is required. Install dependencies from requirements.txt.") from exc
    source, root = Path(model_panel_path), Path(artifact_root)
    if not source.is_file():
        raise FileNotFoundError(f"Model panel not found: {source}")
    root.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        columns = [r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]).fetchall()]
        products = [c.removeprefix("acq_") for c in columns if c.startswith("acq_")]
        if len(products) != 24:
            raise ValueError(f"Expected 24 acquisition labels, found {len(products)}.")
        default_features = [c for c in columns if c not in {"ncodpers", "fecha_dato", "previous_observation", *["acq_" + p for p in products]}]
        selected = list(feature_names or default_features)
        missing = set(selected).difference(columns)
        if missing:
            raise ValueError(f"Requested features missing from model panel: {sorted(missing)}")
        result: dict[str, str] = {}
        for product in products:
            # Eligibility is product-specific: only 0 -> {0,1} transitions.
            query = f'SELECT {", ".join(_quote(c) for c in [*selected, "acq_" + product])} FROM read_parquet(?) WHERE previous_observation = 1 AND COALESCE({_quote("prev_" + product)}, 0) = 0'
            if max_rows_per_product:
                query += f" LIMIT {int(max_rows_per_product)}"
            frame = con.execute(query, [str(source)]).fetchdf()
            # The baseline deliberately uses numeric prepared features.  A
            # categorical pipeline may pass its encoded feature_names instead;
            # retaining raw object columns here would make the model artifact
            # non-reproducible at inference time.
            if feature_names is None and not result:
                selected = [c for c in selected if pd.api.types.is_numeric_dtype(frame[c])]
                if not selected:
                    raise ValueError("No numeric model features found; provide encoded feature_names.")
                frame = frame[[*selected, "acq_" + product]]
            non_numeric = [c for c in selected if not pd.api.types.is_numeric_dtype(frame[c])]
            if non_numeric:
                raise TypeError(f"Features must be numeric/encoded for LightGBM baseline: {non_numeric}")
            if frame.empty or frame["acq_" + product].nunique() < 2:
                raise ValueError(f"Product {product} has insufficient eligible examples from both classes.")
            X, y = frame[selected], frame.pop("acq_" + product)
            classifier = LGBMClassifier(objective="binary", n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
            classifier.fit(X, y)
            directory = root / product
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "model.pkl").open("wb") as handle:
                pickle.dump(classifier, handle)
            schema = InputSchema(feature_names=selected, dtypes={c: str(X[c].dtype) for c in selected}, version="1")
            write_artifact_metadata(directory, ModelArtifact(product=product, model_version=model_version, schema=schema))
            result[product] = str(directory)
        return result
    finally:
        con.close()


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
