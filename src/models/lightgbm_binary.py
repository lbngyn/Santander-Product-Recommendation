"""LightGBM baseline: one acquisition classifier per Santander product."""
from __future__ import annotations

import pickle
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import pandas as pd

from src.inference.predict import InputSchema, ModelArtifact, write_artifact_metadata


MODEL_METADATA_COLUMNS = {"ncodpers", "fecha_dato", "previous_observation"}


def train_product_classifiers(
    model_panel_path: str | Path,
    artifact_root: str | Path,
    *,
    feature_names: Sequence[str] | None = None,
    model_version: str = "lightgbm-binary-v1",
    max_rows_per_product: int | None = None,
    max_negative_rows_per_product: int | None = None,
    random_state: int = 42,
    n_estimators: int = 300,
    n_jobs: int = 1,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
    lightgbm_params: Mapping[str, Any] | None = None,
    training_log_period: int = 5,
) -> dict[str, str]:
    """Fit 24 independent classifiers using only customers unowned at t-1.

    The panel must come from :func:`build_model_panel`; its current product
    states are absent, while ``acq_*`` is a label.  Models are trained one at a
    time, limiting peak memory to one product sample.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:  # clear installation failure instead of a cryptic import later
        raise ImportError("LightGBM is required. Install dependencies from requirements.txt.") from exc
    source, root = Path(model_panel_path), Path(artifact_root)
    if not source.is_file():
        raise FileNotFoundError(f"Model panel not found: {source}")
    root.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        if memory_limit:
            con.execute(f"SET memory_limit = '{memory_limit}'")
        if temp_directory:
            tmp = Path(temp_directory)
            tmp.mkdir(parents=True, exist_ok=True)
            escaped_tmp = str(tmp).replace("'", "''")
            con.execute(f"SET temp_directory = '{escaped_tmp}'")
        columns = [r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]).fetchall()]
        products = [c.removeprefix("acq_") for c in columns if c.startswith("acq_")]
        if len(products) != 24:
            raise ValueError(f"Expected 24 acquisition labels, found {len(products)}.")
        default_features = [c for c in columns if c not in { *MODEL_METADATA_COLUMNS, *["acq_" + p for p in products]}]
        selected = list(feature_names or default_features)
        forbidden = set(selected).intersection(MODEL_METADATA_COLUMNS)
        if forbidden:
            raise ValueError(f"Metadata columns cannot be model features: {sorted(forbidden)}")
        missing = set(selected).difference(columns)
        if missing:
            raise ValueError(f"Requested features missing from model panel: {sorted(missing)}")
        if training_log_period < 0:
            raise ValueError("training_log_period must be non-negative.")
        result: dict[str, str] = {}
        training_started = time.perf_counter()
        for model_number, product in enumerate(products, start=1):
            # Eligibility is product-specific and only admits an observed
            # adjacent-month 0 -> {0,1} transition.  A gap cannot safely be
            # interpreted as a monthly non-acquisition.
            projection = ", ".join(_quote(c) for c in [*selected, "acq_" + product])
            eligibility = f'record_gap_months = 1 AND COALESCE({_quote("prev_" + product)}, 0) = 0'
            query = f'SELECT {projection} FROM read_parquet(?) WHERE {eligibility}'
            if max_rows_per_product:
                query += f" LIMIT {int(max_rows_per_product)}"
            elif max_negative_rows_per_product:
                label = _quote("acq_" + product)
                # Keep every rare acquisition and cap only negatives. A plain
                # LIMIT can discard all positives for low-incidence products.
                query = (
                    f'SELECT {projection} FROM read_parquet(?) WHERE {eligibility} AND {label} = 1 '
                    f'UNION ALL '
                    f'SELECT {projection} FROM (SELECT {projection} FROM read_parquet(?) '
                    f'WHERE {eligibility} AND {label} = 0 LIMIT {int(max_negative_rows_per_product)})'
                )
            parameters = [str(source), str(source)] if max_negative_rows_per_product and not max_rows_per_product else [str(source)]
            frame = con.execute(query, parameters).fetchdf()
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
            positives = int(y.sum())
            print(
                f"[LightGBM {model_number}/{len(products)}] {product} | "
                f"eligible_rows={len(y):,}, acquisitions={positives:,} ({positives / len(y):.4%}), "
                f"features={len(selected)}",
                flush=True,
            )
            classifier = lgb.LGBMClassifier(
                objective="binary", n_estimators=n_estimators, random_state=random_state,
                n_jobs=n_jobs, **dict(lightgbm_params or {}),
            )
            callbacks = [lgb.log_evaluation(period=training_log_period)] if training_log_period else []
            model_started = time.perf_counter()
            classifier.fit(
                X, y,
                eval_set=[(X, y)], eval_names=["training"],
                eval_metric=["binary_logloss", "auc"], callbacks=callbacks,
            )
            duration_seconds = time.perf_counter() - model_started
            directory = root / product
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "model.pkl").open("wb") as handle:
                pickle.dump(classifier, handle)
            schema = InputSchema(feature_names=selected, dtypes={c: str(X[c].dtype) for c in selected}, version="1")
            write_artifact_metadata(directory, ModelArtifact(product=product, model_version=model_version, schema=schema))
            metrics = {name: float(values[-1]) for name, values in classifier.evals_result_["training"].items()}
            metrics.update({"eligible_rows": len(y), "acquisitions": positives, "duration_seconds": duration_seconds})
            (directory / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            elapsed_seconds = time.perf_counter() - training_started
            average_seconds = elapsed_seconds / model_number
            eta_seconds = average_seconds * (len(products) - model_number)
            summary = ", ".join(f"{name}={value:.6f}" for name, value in metrics.items() if isinstance(value, float) and name != "duration_seconds")
            print(
                f"[LightGBM {model_number}/{len(products)} complete] {product} | {summary} | "
                f"model_time={duration_seconds:.1f}s, estimated_remaining={eta_seconds / 60:.1f}m",
                flush=True,
            )
            result[product] = str(directory)
        return result
    finally:
        con.close()


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
