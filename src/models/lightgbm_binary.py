"""LightGBM baseline: one acquisition classifier per Santander product."""
from __future__ import annotations

import gc
import pickle
import json
import os
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
    max_total_rows_per_product: int | None = None,
    random_state: int = 42,
    n_estimators: int = 300,
    n_jobs: int = 1,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
    lightgbm_params: Mapping[str, Any] | None = None,
    training_log_period: int = 5,
    require_adjacent_month: bool = True,
    categorical_feature_names: Sequence[str] | None = None,
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
        categorical = list(categorical_feature_names or [])
        invalid_categorical = set(categorical).difference(selected)
        if invalid_categorical:
            raise ValueError(f"Categorical features are absent from model inputs: {sorted(invalid_categorical)}")
        if training_log_period < 0:
            raise ValueError("training_log_period must be non-negative.")
        result: dict[str, str] = {}
        training_started = time.perf_counter()
        if max_total_rows_per_product is not None and max_total_rows_per_product <= 0:
            raise ValueError("max_total_rows_per_product must be positive when set.")
        for model_number, product in enumerate(products, start=1):
            # Eligibility is product-specific and only admits an observed
            # adjacent-month 0 -> {0,1} transition.  A gap cannot safely be
            # interpreted as a monthly non-acquisition.
            # Materialise exactly the types LightGBM needs.  DuckDB otherwise
            # returns most numerics as float64 / int64, doubling DataFrame RAM.
            projection = ", ".join(
                f"CAST({_quote(name)} AS INTEGER) AS {_quote(name)}"
                if name in categorical
                else f"CAST({_quote(name)} AS FLOAT) AS {_quote(name)}"
                for name in selected
            )
            label_name = "acq_" + product
            label_projection = f"CAST({_quote(label_name)} AS TINYINT) AS {_quote(label_name)}"
            projection = f"{projection}, {label_projection}"
            transition_filter = "record_gap_months = 1 AND " if require_adjacent_month else "previous_observation = 1 AND "
            eligibility = f'{transition_filter}COALESCE({_quote("prev_" + product)}, 0) = 0'
            query = f'SELECT {projection} FROM read_parquet(?) WHERE {eligibility}'
            if max_total_rows_per_product is not None or max_negative_rows_per_product:
                label = _quote("acq_" + product)
                positive_count = int(con.execute(
                    f'SELECT COUNT(*) FROM read_parquet(?) WHERE {eligibility} AND {label} = 1',
                    [str(source)],
                ).fetchone()[0])
                positive_limit = positive_count
                negative_limit = int(max_negative_rows_per_product or positive_count)
                if max_total_rows_per_product is not None:
                    total_limit = int(max_total_rows_per_product)
                    positive_limit = min(positive_count, total_limit)
                    negative_limit = min(negative_limit, max(0, total_limit - positive_limit))
                    # Preserve both classes when positives alone exceed the
                    # total budget; otherwise LightGBM cannot fit a binary
                    # classifier even though the row cap is respected.
                    if positive_count >= total_limit and total_limit > 1:
                        positive_limit = total_limit - 1
                        negative_limit = 1
                # Keep every rare acquisition and cap only negatives. A plain
                # LIMIT can discard all positives for low-incidence products.
                query = (
                    f'SELECT {projection} FROM (SELECT {projection} FROM read_parquet(?) '
                    f'WHERE {eligibility} AND {label} = 1 LIMIT {positive_limit}) '
                    f'UNION ALL '
                    f'SELECT {projection} FROM (SELECT {projection} FROM read_parquet(?) '
                    f'WHERE {eligibility} AND {label} = 0 LIMIT {negative_limit})'
                )
            elif max_rows_per_product:
                query += f" LIMIT {int(max_rows_per_product)}"
            uses_two_sources = max_total_rows_per_product is not None or bool(max_negative_rows_per_product)
            parameters = [str(source), str(source)] if uses_two_sources else [str(source)]
            rss_before_load = _rss_bytes()
            frame = con.execute(query, parameters).fetchdf()
            rss_after_load = _rss_bytes()
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
            if frame.empty or frame[label_name].nunique() < 2:
                raise ValueError(f"Product {product} has insufficient eligible examples from both classes.")
            # ``frame`` now contains only selected input columns, so popping
            # the label and reusing it as X avoids a second full DataFrame.
            y = frame.pop(label_name)
            X = frame
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
                categorical_feature=categorical,
            )
            duration_seconds = time.perf_counter() - model_started
            rss_after_fit = _rss_bytes()
            directory = root / product
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "model.pkl").open("wb") as handle:
                pickle.dump(classifier, handle)
            schema = InputSchema(feature_names=selected, dtypes={c: str(X[c].dtype) for c in selected}, version="1")
            write_artifact_metadata(directory, ModelArtifact(product=product, model_version=model_version, schema=schema))
            metrics = {name: float(values[-1]) for name, values in classifier.evals_result_["training"].items()}
            metrics.update({
                "eligible_rows": len(y),
                "acquisitions": positives,
                "duration_seconds": duration_seconds,
                "rss_before_load_bytes": rss_before_load,
                "rss_after_load_bytes": rss_after_load,
                "rss_after_fit_bytes": rss_after_fit,
            })
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
            # The serialized model is all that needs to survive.  Explicitly
            # free C++ Dataset state and Python matrices before the next
            # product, then capture whether the process RSS actually falls.
            classifier.booster_.free_dataset()
            del classifier, X, y, frame, callbacks
            gc.collect()
            metrics["rss_after_cleanup_bytes"] = _rss_bytes()
            (directory / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(
                f"[LightGBM {model_number}/{len(products)} memory] "
                f"load={_format_bytes(rss_after_load)}, fit={_format_bytes(rss_after_fit)}, "
                f"cleanup={_format_bytes(metrics['rss_after_cleanup_bytes'])}",
                flush=True,
            )
        return result
    finally:
        con.close()


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _rss_bytes() -> int | None:
    """Return current process RSS on Linux (including Colab), if available."""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _format_bytes(value: int | None) -> str:
    return "unavailable" if value is None else f"{value / 1024**3:.2f}GiB"
