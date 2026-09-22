"""LightGBM baseline: one acquisition classifier per Santander product."""
from __future__ import annotations

import gc
import pickle
import json
import os
import threading
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
    product_names: Sequence[str] | None = None,
    max_rows_per_product: int | None = None,
    max_negative_rows_per_product: int | None = None,
    max_total_rows_per_product: int | None = None,
    random_state: int = 42,
    negative_sampling_strategy: str = "head",
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
    states are absent, while ``acq_*`` is a label. Models are trained one at a
    time, limiting peak memory to one product sample. When negative examples
    are capped, ``negative_sampling_strategy='random'`` orders them by a
    deterministic DuckDB hash using ``random_state`` and the product ordinal.
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
        all_products = [c.removeprefix("acq_") for c in columns if c.startswith("acq_")]
        if len(all_products) != 24:
            raise ValueError(f"Expected 24 acquisition labels, found {len(all_products)}.")
        requested_products = set(product_names or all_products)
        unknown_products = requested_products.difference(all_products)
        if unknown_products:
            raise ValueError(f"Requested product(s) are absent from the model panel: {sorted(unknown_products)}")
        products = [product for product in all_products if product in requested_products]
        if not products:
            raise ValueError("At least one product must be selected for training.")
        default_features = [c for c in columns if c not in { *MODEL_METADATA_COLUMNS, *["acq_" + p for p in all_products]}]
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
        if negative_sampling_strategy not in {"head", "random"}:
            raise ValueError("negative_sampling_strategy must be either 'head' or 'random'.")
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
                # Preserve semantic category values in the panel.  Pandas
                # creates compact, unordered category codes only at the
                # LightGBM boundary; no one-hot, hash, or manual label
                # encoding is used.
                f"CAST({_quote(name)} AS VARCHAR) AS {_quote(name)}"
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
                # Keep every rare acquisition and cap only negatives. The
                # random order is deterministic, so reruns with the same seed
                # select the same rows without materialising all negatives.
                negative_order = _negative_order_sql(
                    negative_sampling_strategy,
                    random_state=random_state,
                    product_ordinal=model_number,
                )
                query = (
                    f'SELECT {projection} FROM (SELECT {projection} FROM read_parquet(?) '
                    f'WHERE {eligibility} AND {label} = 1 LIMIT {positive_limit}) '
                    f'UNION ALL '
                    f'SELECT {projection} FROM (SELECT {projection} FROM read_parquet(?) '
                    f'WHERE {eligibility} AND {label} = 0{negative_order} LIMIT {negative_limit})'
                )
            elif max_rows_per_product:
                query += f" LIMIT {int(max_rows_per_product)}"
            uses_two_sources = max_total_rows_per_product is not None or bool(max_negative_rows_per_product)
            parameters = [str(source), str(source)] if uses_two_sources else [str(source)]
            rss_before_load = _rss_bytes()
            frame = con.execute(query, parameters).fetchdf()
            rss_after_load = _rss_bytes()
            for name in categorical:
                # ``MISSING`` is a meaningful persona state.  ``category``
                # gives LightGBM compact internal IDs and native categorical
                # splits while retaining the values' unordered semantics.
                frame[name] = frame[name].fillna("MISSING").astype("category")
            if feature_names is None and not result:
                selected = [
                    c for c in selected
                    if c in categorical or pd.api.types.is_numeric_dtype(frame[c])
                ]
                if not selected:
                    raise ValueError("No numeric or categorical model features found.")
                frame = frame[[*selected, "acq_" + product]]
            non_numeric = [
                c for c in selected
                if c not in categorical and not pd.api.types.is_numeric_dtype(frame[c])
            ]
            if non_numeric:
                raise TypeError(f"Features must be numeric or Pandas categorical for LightGBM: {non_numeric}")
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
            directory = root / product
            directory.mkdir(parents=True, exist_ok=True)
            monitor = _MemoryMonitor(directory / "memory_samples.jsonl")
            model_started = time.perf_counter()
            monitor.start()
            try:
                # Evaluating against X/y during fit is not validation and can
                # construct another full LightGBM Dataset.  Omit it to reduce
                # the peak allocation; real temporal validation happens in
                # the pipeline after training.
                classifier.fit(X, y, categorical_feature=categorical)
            finally:
                monitor.stop()
            duration_seconds = time.perf_counter() - model_started
            rss_after_fit = _rss_bytes()
            with (directory / "model.pkl").open("wb") as handle:
                pickle.dump(classifier, handle)
            schema = InputSchema(
                feature_names=selected,
                dtypes={c: str(X[c].dtype) for c in selected},
                category_values={c: X[c].cat.categories.tolist() for c in categorical},
                version="1",
            )
            write_artifact_metadata(directory, ModelArtifact(product=product, model_version=model_version, schema=schema))
            metrics: dict[str, float | int | None] = {}
            metrics.update({
                "eligible_rows": len(y),
                "acquisitions": positives,
                "negative_sampling_strategy": negative_sampling_strategy,
                "negative_sampling_seed": random_state + model_number if negative_sampling_strategy == "random" else None,
                "duration_seconds": duration_seconds,
                "rss_before_load_bytes": rss_before_load,
                "rss_after_load_bytes": rss_after_load,
                "rss_after_fit_bytes": rss_after_fit,
                "rss_peak_during_fit_bytes": monitor.peak_rss_bytes,
                "mem_available_min_during_fit_bytes": monitor.min_mem_available_bytes,
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
            del classifier, X, y, frame
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


def _negative_order_sql(strategy: str, *, random_state: int, product_ordinal: int) -> str:
    """Return a stable random ordering for one product's eligible negatives."""
    if strategy == "head":
        return ""
    if strategy != "random":
        raise ValueError("negative_sampling_strategy must be either 'head' or 'random'.")
    sampling_seed = random_state + product_ordinal
    return f" ORDER BY hash(ncodpers, fecha_dato, {sampling_seed}), ncodpers, fecha_dato"


def _rss_bytes() -> int | None:
    """Return current process RSS on Linux (including Colab), if available."""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _format_bytes(value: int | None) -> str:
    return "unavailable" if value is None else f"{value / 1024**3:.2f}GiB"


def _mem_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, IndexError, OSError, ValueError):
        pass
    return None


class _MemoryMonitor:
    """Persist RSS/system-memory samples while C++ LightGBM code is running."""

    def __init__(self, path: Path, interval_seconds: float = 0.5) -> None:
        self.path = path
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes: int | None = None
        self.min_mem_available_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 3)

    def _sample(self) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set():
                rss, available = _rss_bytes(), _mem_available_bytes()
                if rss is not None:
                    self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)
                if available is not None:
                    self.min_mem_available_bytes = min(self.min_mem_available_bytes or available, available)
                handle.write(json.dumps({"timestamp": time.time(), "rss_bytes": rss, "mem_available_bytes": available}) + "\n")
                handle.flush()
                self._stop.wait(self.interval_seconds)
