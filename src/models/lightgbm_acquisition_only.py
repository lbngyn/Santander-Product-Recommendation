"""Native LightGBM for both strategies; batch-written candidate caches."""
from __future__ import annotations

import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from src.features.acquisition_only_contract import matrix, model_frame, paired_batches
from src.utils.modeling_runtime import Measurement
from src.utils.run_files import read_json, write_json
from src.inference.predict import InputSchema, ModelArtifact, load_artifact, predict, write_artifact_metadata
from src.tracking.run_bundle import safe_file
from src.products import PRODUCT_COLUMNS


class BaselineBooster:
    """Portable scorer: category codes always come from the saved contract."""
    def __init__(self, booster, contract, joint=False):
        self.booster, self.contract, self.joint = booster, contract, joint

    def predict_proba(self, frame):
        scores = np.asarray(self.booster.predict(matrix(model_frame(frame, self.contract, self.joint))))
        return np.column_stack((1 - scores, scores))


def candidate_batch_scorer(run):
    """Adapt saved independent/joint models to the shared submission scorer."""
    run = Path(run)
    result = read_json(run / "run_result.json")
    contract = read_json(run / "feature_contract.json")
    index = read_json(run / "model_index.json")
    joint = result["approach"] == "joint"
    expected = {"joint_model"} if joint else set(PRODUCT_COLUMNS)
    if set(index) != expected:
        raise ValueError("Incomplete model index")
    loaded = {}
    for key, relative in index.items():
        artifact, model = load_artifact(safe_file(run, relative))
        expected_product = "__joint_product_candidate__" if joint else key
        if artifact.product != expected_product or model.contract != contract or model.joint != joint:
            raise ValueError("Model identity or feature contract differs from run")
        loaded[key] = (artifact, model)

    def score_batch(inputs):
        candidates = []
        for product_id, product in enumerate(PRODUCT_COLUMNS):
            previous = inputs["prev_" + product]
            if not previous.isin([0, 1]).all():
                raise ValueError("Previous ownership must be binary and non-null")
            eligible = previous.eq(0)
            if not eligible.any():
                continue
            frame = inputs.loc[eligible].copy()
            if joint:
                frame["product_id"] = product_id
            artifact, model = loaded["joint_model" if joint else product]
            probabilities = predict(model_frame(frame, contract, joint), artifact, model).to_numpy()
            if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError("Invalid candidate probabilities")
            candidates.append(pd.DataFrame({"ncodpers": frame.ncodpers.to_numpy(),
                                            "product": product, "score": probabilities}))
        return (pd.concat(candidates, ignore_index=True) if candidates
                else pd.DataFrame(columns=["ncodpers", "product", "score"]))

    return score_batch


def _write_candidates(dataset, contract, products, joint, destination, batch_size):
    names = [*contract["feature_names"], *(["product_id"] if joint else [])]
    schema = pa.schema([("target", pa.uint8()), *[(f"f{i}", pa.float32()) for i in range(len(names))]])
    rows = positives = 0
    with pacsv.CSVWriter(destination, schema, write_options=pacsv.WriteOptions(include_header=False)) as writer:
        for inputs, targets in paired_batches(dataset, "train", batch_size):
            for product in products:
                mask = inputs["prev_" + product].eq(0)
                if not mask.any():
                    continue
                frame = inputs.loc[mask].copy()
                if joint:
                    frame["product_id"] = PRODUCT_COLUMNS.index(product)
                values = matrix(model_frame(frame, contract, joint))
                labels = targets.loc[mask, "acq_" + product].to_numpy(dtype="uint8")
                columns = [pa.array(labels), *[pa.array(values[:, i], type=pa.float32(), from_pandas=True) for i in range(values.shape[1])]]
                writer.write_table(pa.Table.from_arrays(columns, schema=schema))
                rows += len(labels); positives += int(labels.sum())
    if not 0 < positives < rows:
        raise ValueError(f"Insufficient binary class support: rows={rows}, positives={positives}")
    return rows, positives


def train(dataset, run_dir, config, work):
    import lightgbm as lgb
    dataset, run_dir, work = Path(dataset), Path(run_dir), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    contract = read_json(dataset / "feature_contract.json")
    audit = read_json(dataset / "cohort_audit.json")
    joint = config["pipeline"]["approach"] == "joint"
    # Fail before training any models, rather than invent a fallback for rare products.
    if not joint:
        invalid = [p["product"] for p in audit["train"]["per_product"] if not p["positives"] or not p["negatives"]]
        if invalid:
            raise ValueError(f"Products without two training classes: {invalid}")
    names = [*contract["feature_names"], *(["product_id"] if joint else [])]
    categorical_indices = [i for i, n in enumerate(names) if n in contract["category_values"] or n == "product_id"]
    jobs = [("joint_model", list(PRODUCT_COLUMNS))] if joint else [(p, [p]) for p in PRODUCT_COLUMNS]
    index, all_metrics = {}, {}
    with Measurement() as total:
        for name, products in jobs:
            directory = run_dir / ("joint_model" if joint else f"models/{name}")
            directory.mkdir(parents=True, exist_ok=True)
            text = work / f"{name}.csv"
            with Measurement() as stage:
                rows, positives = _write_candidates(dataset, contract, products, joint, text, int(config["runtime"]["batch_customer_months"]))
                import time
                began = time.perf_counter()
                params = {"two_round": True, "max_bin": int(config["model"]["lightgbm_params"].get("max_bin", 255)), "header": False, "label_column": 0,
                          "num_threads": int(config["runtime"]["threads"])}
                native = lgb.Dataset(str(text), feature_name=[f"f{i}" for i in range(len(names))], categorical_feature=categorical_indices, params=params, free_raw_data=True)
                native.construct()
                construct_seconds = time.perf_counter() - began
                began = time.perf_counter()
                booster = lgb.train({**config["model"]["lightgbm_params"], "objective": "binary", "seed": int(config["model"]["random_state"]),
                                     "num_threads": int(config["runtime"]["threads"])}, native,
                                    num_boost_round=int(config["model"]["n_estimators"]))
                fit_seconds = time.perf_counter() - began
                booster.free_dataset()
                model = BaselineBooster(booster, contract, joint)
                with (directory / "model.pkl").open("wb") as handle:
                    pickle.dump(model, handle)
                category_values = dict(contract["category_values"])
                if joint:
                    category_values["product_id"] = list(range(24))
                schema = InputSchema(names, {n: "category" if n in category_values else "float32" for n in names}, "acquisition-baseline-v1", category_values)
                write_artifact_metadata(directory, ModelArtifact("__joint_product_candidate__" if joint else name, config["model"]["version"], schema))
                del native, model, booster
                gc.collect()
            metrics = {**stage.metrics(), "eligible_rows": rows, "acquisitions": positives,
                       "negative_rows": rows - positives, "construct_seconds": construct_seconds, "fit_seconds": fit_seconds,
                       "model_size_bytes": (directory / "model.pkl").stat().st_size}
            write_json(directory / "training_metrics.json", metrics)
            index[name] = directory.relative_to(run_dir).as_posix(); all_metrics[name] = metrics
            write_json(run_dir / "model_index.json", index)  # retain completed products if a later model fails
            text.unlink()  # only the cache created in this run
    summary = {**total.metrics(), "number_of_models": len(index),
               "candidate_rows": sum(m["eligible_rows"] for m in all_metrics.values()),
               "model_size_bytes": sum(m["model_size_bytes"] for m in all_metrics.values()),
               "construct_seconds": sum(m["construct_seconds"] for m in all_metrics.values()),
               "fit_seconds": sum(m["fit_seconds"] for m in all_metrics.values())}
    write_json(run_dir / "model_index.json", index)
    write_json(run_dir / "training_metrics.json", summary)
    return summary
