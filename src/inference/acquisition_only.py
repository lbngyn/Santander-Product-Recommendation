"""Inference from the same prepared original-information feature contract."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.acquisition_only_contract import KEYS, model_frame
from src.utils.run_files import read_json
from src.inference.predict import load_artifact
from src.products import PRODUCT_COLUMNS
from src.tracking.run_bundle import safe_file


def rank_prepared_batch(inputs, artifact_root, *, max_batch_customers=10000):
    """No labels required. Input dates already use saved day-epoch encoding."""
    if len(inputs) > max_batch_customers:
        raise ValueError("Split inference into customer batches to bound memory")
    root = Path(artifact_root); contract = read_json(root / "feature_contract.json")
    result = read_json(root / "run_result.json")
    if not result.get("inference_ready"):
        raise ValueError("Run is not inference-ready")
    required = {*KEYS, *contract["feature_names"]}
    if missing := required - set(inputs.columns):
        raise ValueError(f"Missing prepared inference fields: {sorted(missing)}")
    if inputs[KEYS].isna().any().any() or inputs.duplicated(KEYS).any():
        raise ValueError("Inference keys must be unique/non-null")
    index = read_json(root / "model_index.json")
    scores = np.full((len(inputs), 24), -np.inf, dtype="float32")
    joint = result["approach"] == "joint"
    if joint:
        _, shared = load_artifact(safe_file(root, index["joint_model"]))
    for product_id, product in enumerate(PRODUCT_COLUMNS):
        state = inputs["prev_" + product]
        if state.isna().any() or not state.isin([0, 1]).all():
            raise ValueError("Previous product states must be binary/non-null")
        eligible = state.eq(0).to_numpy()
        if not eligible.any():
            continue
        frame = inputs.loc[eligible].copy()
        if joint:
            frame["product_id"] = product_id; model = shared
        else:
            _, model = load_artifact(safe_file(root, index[product]))
        values = model.predict_proba(model_frame(frame, contract, joint))[:, 1]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite model scores")
        scores[eligible, product_id] = values
        if not joint:
            del model
    ranking = np.argsort(-scores, axis=1, kind="stable")[:, :7]
    names = np.asarray(PRODUCT_COLUMNS)
    output = inputs.loc[:, KEYS].copy()
    output["added_products"] = [" ".join(names[row][np.isfinite(scores[i, row])]) for i, row in enumerate(ranking)]
    return output
