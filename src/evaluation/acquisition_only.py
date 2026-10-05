"""Shared full-May evaluation; batched scoring and exact per-product AP."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.acquisition_only_contract import model_frame, paired_batches
from src.utils.modeling_runtime import Measurement
from src.utils.run_files import read_json, write_json
from src.inference.predict import load_artifact
from src.products import PRODUCT_COLUMNS

METRIC_CONTRACT = {
    "version": "ranking-full-eligible-v1", "top_k": 7,
    "map_denominator": "all_eligible_customer_months; empty_target_AP=0",
    "recall": "micro_product_hits/total_acquisitions",
    "hit_rate": "customers_with_hit/all_eligible_customers",
    "product_pr": "average_precision_on_unowned_candidates; no_positive=null",
    "tie_break": "canonical_product_order",
}


def evaluate(dataset, run_dir, config, work):
    from sklearn.metrics import average_precision_score
    dataset, run_dir, work = Path(dataset), Path(run_dir), Path(work)
    scores_dir = work / "validation_scores"; scores_dir.mkdir(parents=True, exist_ok=True)
    contract = read_json(dataset / "feature_contract.json")
    index = read_json(run_dir / "model_index.json")
    joint = config["pipeline"]["approach"] == "joint"
    batch_size = int(config["runtime"]["batch_customer_months"])
    score_schema = pa.schema([("score", pa.float32()), ("target", pa.uint8()), ("eligible", pa.bool_())])
    paths = {p: scores_dir / f"{p}.parquet" for p in PRODUCT_COLUMNS}
    with Measurement() as inference:
        if joint:
            _, model = load_artifact(run_dir / index["joint_model"])
            writers = {p: pq.ParquetWriter(paths[p], score_schema, compression="zstd") for p in PRODUCT_COLUMNS}
            try:
                for inputs, targets in paired_batches(dataset, "validation", batch_size):
                    for product_id, product in enumerate(PRODUCT_COLUMNS):
                        eligible = inputs["prev_" + product].eq(0).to_numpy()
                        score = np.full(len(inputs), -np.inf, dtype="float32")
                        if eligible.any():
                            frame = inputs.loc[eligible].copy(); frame["product_id"] = product_id
                            score[eligible] = model.predict_proba(model_frame(frame, contract, True))[:, 1]
                        writers[product].write_table(_score_table(score_schema, score, targets["acq_" + product], eligible))
            finally:
                for writer in writers.values():
                    writer.close()
            del model
        else:
            for product in PRODUCT_COLUMNS:
                _, model = load_artifact(run_dir / index[product])
                with pq.ParquetWriter(paths[product], score_schema, compression="zstd") as writer:
                    for inputs, targets in paired_batches(dataset, "validation", batch_size):
                        eligible = inputs["prev_" + product].eq(0).to_numpy()
                        score = np.full(len(inputs), -np.inf, dtype="float32")
                        if eligible.any():
                            score[eligible] = model.predict_proba(model_frame(inputs.loc[eligible], contract))[:, 1]
                        writer.write_table(_score_table(score_schema, score, targets["acq_" + product], eligible))
                del model
    totals = dict(customers=0, positives=0, hits=0, hit_customers=0, positive_customers=0, ap_sum=0.0)
    iterators = [iter(pq.ParquetFile(paths[p]).iter_batches(batch_size=batch_size)) for p in PRODUCT_COLUMNS]
    from itertools import zip_longest
    for batches in zip_longest(*iterators):
        if any(b is None for b in batches) or len({len(b) for b in batches}) != 1:
            raise ValueError("Product score row counts differ")
        frames = [b.to_pandas() for b in batches]
        scores = np.column_stack([f["score"].to_numpy() for f in frames])
        labels = np.column_stack([f["target"].to_numpy() for f in frames])
        eligibility = np.column_stack([f["eligible"].to_numpy() for f in frames])
        if not np.isfinite(scores[eligibility]).all() or ((labels == 1) & ~eligibility).any():
            raise ValueError("Nonfinite candidate scores or ineligible positive labels")
        ranking = np.argsort(-scores, axis=1, kind="stable")[:, :7]
        selected = np.take_along_axis(scores, ranking, axis=1)
        hits = np.take_along_axis(labels, ranking, axis=1).astype(bool) & np.isfinite(selected)
        positive_counts = labels.sum(axis=1)
        precision = np.cumsum(hits, axis=1) / np.arange(1, 8)
        ap = np.divide((precision * hits).sum(axis=1), np.minimum(positive_counts, 7), out=np.zeros(len(labels)), where=positive_counts > 0)
        totals["customers"] += len(labels); totals["positives"] += int(positive_counts.sum())
        totals["hits"] += int(hits.sum()); totals["hit_customers"] += int(hits.any(axis=1).sum())
        totals["positive_customers"] += int((positive_counts > 0).sum()); totals["ap_sum"] += float(ap.sum())
    per_product = []
    for product in PRODUCT_COLUMNS:
        # One product at a time: exact global AP, never an average of batch AP.
        frame = pd.read_parquet(paths[product]); frame = frame.loc[frame["eligible"]]
        positives = int(frame["target"].sum())
        per_product.append({"product": product, "positives": positives, "negatives": len(frame) - positives,
                            "average_precision": float(average_precision_score(frame["target"], frame["score"])) if positives else None,
                            "status": "evaluated" if positives else "no_positive_support"})
    if not totals["customers"]:
        raise ValueError("Empty validation universe")
    metrics = {"map_at_7": totals["ap_sum"] / totals["customers"],
               "product_recall_at_7": totals["hits"] / totals["positives"] if totals["positives"] else 0.0,
               "hit_rate_at_7_all": totals["hit_customers"] / totals["customers"],
               "hit_rate_at_7_positive": totals["hit_customers"] / totals["positive_customers"] if totals["positive_customers"] else 0.0,
               "validation_customers": totals["customers"], "validation_acquisitions": totals["positives"],
               "inference_duration_seconds": inference.seconds, "inference_peak_rss_bytes": inference.peak,
               "inference_timing_scope": "model_load+batch_scoring+score_parquet_write; excludes ranking/AP"}
    write_json(run_dir / "validation_metrics.json", metrics)
    write_json(run_dir / "per_product_metrics.json", per_product)
    write_json(run_dir / "evaluation_contract.json", METRIC_CONTRACT)
    return metrics


def _score_table(schema, scores, targets, eligible):
    return pa.Table.from_arrays([pa.array(scores, type=pa.float32()), pa.array(targets, type=pa.uint8()), pa.array(eligible, type=pa.bool_())], schema=schema)
