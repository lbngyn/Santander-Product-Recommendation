"""Temporal Top-K evaluation for independent product-acquisition models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import duckdb
import numpy as np
import pandas as pd

from src.inference.predict import load_artifact, load_artifact_metadata, predict


def evaluate_independent_models(
    panel_path: str | Path,
    artifact_directories: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    validation_date: str,
    top_k: int,
) -> dict[str, float | int | str]:
    """Score a temporal holdout and persist masked Top-K recommendations."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    products = list(artifact_directories)
    metadata = {product: load_artifact_metadata(path) for product, path in artifact_directories.items()}
    feature_names = sorted({feature for artifact in metadata.values() for feature in artifact.schema.feature_names})
    required = ["ncodpers", *feature_names, *["prev_" + product for product in products], *["acq_" + product for product in products]]
    frame = _load_holdout(panel_path, required, validation_date)
    if frame.empty:
        raise ValueError(f"No adjacent-month validation rows found for {validation_date}.")

    scores: dict[str, pd.Series] = {}
    for product, directory in artifact_directories.items():
        artifact, model = load_artifact(directory)
        if artifact.product != product:
            raise ValueError(f"Artifact product mismatch: key={product}, artifact={artifact.product}")
        scores[product] = predict(frame.loc[:, artifact.schema.feature_names], artifact, model)
    score_frame = pd.DataFrame(scores, index=frame.index)
    for product in products:
        score_frame.loc[pd.to_numeric(frame["prev_" + product], errors="raise").eq(1), product] = -np.inf

    values = score_frame.to_numpy(copy=False)
    product_array = np.asarray(products, dtype=object)
    ranking = np.argsort(-values, axis=1, kind="stable")[:, :top_k]
    selected_scores = np.take_along_axis(values, ranking, axis=1)
    selected_products = product_array[ranking]
    recommendations = pd.DataFrame({
        "ncodpers": frame["ncodpers"].to_numpy(),
        "added_products": [
            " ".join(row[np.isfinite(row_scores)])
            for row, row_scores in zip(selected_products, selected_scores, strict=True)
        ],
    })
    targets = frame.loc[:, ["acq_" + product for product in products]].to_numpy(dtype=np.int8, copy=False)
    metrics = recommendation_metrics(targets, ranking, selected_scores, top_k=top_k)
    metrics.update({"validation_date": validation_date, "top_k": top_k, "validation_customers": int(len(frame))})

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(destination / "validation_recommendations.csv", index=False)
    (destination / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def recommendation_metrics(targets: np.ndarray, ranking: np.ndarray, selected_scores: np.ndarray, *, top_k: int) -> dict[str, float | int]:
    """Compute MAP, micro/product recall, customer recall and NDCG at ``k``."""
    if targets.ndim != 2 or ranking.shape != selected_scores.shape or ranking.shape[0] != targets.shape[0]:
        raise ValueError("Targets, ranking and selected scores have incompatible shapes.")
    valid = np.isfinite(selected_scores)
    hits = targets[np.arange(len(targets))[:, None], ranking].astype(bool) & valid
    positive_counts = targets.sum(axis=1)
    ranks = np.arange(1, ranking.shape[1] + 1, dtype=float)
    cumulative_hits = np.cumsum(hits, axis=1)
    denominators = np.minimum(positive_counts, top_k)
    average_precision = np.divide(
        (cumulative_hits / ranks * hits).sum(axis=1), denominators,
        out=np.zeros(len(targets), dtype=float), where=denominators > 0,
    )
    discounts = 1.0 / np.log2(ranks + 1.0)
    dcg = (hits * discounts).sum(axis=1)
    ideal = np.array([discounts[:min(int(count), top_k)].sum() for count in positive_counts])
    ndcg = np.divide(dcg, ideal, out=np.zeros(len(targets), dtype=float), where=ideal > 0)
    total_positives = int(positive_counts.sum())
    positive_customers = positive_counts > 0
    return {
        f"map_at_{top_k}": float(average_precision.mean()),
        f"product_recall_at_{top_k}": float(hits.sum() / total_positives) if total_positives else 0.0,
        f"customer_recall_at_{top_k}": float(hits.any(axis=1)[positive_customers].mean()) if positive_customers.any() else 0.0,
        f"ndcg_at_{top_k}": float(ndcg.mean()),
        "validation_positive_products": total_positives,
        "validation_positive_customers": int(positive_customers.sum()),
    }


def _load_holdout(panel_path: str | Path, columns: list[str], validation_date: str) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    try:
        available = {row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(panel_path)]).fetchall()}
        missing = set(columns).difference(available)
        if missing:
            raise ValueError(f"Validation panel is missing columns: {sorted(missing)}")
        projection = ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
        return con.execute(
            f"SELECT {projection} FROM read_parquet(?) WHERE CAST(fecha_dato AS DATE) = CAST(? AS DATE) AND record_gap_months = 1",
            [str(panel_path), validation_date],
        ).fetchdf()
    finally:
        con.close()
