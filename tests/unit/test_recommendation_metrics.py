import numpy as np
import pytest

from src.evaluation.recommendation import recommendation_metrics


def test_recommendation_metrics_respect_rank_and_ignore_masked_slots() -> None:
    targets = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)
    ranking = np.array([[0, 1], [0, 1]])
    scores = np.array([[0.9, 0.8], [0.7, float("-inf")]])

    metrics = recommendation_metrics(targets, ranking, scores, top_k=2)

    assert metrics["map_at_2"] == 0.25
    assert metrics["product_recall_at_2"] == 1 / 3
    assert metrics["customer_recall_at_2"] == 0.5
    assert metrics["ndcg_at_2"] == pytest.approx(0.3065735964)
