import json
from pathlib import Path

from src.tracking.mlflow_utils import collect_lightgbm_training_metrics


def test_collect_lightgbm_training_metrics_summarises_product_artifacts(tmp_path: Path) -> None:
    metrics_a = tmp_path / "models" / "product_a" / "training_metrics.json"
    metrics_b = tmp_path / "models" / "product_b" / "training_metrics.json"
    metrics_a.parent.mkdir(parents=True)
    metrics_b.parent.mkdir(parents=True)
    metrics_a.write_text(json.dumps({
        "eligible_rows": 100, "acquisitions": 5, "duration_seconds": 2.0,
        "rss_peak_during_fit_bytes": 1000, "negative_sampling_strategy": "random",
    }), encoding="utf-8")
    metrics_b.write_text(json.dumps({
        "eligible_rows": 50, "acquisitions": 10, "duration_seconds": 4.0,
        "rss_peak_during_fit_bytes": 2000,
    }), encoding="utf-8")

    summary, per_product = collect_lightgbm_training_metrics(tmp_path)

    assert set(per_product) == {"product_a", "product_b"}
    assert summary == {
        "products_trained": 2.0,
        "eligible_rows_total": 150.0,
        "acquisitions_total": 15.0,
        "positive_rate_weighted": 0.1,
        "train_duration_seconds_total": 6.0,
        "train_duration_seconds_max": 4.0,
        "rss_peak_during_fit_bytes_max": 2000.0,
    }
