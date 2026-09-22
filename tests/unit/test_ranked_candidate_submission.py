from pathlib import Path

import pandas as pd

from src.submission.competition import run_ranked_candidate_submission


def test_ranked_candidate_submission_preserves_template_schema_order_and_short_lists(tmp_path: Path) -> None:
    prepared = pd.DataFrame({"ncodpers": [2, 1, 3], "feature": [0, 0, 0]})
    template = tmp_path / "sample_submission.csv"
    pd.DataFrame({"ncodpers": [1, 2, 3], "added_products": ["", "", ""]}).to_csv(template, index=False)

    def score_batch(batch: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for customer in batch["ncodpers"]:
            if customer == 1:
                rows.extend([(1, "product_b", 0.2), (1, "product_a", 0.9)])
            if customer == 2:
                rows.append((2, "product_c", 0.5))
        return pd.DataFrame(rows, columns=["ncodpers", "product", "score"])

    output_path = run_ranked_candidate_submission(
        prepared, template, tmp_path / "submission.csv", score_batch=score_batch, top_k=2, batch_customers=2,
    )
    output = pd.read_csv(output_path, keep_default_na=False)

    assert list(output.columns) == ["ncodpers", "added_products"]
    assert output.to_dict("records") == [
        {"ncodpers": 1, "added_products": "product_a product_b"},
        {"ncodpers": 2, "added_products": "product_c"},
        {"ncodpers": 3, "added_products": ""},
    ]
