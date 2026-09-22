import pandas as pd

from src.models.lightgbm_joint import expand_joint_inference_candidates, rank_joint_candidates
from src.products import PRODUCT_COLUMNS, PRODUCT_ID_MAP


def test_inference_expansion_only_emits_unowned_products_and_uses_category_dtype() -> None:
    row = {"ncodpers": 10, "fecha_dato": "2016-06-28", "numeric_feature": 1.5}
    row.update({"prev_" + product: 0 for product in PRODUCT_COLUMNS})
    row["prev_ind_cco_fin_ult1"] = 1

    candidates = expand_joint_inference_candidates(pd.DataFrame([row]), feature_names=["numeric_feature"])

    assert len(candidates) == 23
    assert str(candidates["product_id"].dtype) == "category"
    assert PRODUCT_ID_MAP["ind_cco_fin_ult1"] not in candidates["product_id"].astype("uint8").tolist()


def test_joint_ranking_returns_original_product_names() -> None:
    candidates = pd.DataFrame({
        "ncodpers": [1, 1, 1],
        "product_id": pd.Categorical([0, 1, 2], categories=list(range(24)), ordered=False),
    })
    ranked = rank_joint_candidates(candidates, [0.2, 0.8, 0.5], top_k=2)

    assert ranked["product"].tolist() == ["ind_aval_fin_ult1", "ind_cco_fin_ult1"]
