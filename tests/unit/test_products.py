import pandas as pd
import pytest

from src.products import PRODUCT_COLUMNS, PRODUCT_ID_MAP, PRODUCT_NAME_BY_ID, categorical_product_id, validate_product_columns


def test_product_id_mapping_is_complete_stable_and_reversible() -> None:
    assert len(PRODUCT_COLUMNS) == 24
    assert PRODUCT_ID_MAP["ind_ahor_fin_ult1"] == 0
    assert PRODUCT_ID_MAP["ind_nom_pens_ult1"] == 23
    assert {PRODUCT_NAME_BY_ID[index] for index in range(24)} == set(PRODUCT_COLUMNS)


def test_product_id_is_an_unordered_categorical_feature() -> None:
    product_id = categorical_product_id(pd.Series([0, 23, 2], name="product_id"))

    assert str(product_id.dtype) == "category"
    assert product_id.cat.ordered is False
    assert product_id.cat.categories.tolist() == list(range(24))


def test_product_mapping_validation_rejects_noncanonical_vocabulary() -> None:
    validate_product_columns(PRODUCT_COLUMNS)
    with pytest.raises(ValueError, match="canonical Santander vocabulary"):
        validate_product_columns(PRODUCT_COLUMNS[:-1])
