"""Canonical Santander product identity used by model-specific strategies."""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


# The competition's product-column order is the stable product vocabulary.  It
# must not be inferred independently by train and test code.
PRODUCT_COLUMNS: tuple[str, ...] = (
    "ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1",
    "ind_cder_fin_ult1", "ind_cno_fin_ult1", "ind_ctju_fin_ult1",
    "ind_ctma_fin_ult1", "ind_ctop_fin_ult1", "ind_ctpp_fin_ult1",
    "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1",
    "ind_plan_fin_ult1", "ind_pres_fin_ult1", "ind_reca_fin_ult1",
    "ind_recibo_ult1", "ind_tjcr_fin_ult1", "ind_valo_fin_ult1",
    "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1",
)

PRODUCT_ID_MAP: dict[str, int] = {product: index for index, product in enumerate(PRODUCT_COLUMNS)}
PRODUCT_NAME_BY_ID: dict[int, str] = {index: product for product, index in PRODUCT_ID_MAP.items()}


def validate_product_columns(columns: Sequence[str]) -> None:
    """Fail early when a panel does not expose the canonical 24 products."""
    unexpected = set(columns).difference(PRODUCT_COLUMNS)
    missing = set(PRODUCT_COLUMNS).difference(columns)
    if missing or unexpected or len(columns) != len(PRODUCT_COLUMNS):
        raise ValueError(
            "Product columns must be exactly the canonical Santander vocabulary; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )


def categorical_product_id(values: pd.Series) -> pd.Series:
    """Return an unordered categorical product identity for LightGBM input.

    Product IDs are stored in the mapping as compact ``uint8`` values, but the
    model input must be categorical: their integer labels have no ordinal
    meaning.  The complete category vocabulary is retained even if a small
    train split contains no examples of one product.
    """
    numeric = pd.to_numeric(values, errors="raise").astype("uint8")
    invalid = ~numeric.isin(PRODUCT_NAME_BY_ID)
    if invalid.any():
        raise ValueError(f"product_id contains invalid values: {sorted(numeric[invalid].unique().tolist())}")
    dtype = pd.CategoricalDtype(categories=list(range(len(PRODUCT_COLUMNS))), ordered=False)
    return pd.Series(pd.Categorical(numeric, dtype=dtype), index=values.index, name=values.name)
