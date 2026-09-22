from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.models.lightgbm_joint_native import CompactJointDatasetBuilder
from src.products import PRODUCT_COLUMNS


def test_native_candidate_writer_expands_only_unowned_products_in_bounded_batches(tmp_path: Path) -> None:
    row = {"ncodpers": 1, "fecha_dato": "2016-04-28", "previous_observation": 1, "feature": 3.0}
    row.update({"prev_" + product: 0 for product in PRODUCT_COLUMNS})
    row.update({"acq_" + product: 0 for product in PRODUCT_COLUMNS})
    row["prev_ind_cco_fin_ult1"] = 1
    row["acq_ind_aval_fin_ult1"] = 1
    panel = tmp_path / "panel.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame([row])), panel)

    builder = CompactJointDatasetBuilder(panel)
    compact = builder.materialize_compact(tmp_path / "compact.parquet", feature_names=["feature", *["prev_" + product for product in PRODUCT_COLUMNS]], months=["2016-04-28"])
    files = builder.write_native_candidates(compact, tmp_path / "candidates.csv", feature_names=["feature", *["prev_" + product for product in PRODUCT_COLUMNS]], batch_customer_months=1)

    candidate = pd.read_csv(files.text_path, header=None)
    assert len(candidate) == 23
    assert candidate.iloc[:, 0].sum() == 1
    assert candidate.iloc[:, -1].between(0, len(PRODUCT_COLUMNS) - 1).all()
