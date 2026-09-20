from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.submission.prepare import materialize_competition_test_input


def test_competition_input_uses_may_history_and_test_month_profile(tmp_path: Path) -> None:
    products = [f"product_{index}_ult1" for index in range(24)]
    history = {
        "ncodpers": [1, 1, 2],
        "fecha_dato": ["2016-04-28", "2016-05-28", "2016-05-28"],
        "age": [30.0, 31.0, 40.0],
    }
    for index, product in enumerate(products):
        history[product] = pa.array([0, 1 if index == 0 else 0, 1 if index == 1 else 0], type=pa.int8())
    history_path = tmp_path / "train.parquet"
    pq.write_table(pa.table(history), history_path)
    test_path = tmp_path / "test.parquet"
    pq.write_table(pa.table({"ncodpers": [1, 2, 3], "fecha_dato": ["2016-06-28"] * 3, "age": [32.0, 41.0, 50.0]}), test_path)

    prepared = materialize_competition_test_input(
        test_path, history_path, tmp_path / "prepared.parquet", product_names=products,
        feature_names=["age", "prev_product_0_ult1"], history_date="2016-05-28",
    )
    rows = duckdb.connect().execute(
        "SELECT ncodpers, age, prev_product_0_ult1, prev_product_1_ult1 FROM read_parquet(?) ORDER BY ncodpers",
        [str(prepared)],
    ).fetchall()
    assert rows == [(1, 32.0, 1, 0), (2, 41.0, 0, 1), (3, 50.0, 0, 0)]
