from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.models.popularity import run_popularity_baseline


def test_popularity_excludes_previously_owned_products_and_uses_only_history(tmp_path: Path) -> None:
    source = tmp_path / "train.parquet"
    pq.write_table(pa.table({
        "ncodpers": [1, 1, 1, 2, 2, 2],
        "fecha_dato": ["2016-03-28", "2016-04-28", "2016-05-28"] * 2,
        "age": [30, 30, 31, 40, 40, 41],
        "ind_a_ult1": pa.array([0, 1, 1, 0, 0, 1], type=pa.int8()),
        "ind_b_ult1": pa.array([0, 0, 1, 0, 1, 1], type=pa.int8()),
    }), source)

    result = run_popularity_baseline(source, tmp_path / "out", validation_date="2016-05-28", top_k=1)
    prediction = duckdb.connect().execute(
        "SELECT ncodpers, added_products FROM read_parquet(?) ORDER BY ncodpers",
        [str(tmp_path / "out" / "validation_predictions.parquet")],
    ).fetchall()
    input_columns = [row[0] for row in duckdb.connect().execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(tmp_path / "out" / "validation_input.parquet")]
    ).fetchall()]

    assert prediction == [(1, "ind_b_ult1"), (2, "ind_a_ult1")]
    assert "ind_a_ult1" not in input_columns and "prev_ind_a_ult1" in input_columns
    assert result["metrics"]["validation_date"] == "2016-05-28"
