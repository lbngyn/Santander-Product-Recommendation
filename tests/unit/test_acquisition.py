from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.acquisition import build_acquisition_features


def test_acquisition_targets_use_nearest_previous_customer_record(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    pq.write_table(
        pa.table(
            {
                "ncodpers": [1, 1, 1, 2, 2],
                "fecha_dato": ["2016-01-28", "2016-03-28", "2016-04-28", "2016-01-28", "2016-02-28"],
                "ind_x_ult1": pa.array([0, 1, 1, 1, 0], type=pa.int8()),
            }
        ),
        train_path,
    )

    result = build_acquisition_features(
        train_path,
        test_path=None,
        processed_dir=tmp_path / "processed",
        memory_limit="1GB",
        temp_directory=tmp_path / "duckdb_temp",
    )

    rows = duckdb.connect().execute(
        """
        SELECT ncodpers, fecha_dato, ind_x_ult1, acq_ind_x_ult1, typeof(acq_ind_x_ult1)
        FROM read_parquet(?)
        ORDER BY ncodpers, fecha_dato
        """,
        [result["train"]],
    ).fetchall()

    assert rows == [
        (1, "2016-01-28", 0, 0, "TINYINT"),
        (1, "2016-03-28", 1, 1, "TINYINT"),
        (1, "2016-04-28", 1, 0, "TINYINT"),
        (2, "2016-01-28", 1, 0, "TINYINT"),
        (2, "2016-02-28", 0, 0, "TINYINT"),
    ]
