from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing.customer_profile import (
    fit_baseline_profile_preprocessing,
    transform_customer_profiles_bidirectional,
)


def test_baseline_uses_customer_neighbours_preserves_columns_and_marks_residual_categories(tmp_path: Path) -> None:
    source = tmp_path / "train.parquet"
    destination = tmp_path / "processed.parquet"
    pq.write_table(
        pa.table(
            {
                "ncodpers": [1, 1, 1, 2],
                "fecha_dato": ["2015-01-28", "2015-02-28", "2015-03-28", "2015-01-28"],
                "age": [20.0, None, 40.0, None],
                "antiguedad": [10.0, -999999.0, 30.0, None],
                "renta": [100.0, None, 300.0, None],
                "sexo": ["H", None, "V", None],
                "fecha_alta": ["2014-01-01", None, "2014-01-01", None],
                "indrel_1mes": ["1.0", None, "1", None],
                "tipodom": [1, None, 1, None],
                "ind_cco_fin_ult1": pa.array([0, 0, 1, 0], type=pa.int8()),
            }
        ),
        source,
    )

    stats = fit_baseline_profile_preprocessing(source)
    transform_customer_profiles_bidirectional(source, destination, stats, allow_future_values=True)
    rows = duckdb.connect().execute(
        """
        SELECT ncodpers, fecha_dato, age, antiguedad, renta, sexo,
               fecha_alta, indrel_1mes, tipodom, ind_cco_fin_ult1
        FROM read_parquet(?) ORDER BY ncodpers, fecha_dato
        """,
        [str(destination)],
    ).fetchall()

    # Numeric gap: arithmetic mean of the two observed customer records.
    assert rows[1][2:5] == (30.0, 20.0, 200.0)
    # The closest categorical record is selected (the following March record).
    assert rows[1][5:9] == ("V", "2014-01-01", "1", "1")
    # No customer observation: numeric falls back to fitted median; category is explicit.
    assert rows[3][2] == stats.age_median
    assert rows[3][5] == "__MISSING__"
    # No encoding/drop: original profile and product columns remain.
    assert rows[3][8:] == ("__MISSING__", 0)
