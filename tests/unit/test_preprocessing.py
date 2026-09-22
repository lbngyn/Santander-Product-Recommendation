from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.preprocessing import fit_profile_preprocessing, transform_customer_profiles


def test_profile_preprocessing_uses_only_past_history_and_applies_documented_cleaning(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    processed_train = tmp_path / "processed_train.parquet"
    processed_test = tmp_path / "processed_test.parquet"
    pq.write_table(
        pa.table(
            {
                "ncodpers": [1, 1, 2],
                "fecha_dato": ["2015-01-28", "2015-02-28", "2015-01-28"],
                "age": [35.0, None, 2.0],
                "antiguedad": [5, None, -999999],
                "renta": [100.0, None, 1000.0],
                "sexo": ["H", None, None],
                "pais_residencia": ["ES", None, None],
                "cod_prov": [28, None, None],
                "fecha_alta": ["2014-01-01", None, None],
                "indrel_1mes": ["1.0", None, "P"],
                "indfall": ["N", None, None],
                "conyuemp": [None, None, None],
                "nomprov": ["MADRID", None, None],
                "ult_fec_cli_1t": [None, None, None],
                "tipodom": [1, None, None],
            }
        ),
        train_path,
    )
    pq.write_table(
        pa.table(
            {
                "ncodpers": [1],
                "fecha_dato": ["2015-03-28"],
                "age": [None],
                "antiguedad": [None],
                "renta": [None],
                "sexo": [None],
                "pais_residencia": [None],
                "cod_prov": [None],
                "fecha_alta": [None],
                "indrel_1mes": ["1.0"],
                "indfall": [None],
                "conyuemp": [None],
                "nomprov": [None],
                "ult_fec_cli_1t": [None],
                "tipodom": [None],
            }
        ),
        test_path,
    )

    stats = fit_profile_preprocessing(train_path)
    transform_customer_profiles(train_path, processed_train, stats)
    transform_customer_profiles(test_path, processed_test, stats, history_path=train_path)

    con = duckdb.connect()
    try:
        train_rows = con.execute(
            """
            SELECT ncodpers, fecha_dato, age, antiguedad, renta, sexo,
                   pais_residencia, cod_prov, fecha_alta, indrel_1mes, indfall
            FROM read_parquet(?)
            ORDER BY ncodpers, fecha_dato
            """,
            [str(processed_train)],
        ).fetchall()
        test_rows = con.execute(
            """
            SELECT age, antiguedad, renta, sexo, pais_residencia, cod_prov,
                   fecha_alta, indrel_1mes, indfall
            FROM read_parquet(?)
            """,
            [str(processed_test)],
        ).fetchall()
        output_columns = [row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(processed_train)]).fetchall()]
    finally:
        con.close()

    # Age=2 is retained: the EDA explicitly rejects blanket 18--90 clipping.
    assert train_rows[1][2:9] == (35.0, 5.0, 100.0, "H", "ES", "28", "2014-01-01")
    assert train_rows[2][2] == 2.0
    assert train_rows[2][3] == stats.antiguedad_median
    assert train_rows[2][5:8] == ("__MISSING__", "__MISSING__", "__MISSING__")
    assert train_rows[2][8] is None
    assert train_rows[0][9] == "1"
    assert train_rows[1][9:11] == ("__MISSING__", "__MISSING__")
    assert test_rows == [(35.0, 5.0, 100.0, "H", "ES", "28", "2014-01-01", "1", "__MISSING__")]
    assert set(("conyuemp", "nomprov", "ult_fec_cli_1t", "tipodom")).isdisjoint(output_columns)
    assert all(stats.renta_clip_lower <= row[4] <= stats.renta_clip_upper for row in train_rows)
