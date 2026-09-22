from math import log
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.pipeline.lightgbm_v2 import build_lightgbm_v2_panel
from src.inference.predict import InputSchema


def test_persona_features_apply_mapping_and_past_only_fill(tmp_path: Path) -> None:
    """Stable profile fields may use January for February, never the reverse."""
    source = tmp_path / "profiles.parquet"
    pq.write_table(pa.table({
        "ncodpers": [7, 7],
        "fecha_dato": ["2016-01-28", "2016-02-28"],
        "ind_demo_ult1": pa.array([0, 1], type=pa.int8()),
        "fecha_alta": ["2015-01-01", None], "age": [35.0, None], "renta": [99.0, None],
        "ind_nuevo": ["1", None], "indrel": ["1.0", None], "indrel_1mes": ["1.0", None],
        "tiprel_1mes": ["A", None], "ult_fec_cli_1t": [None, None],
        "ind_actividad_cliente": ["1", None], "ind_empleado": ["A", None],
        "sexo": ["H", None], "indfall": ["N", None], "pais_residencia": ["ES", None],
        "cod_prov": ["28", None], "indresi": ["S", None], "canal_entrada": ["KHE", None],
        "segmento": ["01 - TOP", None], "antiguedad": [12, None], "tipodom": [1, None], "indext": ["N", None],
    }), source)

    panel = build_lightgbm_v2_panel(source, tmp_path / "panel.parquet")
    rows = duckdb.connect().execute(
        """SELECT time_idx, snapshot_month, account_age_months, age, income_log,
                  is_new_customer, is_active_customer, is_male, is_deceased,
                  is_domestic, profile_missing_structural
           FROM read_parquet(?) ORDER BY fecha_dato""",
        [str(panel)],
    ).fetchall()
    assert rows[0] == (0, 1, 12, 35.0, pytest.approx(log(100)), 1, 1, 1, 0, 1, 0)
    # Dynamic states remain unknown; stable fields use only the preceding row.
    assert rows[1] == (1, 2, 13, 35.0, pytest.approx(log(100)), None, None, 1, 0, 1, 1)


def test_persona_categoricals_remain_semantic_values_with_explicit_missing(tmp_path: Path) -> None:
    source = tmp_path / "profiles.parquet"
    pq.write_table(pa.table({
        "ncodpers": [7, 7], "fecha_dato": ["2016-01-28", "2016-02-28"],
        "ind_demo_ult1": pa.array([0, 1], type=pa.int8()),
        "indrel_1mes": ["1.0", None], "ind_empleado": ["A", None],
        "pais_residencia": ["ES", None], "cod_prov": ["28", None],
        "canal_entrada": ["KHE", None], "segmento": ["01 - TOP", None],
    }), source)

    panel = build_lightgbm_v2_panel(source, tmp_path / "panel.parquet")
    rows = duckdb.connect().execute(
        """SELECT customer_relationship_status, employee_status, country,
                  province, entry_channel, customer_segment
           FROM read_parquet(?) ORDER BY fecha_dato""",
        [str(panel)],
    ).fetchall()
    assert rows == [
        ("1", "A", "ES", "28", "KHE", "01 - TOP"),
        ("MISSING", "A", "ES", "28", "KHE", "MISSING"),
    ]


def test_input_schema_reuses_training_category_vocabulary() -> None:
    schema = InputSchema(
        feature_names=["country"], dtypes={"country": "category"}, version="1",
        category_values={"country": ["ES", "FR", "MISSING"]},
    )
    prepared = schema.validate(pd.DataFrame({"country": ["ES", "UNKNOWN", "MISSING"]}))
    assert str(prepared["country"].dtype) == "category"
    assert prepared["country"].cat.categories.tolist() == ["ES", "FR", "MISSING"]
    assert prepared["country"].tolist()[0] == "ES"
    assert pd.isna(prepared["country"].iloc[1])
