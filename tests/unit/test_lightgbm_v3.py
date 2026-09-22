from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.customer_history import HISTORY_FEATURE_NAMES
from src.features.persona import PERSONA_FEATURES
from src.pipeline.lightgbm_v3 import build_lightgbm_v3_panel


def test_v3_panel_combines_canonical_persona_and_history_features(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.table(
            {
                "ncodpers": [1, 1],
                "fecha_dato": ["2016-01-28", "2016-02-28"],
                "age": [30.0, 30.0],
                "ind_demo_ult1": pa.array([0, 1], type=pa.int8()),
            }
        ),
        source,
    )

    panel = build_lightgbm_v3_panel(source, tmp_path / "panel.parquet")
    columns = {
        row[0]
        for row in duckdb.connect()
        .execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(panel)])
        .fetchall()
    }

    assert set(PERSONA_FEATURES).issubset(columns)
    assert set(HISTORY_FEATURE_NAMES).issubset(columns)
    assert {"prev_ind_demo_ult1", "acq_ind_demo_ult1"}.issubset(columns)


def test_v3_preselection_materialises_candidates_not_context_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({
        "ncodpers": [1, 1, 1, 1, 2, 2, 2, 2],
        "fecha_dato": ["2016-01-28", "2016-02-28", "2016-04-28", "2016-05-28"] * 2,
        "ind_demo_ult1": pa.array([0, 1, 1, 1, 0, 0, 0, 1], type=pa.int8()),
    }), source)

    panel = build_lightgbm_v3_panel(
        source, tmp_path / "panel.parquet",
        selection_config={"validation_date": "2016-05-28", "max_negative_rows_per_product": 1, "random_state": 42},
    )
    rows = duckdb.connect().execute(
        "SELECT fecha_dato, sampled_for_ind_demo_ult1 FROM read_parquet(?) ORDER BY ncodpers, fecha_dato",
        [str(panel)],
    ).fetchall()

    # January/April are context-only. February has one positive plus one sampled
    # negative; every adjacent-month May row remains for temporal validation.
    assert all(date in {"2016-02-28", "2016-05-28"} for date, _ in rows)
    assert sum(flag for _, flag in rows if flag is not None) == 1
