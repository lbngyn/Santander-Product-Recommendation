from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.model_panel import build_model_panel


def test_history_features_exclude_current_event_and_ignore_gapped_transition(tmp_path: Path) -> None:
    source = tmp_path / "checkpoints.parquet"
    pq.write_table(pa.table({
        "ncodpers": [1, 1, 1, 1],
        "fecha_dato": ["2016-01-28", "2016-02-28", "2016-04-28", "2016-05-28"],
        "ind_a_ult1": pa.array([0, 1, 0, 1], type=pa.int8()),
        "ind_b_ult1": pa.array([1, 1, 0, 0], type=pa.int8()),
    }), source)

    panel = build_model_panel(source, tmp_path / "panel.parquet")
    rows = duckdb.connect().execute(
        """SELECT fecha_dato, record_gap_months, customer_history_length,
                  products_owned_count, cumulative_acquisitions, cumulative_drops,
                  months_since_last_acquisition, never_acquired_before,
                  acquisitions_last_1m, acquisitions_last_3m, acquisitions_last_6m,
                  acq_ind_a_ult1, acq_ind_b_ult1
           FROM read_parquet(?) ORDER BY fecha_dato""",
        [str(panel)],
    ).fetchall()

    assert rows == [
        ("2016-01-28", None, 0, 0, 0, 0, None, 1, 0, 0, 0, 0, 0),
        ("2016-02-28", 1, 1, 1, 0, 0, None, 1, 0, 0, 0, 1, 0),
        # The 1->0 states across the two-month gap are not drops.
        ("2016-04-28", 2, 2, 2, 1, 0, 2, 0, 0, 1, 1, 0, 0),
        # The April->May acquisition is the label, never part of t's features.
        ("2016-05-28", 1, 3, 0, 1, 0, 3, 0, 0, 1, 1, 1, 0),
    ]
