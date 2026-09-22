from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.pipeline.baseline_v0 import run_baseline_v0


def test_baseline_pipeline_uses_upstream_parquet_and_records_lineage(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    pq.write_table(pa.table({
        "ncodpers": [1, 1, 1],
        "fecha_dato": ["2016-03-28", "2016-04-28", "2016-05-28"],
        "ind_cco_fin_ult1": pa.array([0, 1, 1], type=pa.int8()),
        "ind_recibo_ult1": pa.array([0, 0, 1], type=pa.int8()),
    }), train_path)
    config = {
        "pipeline": {"version": "baseline-v0"},
        "data": {"train_path": str(train_path), "output_dir": str(tmp_path / "processed")},
        "split": {"validation_date": "2016-05-28"},
        "model": {"top_k": 2},
        "runtime": {"memory_limit": "1GB", "temp_directory": str(tmp_path / "duckdb")},
        "tracking": {"artifact_dir": str(tmp_path / "artifacts")},
    }
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text("pipeline: baseline-v0\n", encoding="utf-8")

    result = run_baseline_v0(config, resolved_config_path=config_path)

    assert Path(result["manifest"]).is_file()
    assert Path(result["train"]) == train_path
    assert duckdb.connect().execute("SELECT COUNT(*) FROM read_parquet(?)", [str(tmp_path / "processed" / "validation_predictions.parquet")]).fetchone()[0] == 1
