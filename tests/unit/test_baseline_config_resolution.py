from src.pipeline.baseline_v0 import _resolve_environment


def test_environment_resolution_uses_hydra_default_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("SANTANDER_DUCKDB_TEMP_DIRECTORY", raising=False)
    assert _resolve_environment("${oc.env:SANTANDER_DUCKDB_TEMP_DIRECTORY,data/.duckdb_tmp}") == "data/.duckdb_tmp"
