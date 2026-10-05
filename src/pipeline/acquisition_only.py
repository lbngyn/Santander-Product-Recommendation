"""Notebook entry points; no changes to legacy model orchestration."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import yaml

from src.features import acquisition_only as data
from src.evaluation import acquisition_only as evaluation
from src.models import lightgbm_acquisition_only as models
from src.utils.run_files import now, read_json, write_json
from src.tracking.run_bundle import build_bundle, environment, safe_id
from src.tracking.run_context import build_run_manifest, create_run_id, write_run_manifest


def load_config(path, *, comparison_id=None, checkpoint_version=None):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Pipeline YAML must contain a mapping")
    for section in ("pipeline", "data", "split", "features", "model", "runtime", "tracking"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"Missing pipeline configuration section: {section}")
    expected_features = {"scope": "original_information", "income_policy": "accepted_canonical_static",
                         "acquisition_policy": "nearest_previous_observed_record"}
    if config["features"] != expected_features:
        raise ValueError("Feature policies differ from this pipeline's approved contract")
    if comparison_id is not None:
        config["pipeline"]["comparison_id"] = safe_id(comparison_id)
    if checkpoint_version is not None:
        config["data"]["checkpoint_version"] = int(checkpoint_version)
    if config["pipeline"]["approach"] not in {"independent", "joint"}:
        raise ValueError("approach must be independent or joint")
    safe_id(config["pipeline"]["comparison_id"]); safe_id(config["pipeline"]["project_id"])
    if int(config["runtime"]["batch_customer_months"]) < 1 or int(config["model"]["n_estimators"]) < 1:
        raise ValueError("Positive batch size and estimator count required")
    if int(config["runtime"]["threads"]) < 1:
        raise ValueError("threads must be positive")
    return config


def locations(config):
    root = Path(os.getenv("SANTANDER_DATA_ROOT", "data")).expanduser().resolve()
    work = Path(os.getenv("SANTANDER_MODELING_WORK_ROOT", "/content/santander_modeling_work" if os.getenv("SANTANDER_RUNTIME") == "colab" else str(root / ".modeling_work")))
    return root, work.resolve()


def preview(config):
    root, work = locations(config)
    return data.preview(config, root, work / config["pipeline"]["comparison_id"])


def prepare_dataset(config):
    root, work = locations(config)
    return data.prepare(config, root, work / config["pipeline"]["comparison_id"])


def train_run(config, dataset):
    root, work = locations(config)
    dataset = Path(dataset).resolve()
    manifest = read_json(dataset / "dataset_manifest.json")
    data._check_config(config, manifest)
    for name, digest in manifest["files"].items():
        from src.utils.run_files import sha256
        if sha256(dataset / name) != digest:
            raise ValueError(f"Prepared data changed before training: {name}")
    run_id = create_run_id("acquisition-original-" + config["pipeline"]["approach"])
    run = root / "artifacts/runs" / run_id; run.mkdir(parents=True)
    result = {"schema_version": 1, "project_id": config["pipeline"]["project_id"], "source_run_id": run_id,
              "experiment_name": config["tracking"]["mlflow"]["experiment_name"], "approach": config["pipeline"]["approach"],
              "model_version": config["model"]["version"], "comparison_id": config["pipeline"]["comparison_id"],
              "runtime": os.getenv("SANTANDER_RUNTIME", "local"), "started_at": now(), "ended_at": None,
              "status": "RUNNING", "stages": {"train": "running", "validation": "not_run"}, "inference_ready": False}
    write_json(run / "run_result.json", result)
    try:
        (run / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        for name in ("dataset_manifest.json", "feature_contract.json", "cohort_audit.json"):
            write_json(run / name, read_json(Path(dataset) / name))
        write_json(run / "environment.json", environment())
        provenance = build_run_manifest(run_id=run_id, config=config, inputs={"dataset_manifest": Path(dataset) / "dataset_manifest.json"}, outputs={})
        write_run_manifest(provenance, run / "lineage_manifest.json")
        models.train(dataset, run, config, work / run_id)
        result["stages"]["train"] = "completed"; result["inference_ready"] = True
        result["status"] = "TRAINED"
    except KeyboardInterrupt:
        result["status"] = "KILLED"; result["stages"]["train"] = "killed"
        result["ended_at"] = now()
        raise
    except Exception as error:
        result["status"] = "FAILED"; result["stages"]["train"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        result["ended_at"] = now()
        raise
    finally:
        write_json(run / "run_result.json", result)
    return run


def validate_run(config, dataset, run):
    _, work = locations(config); run = Path(run)
    result = read_json(run / "run_result.json")
    if result["status"] != "TRAINED":
        raise ValueError("Validation requires this run's completed training")
    if read_json(Path(dataset) / "dataset_manifest.json") != read_json(run / "dataset_manifest.json"):
        raise ValueError("Validation dataset differs from trained dataset")
    from src.utils.run_files import sha256
    for name, digest in read_json(run / "dataset_manifest.json")["files"].items():
        if sha256(Path(dataset) / name) != digest:
            raise ValueError(f"Dataset changed before validation: {name}")
    trained_config = yaml.safe_load((run / "config_resolved.yaml").read_text(encoding="utf-8"))
    if config != trained_config:
        raise ValueError("Use the immutable config from this training run")
    try:
        result["stages"]["validation"] = "running"; write_json(run / "run_result.json", result)
        metrics = evaluation.evaluate(dataset, run, config, work / result["source_run_id"])
        result["stages"]["validation"] = "completed"; result["status"] = "FINISHED"
        return metrics
    except KeyboardInterrupt:
        result["status"] = "KILLED"; result["stages"]["validation"] = "killed"
        raise
    except Exception as error:
        result["stages"]["validation"] = "failed"; result["status"] = "FAILED"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["ended_at"] = now(); write_json(run / "run_result.json", result)


def finalize_run(run):
    run = Path(run).resolve(); result = read_json(run / "run_result.json")
    if result["status"] not in {"FINISHED", "FAILED", "KILLED"}:
        raise ValueError("Finish validation or explicitly record failure before finalizing")
    root = run.parents[2]
    return build_bundle(run, root / "artifacts/bundles" / result["source_run_id"])


def publish_run(bundle, config):
    from src.tracking.gcs_runs import publish_bundle
    bucket = config["tracking"]["gcs"].get("bucket") or os.getenv("GCS_BUCKET")
    if not bucket:
        raise ValueError("Set GCS_BUCKET or tracking.gcs.bucket")
    return publish_bundle(bundle, bucket, config["tracking"]["gcs"]["prefix"])


def comparison(run_a, run_b):
    paths = [Path(run_a), Path(run_b)]
    results = [read_json(p / "run_result.json") for p in paths]
    if {r["approach"] for r in results} != {"independent", "joint"} or any(r["status"] != "FINISHED" for r in results):
        raise ValueError("Comparison requires one FINISHED independent and one FINISHED joint run")
    for name in ("dataset_manifest.json", "feature_contract.json", "evaluation_contract.json"):
        values = [read_json(p / name) for p in paths]
        if name == "dataset_manifest.json":
            values = [{k: v[k] for k in ("comparison_id", "checkpoint_version", "checkpoint_sha256", "validation_start", "validation_end", "files")} for v in values]
        if values[0] != values[1]:
            raise ValueError(f"Unfair comparison: {name} differs")
    configs = [yaml.safe_load((p / "config_resolved.yaml").read_text(encoding="utf-8")) for p in paths]
    for key in ("model", "runtime"):
        budgets = [{k: v for k, v in c[key].items() if k != "version"} for c in configs]
        if budgets[0] != budgets[1]:
            raise ValueError(f"Comparison budgets differ: {key}")
    rows = []
    for p, result in zip(paths, results):
        metrics = read_json(p / "validation_metrics.json"); train = read_json(p / "training_metrics.json")
        rows.append({"run_id": result["source_run_id"], "approach": result["approach"], **metrics,
                     "train_seconds": train["duration_seconds"], "train_peak_rss_bytes": train["peak_rss_bytes"],
                     "model_size_bytes": train["model_size_bytes"], "number_of_models": train["number_of_models"]})
    return pd.DataFrame(rows)


def create_submission(run, *, data_root, work_root, raw_test=None, sample_submission=None, history_path=None):
    """Prepare June input and write submission outside the immutable run/bundle."""
    from src.ingestion.checkpoint import build_interim_checkpoint
    from src.submission.prepare import materialize_acquisition_only_test_input
    from src.submission.competition import run_ranked_candidate_submission
    from src.utils.run_files import sha256
    from src.products import PRODUCT_COLUMNS

    run, root, work_root = map(lambda p: Path(p).resolve(), (run, data_root, work_root))
    result = read_json(run / "run_result.json")
    if result["status"] != "FINISHED" or not result.get("inference_ready"):
        raise ValueError("Submission requires a FINISHED, inference-ready run")
    config = yaml.safe_load((run / "config_resolved.yaml").read_text(encoding="utf-8"))
    manifest = read_json(run / "dataset_manifest.json")
    history = Path(history_path or manifest["checkpoint_path"])
    if not history.is_file() or sha256(history) != manifest["checkpoint_sha256"]:
        raise ValueError("Pinned canonical history missing or changed; supply history_path with the same checksum")
    if read_json(run / "feature_contract.json") != manifest["feature_contract"]:
        raise ValueError("Run feature contract differs from the trained dataset")
    from src.tracking.run_bundle import safe_id
    run_id = safe_id(result["source_run_id"])
    work = work_root / run_id / "submission"
    work.mkdir(parents=True, exist_ok=True)
    raw = Path(raw_test) if raw_test is not None else root / "raw/test_ver2.csv"
    template = Path(sample_submission) if sample_submission is not None else root / "raw/sample_submission.csv"
    if not template.is_file():
        raise FileNotFoundError(f"Official sample_submission template not found: {template}")
    # A run-specific fresh checkpoint prevents reuse of an unrelated/stale test.
    test = work / "test.parquet"
    build_interim_checkpoint(raw, test, chunksize=int(config["runtime"]["batch_customer_months"]),
                             force_rebuild=True, show_progress=True)
    prepared = materialize_acquisition_only_test_input(test, history, work / "prepared_june.parquet", config["runtime"], work)
    output = root / "artifacts/submissions" / run_id / "submission.csv"
    inputs = pd.read_parquet(prepared)
    run_ranked_candidate_submission(
        inputs, template, output, score_batch=models.candidate_batch_scorer(run),
        batch_customers=int(config["runtime"]["batch_customer_months"]),
        top_k=7, product_order=PRODUCT_COLUMNS,
    )
    write_json(output.parent / "submission_manifest.json", {
        "source_run_id": run_id, "approach": result["approach"], "created_at": now(),
        "run": str(run), "history": str(history), "history_sha256": manifest["checkpoint_sha256"],
        "raw_test": str(raw), "raw_test_sha256": sha256(raw),
        "sample_submission": str(template), "sample_submission_sha256": sha256(template),
        "feature_contract_sha256": sha256(run / "feature_contract.json"),
        "prepared_input": str(prepared), "submission_sha256": sha256(output), "top_k": 7,
        "tie_break": "canonical_product_order", "unknown_customer_ownership": "all_unowned",
        "profile_residuals": "pinned_canonical_history_medians",
        "income_policy": "shared_customer_history_geo_median_v1_without_clipping",
    })
    return output
