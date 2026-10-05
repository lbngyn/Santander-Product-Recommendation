"""Export existing runs without modifying or retraining the original output."""
from pathlib import Path
import shutil

from src.utils.run_files import read_json, write_json
from src.tracking.run_bundle import build_bundle, environment, safe_file, safe_id
from src.products import PRODUCT_COLUMNS


def export_legacy(run_dir, destination, *, project_id, experiment_name, kind, source_status="unknown"):
    run_dir, destination = Path(run_dir).resolve(), Path(destination).resolve()
    if destination == run_dir or destination.is_relative_to(run_dir):
        raise ValueError("Migration destination must be outside original run")
    if kind not in {"independent", "joint", "popularity"}:
        raise ValueError("kind must be independent/joint/popularity")
    if source_status not in {"unknown", "FINISHED", "FAILED", "KILLED"}:
        raise ValueError("Invalid source_status; only supply FINISHED when evidenced")
    original = read_json(run_dir / "lineage_manifest.json")
    run_id = safe_id(original.get("run_id", run_dir.name)); safe_id(project_id)
    staged = destination / "legacy_export_source"
    if staged.exists() or (destination / "bundle").exists():
        raise FileExistsError("Use a fresh destination for explicit legacy export")
    staged.mkdir(parents=True)
    for filename in ("config_resolved.yaml", "lineage_manifest.json", "validation_metrics.json", "training_metrics.json",
                     "per_product_metrics.json", "feature_contract.json", "evaluation_contract.json"):
        source = run_dir / filename
        if source.is_file():
            shutil.copy2(source, staged / filename)
    index = {}
    if kind == "independent":
        directories = {p.name: p for p in (run_dir / "models").iterdir() if p.is_dir()} if (run_dir / "models").is_dir() else {}
    elif kind == "joint":
        directories = {"joint_model": run_dir / "joint_model"}
    else:
        directories = {}
        ranking = run_dir / "popularity_ranking.parquet"
        if ranking.is_file():
            shutil.copy2(ranking, staged / ranking.name); index["popularity"] = ranking.name
        # Popularity's existing metrics_path can have a different filename.
        if (run_dir / "metrics.json").is_file() and not (staged / "validation_metrics.json").exists():
            write_json(staged / "validation_metrics.json", read_json(run_dir / "metrics.json"))
    for name, directory in directories.items():
        safe_id(name)
        if not (directory / "model.pkl").is_file() or not (directory / "artifact.json").is_file():
            continue
        relative = "joint_model" if kind == "joint" else "models/" + name
        target = safe_file(staged, relative); target.mkdir(parents=True)
        for filename in ("model.pkl", "artifact.json", "training_metrics.json", "joint_metadata.json"):
            if (directory / filename).is_file():
                shutil.copy2(directory / filename, target / filename)
        index[name] = relative
    write_json(staged / "model_index.json", index)
    ready = bool(index) and (kind != "independent" or set(index) == set(PRODUCT_COLUMNS))
    write_json(staged / "run_result.json", {"schema_version": 1, "project_id": project_id,
        "source_run_id": run_id, "experiment_name": experiment_name, "approach": kind,
        "status": source_status, "runtime": "unknown", "started_at": None, "ended_at": None,
        "inference_ready": ready, "migration": "legacy_export", "stages": {},
        "notes": "Missing measurements/temporal provenance are unknown; original files unchanged."})
    # Do not describe the migration host environment as the model training environment.
    write_json(staged / "environment.json", {"training_environment": None, "migration_environment": environment()})
    return build_bundle(staged, destination / "bundle")
