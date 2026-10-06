"""Allowlisted, immutable inference/tracking bundles; never copy whole runs."""
from __future__ import annotations

import importlib.metadata
import re
import shutil
from pathlib import Path, PurePosixPath

from src.utils.run_files import now, read_json, sha256, write_json


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}", value) or value in {".", ".."}:
        raise ValueError(f"Unsafe project/run/comparison identifier: {value!r}")
    return value


def safe_file(root, relative):
    relative = PurePosixPath(str(relative).replace("\\", "/"))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or ":" in str(relative):
        raise ValueError("Unsafe relative bundle path")
    root = Path(root).resolve(); path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Bundle path escapes its root")
    return path


def test_template(run_id):
    return {"schema_version": 1, "source_run_id": run_id, "status": "pending", "test_score": None,
            "test_metric_name": None, "test_set_url": None, "metrics": {}, "notes": None, "evaluated_at": None}


def environment():
    import platform
    packages = {}
    for name in ("lightgbm", "numpy", "pandas", "scikit-learn", "pyarrow", "mlflow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def build_bundle(run_dir, destination):
    """Requires explicit run_result/model_index; source output remains untouched."""
    run_dir, destination = Path(run_dir).resolve(), Path(destination).resolve()
    if destination == run_dir or destination.is_relative_to(run_dir):
        raise ValueError("Bundle must be outside original run directory")
    result = read_json(run_dir / "run_result.json")
    safe_id(result["project_id"]); safe_id(result["source_run_id"])
    existing = destination / "manifest.json"
    if existing.is_file():
        verify_bundle(destination)
        if sha256(run_dir / "run_result.json") != read_json(existing)["files"]["run_result.json"]["sha256"]:
            raise ValueError("Bundle already finalized with another result")
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    selected = ["run_result.json", "config_resolved.yaml", "lineage_manifest.json", "model_index.json",
                "training_metrics.json", "validation_metrics.json", "per_product_metrics.json",
                "feature_contract.json", "dataset_manifest.json", "cohort_audit.json", "evaluation_contract.json", "environment.json"]
    index = read_json(run_dir / "model_index.json") if (run_dir / "model_index.json").is_file() else {}
    for relative in index.values():
        model_dir = safe_file(run_dir, relative)
        if model_dir.is_dir():
            for name in ("model.pkl", "artifact.json", "training_metrics.json", "training_history.json", "joint_metadata.json"):
                selected.append((model_dir / name).relative_to(run_dir).as_posix())
        elif model_dir.is_file():
            selected.append(model_dir.relative_to(run_dir).as_posix())  # popularity artifact
    for name in result.get("additional_inference_files", []):
        selected.append(str(name))
    files = {}
    for name in dict.fromkeys(selected):
        source = safe_file(run_dir, name)
        if not source.is_file():
            continue
        target = safe_file(destination, name); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[name] = {"size": target.stat().st_size, "sha256": sha256(target)}
    for required in ("run_result.json", "config_resolved.yaml", "lineage_manifest.json"):
        if required not in files:
            raise FileNotFoundError(f"Run lacks required tracking artifact: {required}")
    template = write_json(destination / "test_results.template.json", test_template(result["source_run_id"]))
    files[template.name] = {"size": template.stat().st_size, "sha256": sha256(template)}
    manifest = {"schema_version": 1, "project_id": result["project_id"], "source_run_id": result["source_run_id"], "files": files}
    write_json(destination / "manifest.json", manifest)
    write_json(destination / "READY.json", {"manifest_sha256": sha256(destination / "manifest.json")})
    verify_bundle(destination)
    return destination


def verify_bundle(root, cloud=False):
    root = Path(root)
    marker = read_json(root / ("COMMITTED.json" if cloud else "READY.json"))
    if marker["manifest_sha256"] != sha256(root / "manifest.json"):
        raise ValueError("Bundle manifest digest mismatch")
    manifest = read_json(root / "manifest.json")
    safe_id(manifest["project_id"]); safe_id(manifest["source_run_id"])
    for relative, expected in manifest["files"].items():
        path = safe_file(root, relative)
        if not path.is_file() or path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Incomplete/changed bundle artifact: {relative}")
    result = read_json(root / "run_result.json")
    if (result["project_id"], result["source_run_id"]) != (manifest["project_id"], manifest["source_run_id"]):
        raise ValueError("Run identity differs from manifest")
    if result.get("inference_ready"):
        index = read_json(root / "model_index.json")
        if not index:
            raise ValueError("Inference-ready bundle has no model index")
        for relative in index.values():
            path = safe_file(root, relative)
            required = [path / "model.pkl", path / "artifact.json"] if path.is_dir() else [path]
            if any(p.relative_to(root.resolve()).as_posix() not in manifest["files"] for p in required):
                raise ValueError("Inference artifact absent from manifest")
    return manifest
