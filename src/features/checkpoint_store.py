"""Immutable canonical customer-month checkpoint infrastructure.

The store is deliberately independent from a particular model version.  A
version contains one or more named artifacts (currently ``train`` and
``test``), while the manifest is the sole resolver index.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import duckdb


KEY_COLUMNS: tuple[str, str] = ("ncodpers", "fecha_dato")
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ProcessingStep:
    """A processing-function contract used by :func:`run_processing_pipeline`."""

    name: str
    outputs: tuple[str, ...]
    required_inputs: tuple[str, ...]
    run: Callable[[dict[str, Path]], dict[str, Path]]
    validate: Callable[[dict[str, Path]], None] | None = None


class CheckpointStore:
    """Persist and resolve immutable full customer-month checkpoint versions."""

    def __init__(self, root: str | Path, *, runtime: Mapping[str, Any] | None = None) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / MANIFEST_NAME
        self.runtime = dict(runtime or {})

    def resolve(
        self,
        required_features: Sequence[str],
        *,
        required_artifacts: Sequence[str] = ("train", "test"),
    ) -> dict[str, Any] | None:
        """Return the newest retained, valid version satisfying requirements."""
        required = set(required_features)
        for version in reversed(self._manifest()["versions"]):
            if not required.issubset(set(version.get("features", ()) )):
                continue
            artifacts = version.get("artifacts", {})
            if not set(required_artifacts).issubset(artifacts):
                continue
            if self.validate_version(version, required_features=required_features):
                return version
        return None

    def publish(
        self,
        candidates: Mapping[str, str | Path],
        *,
        parent_version: int | None,
        processing: Sequence[Mapping[str, Any]],
        configuration: Mapping[str, Any] | None = None,
        feature_definition_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate candidates, atomically retain them, then append manifest."""
        if not candidates:
            raise ValueError("At least one candidate artifact is required.")
        candidate_paths = {name: Path(path) for name, path in candidates.items()}
        if missing := [str(path) for path in candidate_paths.values() if not path.is_file()]:
            raise FileNotFoundError(f"Checkpoint candidate artifact(s) missing: {missing}")

        manifest = self._manifest()
        number = max((int(v["version"]) for v in manifest["versions"]), default=0) + 1
        version_dir = self.root / f"v{number:06d}"
        if version_dir.exists():
            raise FileExistsError(f"Checkpoint version directory already exists: {version_dir}")
        self.root.mkdir(parents=True, exist_ok=True)
        version_dir.mkdir()
        retained: dict[str, Path] = {}
        try:
            # Copy rather than rename: a failed manifest append must leave the
            # caller's working candidate available for diagnosis/retry.
            for name, candidate in candidate_paths.items():
                target = version_dir / f"{name}.parquet"
                shutil.copy2(candidate, target)
                retained[name] = target
            artifact_metadata = {name: _artifact_metadata(path, self.runtime) for name, path in retained.items()}
            for name, metadata in artifact_metadata.items():
                _validate_artifact(retained[name], metadata["schema"], self.runtime)
            feature_sets = [set(metadata["schema"]) for metadata in artifact_metadata.values()]
            shared_features = sorted(set.intersection(*feature_sets)) if feature_sets else []
            parent = next((v for v in manifest["versions"] if v["version"] == parent_version), None)
            parent_features = set(parent.get("features", ())) if parent else set()
            executed_outputs = {
                feature
                for step in processing if step.get("decision") == "RUN"
                for feature in step.get("declared_outputs", ())
            }
            changes = {
                "added": sorted(set(shared_features) - parent_features),
                "recomputed": sorted(set(shared_features).intersection(parent_features, executed_outputs)),
                "reused": sorted(set(shared_features).intersection(parent_features) - executed_outputs),
            }
            version = {
                "version": number,
                "created_at": datetime.now(UTC).isoformat(),
                "parent_version": parent_version,
                "grain": list(KEY_COLUMNS),
                "artifacts": {
                    name: {"path": str(path.relative_to(self.root)), **artifact_metadata[name]}
                    for name, path in retained.items()
                },
                "features": shared_features,
                "feature_changes": changes,
                "processing": list(processing),
                "configuration": dict(configuration or {}),
                "code_revision": _code_revision(),
                "feature_definition_metadata": dict(feature_definition_metadata or {}),
                "validation": {"status": "passed"},
            }
            manifest["versions"].append(version)
            self._write_manifest(manifest)
            return version
        except Exception:
            shutil.rmtree(version_dir, ignore_errors=True)
            raise

    def cleanup(self, n_latest_cpts: int = 10) -> list[int]:
        """Delete old versions and their manifest entries together."""
        if n_latest_cpts < 1:
            raise ValueError("n_latest_cpts must be at least 1.")
        manifest = self._manifest()
        versions = manifest["versions"]
        stale, retained = versions[:-n_latest_cpts], versions[-n_latest_cpts:]
        for version in stale:
            directory = self.root / f"v{int(version['version']):06d}"
            if directory.exists():
                shutil.rmtree(directory)
        manifest["versions"] = retained
        self._write_manifest(manifest)
        return [int(version["version"]) for version in stale]

    def validate_version(self, version: Mapping[str, Any], *, required_features: Sequence[str] = ()) -> bool:
        """Return false for any invalid or no-longer-retained artifact."""
        if version.get("validation", {}).get("status") != "passed":
            return False
        if tuple(version.get("grain", ())) != KEY_COLUMNS:
            return False
        if not set(required_features).issubset(set(version.get("features", ()))):
            return False
        try:
            for artifact in version.get("artifacts", {}).values():
                path = self.root / artifact["path"]
                _validate_artifact(path, artifact["schema"], self.runtime)
        except (OSError, ValueError, duckdb.Error, KeyError):
            return False
        return True

    def _manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"format_version": 1, "versions": []}
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("versions"), list):
            raise ValueError(f"Invalid checkpoint manifest: {self.manifest_path}")
        return data

    def _write_manifest(self, manifest: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.manifest_path)


def run_processing_pipeline(
    working: Mapping[str, str | Path],
    steps: Sequence[ProcessingStep],
    *,
    force_process: bool,
    required_features: Sequence[str],
    force_features: Sequence[str] = (),
    force_functions: Sequence[str] = (),
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Run/skip declared feature groups and enforce their common contract."""
    current = {name: Path(path) for name, path in working.items()}
    plan = plan_processing_pipeline(steps, _shared_columns(current), force_process=force_process,
                                    force_features=force_features, force_functions=force_functions)
    history: list[dict[str, Any]] = []
    for step, summary in zip(steps, plan):
        columns = _shared_columns(current)
        missing_inputs = set(step.required_inputs).difference(columns)
        if missing_inputs and summary["decision"] == "RUN":
            raise ValueError(f"Processing step {step.name!r} is missing inputs: {sorted(missing_inputs)}")
        decision = summary["decision"]
        if decision == "RUN":
            current = {name: Path(path) for name, path in step.run(dict(current)).items()}
            after = _shared_columns(current)
            missing_after = sorted(set(step.outputs).difference(after))
            if missing_after:
                raise RuntimeError(f"Processing step {step.name!r} did not create outputs: {missing_after}")
            if step.validate is not None:
                step.validate(dict(current))
            summary["validation"] = "passed"
        else:
            summary["reason"] = "all declared outputs already exist and force_process=False"
        history.append(summary)
    final_columns = _shared_columns(current)
    if missing := set(required_features).difference(final_columns):
        raise RuntimeError(f"Pipeline did not satisfy required features: {sorted(missing)}")
    return current, history


def plan_processing_pipeline(
    steps: Sequence[ProcessingStep],
    existing_columns: Sequence[str] | set[str],
    *,
    force_process: bool = False,
    force_features: Sequence[str] = (),
    force_functions: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Plan ordered contracts without invoking functions or creating files.

    Input columns refreshed by an earlier RUN invalidate their consumers,
    transitively. Contracts must be ordered upstream before downstream. Shared
    outputs (such as age) do not create a dependency on a later producer.
    """
    if len({s.name for s in steps}) != len(steps):
        raise ValueError("Processing function names must be unique")
    forced = resolve_forced_steps({s.name: s.outputs for s in steps},
                                 force_features=force_features, force_functions=force_functions)
    columns = set(existing_columns)
    changed_by: dict[str, str] = {}
    plan = []
    for step in steps:
        missing = sorted(set(step.outputs) - columns)
        upstream = sorted({changed_by[c] for c in step.required_inputs if c in changed_by})
        explicit = force_process or step.name in forced
        run = bool(explicit or missing or upstream)
        reasons = []
        if force_process:
            reasons.append("global_force")
        elif explicit:
            reasons.append("selected")
        if missing:
            reasons.append("missing_outputs")
        if upstream:
            reasons.append("upstream_changed")
        plan.append({
            "step": step.name, "declared_outputs": list(step.outputs),
            "required_inputs": list(step.required_inputs),
            "existing_outputs": sorted(set(step.outputs) & columns),
            "missing_outputs": missing, "force_process": explicit or bool(upstream),
            "forced_features": sorted(set(step.outputs) & set(force_features)),
            "forced_function": step.name in force_functions,
            "upstream_functions": upstream, "decision": "RUN" if run else "SKIP",
            "reasons": reasons or ["outputs_exist"],
        })
        if run:
            changed_by.update({c: step.name for c in step.outputs})
            columns.update(step.outputs)
    return plan


def resolve_forced_steps(
    function_outputs: Mapping[str, Sequence[str]],
    *,
    force_features: Sequence[str] = (),
    force_functions: Sequence[str] = (),
) -> set[str]:
    """Map explicit feature/function selections to their declared owners.

    Recompute the whole output contract of each selected function. Fail on
    unknown selections before any processing. This helper selects the starting
    functions; the planner then propagates updated inputs to consumers.
    """
    if isinstance(force_features, str) or isinstance(force_functions, str):
        raise TypeError("force_features and force_functions must be lists/tuples, not strings")
    known_features = {feature for outputs in function_outputs.values() for feature in outputs}
    unknown_features = set(force_features).difference(known_features)
    unknown_functions = set(force_functions).difference(function_outputs)
    if unknown_features or unknown_functions:
        raise ValueError(
            f"Unknown force selections: features={sorted(unknown_features)}, "
            f"functions={sorted(unknown_functions)}. Available functions: {sorted(function_outputs)}"
        )
    return set(force_functions) | {
        name for name, outputs in function_outputs.items() if set(outputs).intersection(force_features)
    }


def _shared_columns(artifacts: Mapping[str, Path]) -> set[str]:
    if not artifacts:
        raise ValueError("Working checkpoint has no artifacts.")
    schemas = [_schema(path) for path in artifacts.values()]
    return set.intersection(*(set(schema) for schema in schemas))


def _artifact_metadata(path: Path, runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
    schema = _schema(path, runtime)
    con = _checkpoint_connection(runtime)
    try:
        rows, minimum, maximum = con.execute(
            "SELECT COUNT(*), MIN(CAST(fecha_dato AS DATE)), MAX(CAST(fecha_dato AS DATE)) FROM read_parquet(?)",
            [str(path)],
        ).fetchone()
    finally:
        con.close()
    return {"schema": schema, "row_count": int(rows), "date_coverage": {"min": str(minimum) if minimum else None, "max": str(maximum) if maximum else None}}


def _validate_artifact(path: Path, expected_schema: Mapping[str, str], runtime: Mapping[str, Any] | None = None) -> None:
    if not path.is_file():
        raise ValueError(f"Checkpoint artifact does not exist: {path}")
    schema = _schema(path, runtime)
    if schema != dict(expected_schema):
        raise ValueError(f"Checkpoint schema does not match manifest: {path}")
    if missing := set(KEY_COLUMNS).difference(schema):
        raise ValueError(f"Checkpoint is missing grain key columns: {sorted(missing)}")
    con = _checkpoint_connection(runtime)
    try:
        valid = con.execute(
            """SELECT COUNT(*) = COUNT(DISTINCT (ncodpers, fecha_dato))
                         AND COUNT(*) = COUNT(ncodpers)
                         AND COUNT(*) = COUNT(fecha_dato)
                 FROM read_parquet(?)""",
            [str(path)],
        ).fetchone()[0]
    finally:
        con.close()
    if not valid:
        raise ValueError(f"Checkpoint violates non-null unique customer-month grain: {path}")


def _schema(path: Path, runtime: Mapping[str, Any] | None = None) -> dict[str, str]:
    con = _checkpoint_connection(runtime)
    try:
        return {row[0]: row[1] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()}
    finally:
        con.close()


def _checkpoint_connection(runtime: Mapping[str, Any] | None = None) -> duckdb.DuckDBPyConnection:
    runtime = dict(runtime or {})
    settings = {key: runtime[key] for key in ('memory_limit', 'threads') if runtime.get(key) is not None}
    con = duckdb.connect(database=':memory:', config=settings)
    if runtime.get('temp_directory'):
        directory = Path(runtime['temp_directory']); directory.mkdir(parents=True, exist_ok=True)
        con.execute('SET temp_directory = ?', [str(directory.resolve())])
    return con


def _code_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
