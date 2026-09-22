"""Dependency-light reproducibility metadata for every pipeline run."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


def create_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"


def build_run_manifest(*, run_id: str, config: Mapping[str, Any], inputs: Mapping[str, str | Path], outputs: Mapping[str, str | Path]) -> dict[str, Any]:
    """Capture config, Git state, and inexpensive file fingerprints."""
    resolved_config = _canonical_json(config)
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git": _git_metadata(),
        "config": config,
        "config_sha256": _sha256_text(resolved_config),
        "inputs": {name: _file_metadata(Path(path)) for name, path in inputs.items()},
        "outputs": {name: _file_metadata(Path(path)) for name, path in outputs.items()},
    }


def write_run_manifest(manifest: Mapping[str, Any], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _git_metadata() -> dict[str, str | bool | None]:
    def read(*args: str) -> str | None:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    status = read("status", "--porcelain")
    return {"commit": read("rev-parse", "HEAD"), "branch": read("branch", "--show-current"), "dirty": bool(status) if status is not None else None}


def _file_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {"path": str(path), "exists": True, "bytes": stat.st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
