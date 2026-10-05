"""GCS publisher/puller with immutable objects and a commit-last protocol."""
from __future__ import annotations

import json
import math
import re
import tempfile
from pathlib import Path

from src.data.gcs_storage import get_gcs_client, normalize_prefix
from src.utils.run_files import read_json, sha256, write_json
from src.tracking.run_bundle import safe_file, safe_id, verify_bundle


def _upload_once(bucket, name, path):
    from google.api_core.exceptions import PreconditionFailed
    blob = bucket.blob(name); digest = sha256(path)
    blob.metadata = {"sha256": digest}
    try:
        blob.upload_from_filename(str(path), if_generation_match=0, checksum="crc32c")
    except PreconditionFailed:
        blob.reload()
        if (blob.metadata or {}).get("sha256") != digest or int(blob.size) != Path(path).stat().st_size:
            raise ValueError(f"Immutable GCS object conflict: gs://{bucket.name}/{name}")
    return int(blob.generation)


def publish_bundle(root, bucket_name, prefix="model-runs", client=None):
    root = Path(root).resolve(); manifest = verify_bundle(root)
    bucket = (client or get_gcs_client()).bucket(bucket_name)
    remote = "/".join(p for p in (normalize_prefix(prefix), "runs", manifest["project_id"], manifest["source_run_id"]) if p)
    cloud = {**manifest, "files": {}}
    for relative, metadata in manifest["files"].items():
        generation = _upload_once(bucket, remote + "/" + relative, safe_file(root, relative))
        cloud["files"][relative] = {**metadata, "generation": generation}
    with tempfile.TemporaryDirectory() as temporary:
        manifest_path = write_json(Path(temporary) / "manifest.json", cloud)
        generation = _upload_once(bucket, remote + "/manifest.json", manifest_path)
        marker = write_json(Path(temporary) / "COMMITTED.json", {"manifest_sha256": sha256(manifest_path), "manifest_generation": generation})
        _upload_once(bucket, remote + "/COMMITTED.json", marker)
    return {"status": "published", "gcs_uri": f"gs://{bucket_name}/{remote}", "source_run_id": manifest["source_run_id"]}


def inventory(bucket_name, prefix, project_id, client=None):
    client = client or get_gcs_client(); safe_id(project_id)
    base = "/".join(p for p in (normalize_prefix(prefix), "runs", project_id) if p) + "/"
    result = []
    for blob in client.list_blobs(bucket_name, prefix=base):
        relative = blob.name[len(base):].split("/")
        if len(relative) == 2 and relative[1] == "COMMITTED.json":
            result.append({"source_run_id": safe_id(relative[0]), "prefix": blob.name.rsplit("/", 1)[0], "marker_generation": int(blob.generation)})
    return sorted(result, key=lambda x: x["source_run_id"])


def pull_bundle(bucket_name, entry, destination, client=None):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    bucket = (client or get_gcs_client()).bucket(bucket_name); prefix = entry["prefix"]
    bucket.blob(prefix + "/COMMITTED.json", generation=entry["marker_generation"]).download_to_filename(str(destination / "COMMITTED.json"), checksum="crc32c")
    marker = read_json(destination / "COMMITTED.json")
    bucket.blob(prefix + "/manifest.json", generation=marker["manifest_generation"]).download_to_filename(str(destination / "manifest.json"), checksum="crc32c")
    if sha256(destination / "manifest.json") != marker["manifest_sha256"]:
        raise ValueError("GCS manifest checksum mismatch")
    manifest = read_json(destination / "manifest.json")
    if manifest["source_run_id"] != entry["source_run_id"]:
        raise ValueError("GCS run identity differs from prefix")
    for relative, metadata in manifest["files"].items():
        path = safe_file(destination, relative); path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size == metadata["size"] and sha256(path) == metadata["sha256"]:
            continue
        temporary = path.with_name(path.name + ".part")
        bucket.blob(prefix + "/" + relative, generation=metadata["generation"]).download_to_filename(str(temporary), checksum="crc32c")
        if temporary.stat().st_size != metadata["size"] or sha256(temporary) != metadata["sha256"]:
            raise ValueError(f"Downloaded artifact mismatch: {relative}")
        temporary.replace(path)
    verify_bundle(destination, cloud=True)
    return destination


def validate_test_results(payload):
    safe_id(payload["source_run_id"])
    if payload.get("status") not in {"pending", "evaluated", "withdrawn"}:
        raise ValueError("Test status must be pending/evaluated/withdrawn")
    metrics = dict(payload.get("metrics", {}))
    score = payload.get("test_score")
    if score is not None:
        if not payload.get("test_metric_name"):
            raise ValueError("test_metric_name is required with test_score")
        key = payload["test_metric_name"]
        if key in metrics and metrics[key] != score:
            raise ValueError("Conflicting test metric values")
        metrics[key] = score
    for key, value in metrics.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid test metric: {key}")
    if payload.get("status") == "evaluated" and not metrics:
        raise ValueError("Evaluated test annotation needs metrics")
    if payload.get("status") != "evaluated" and metrics:
        raise ValueError("Only evaluated annotations may contain metrics")
    url = payload.get("test_set_url")
    if url is not None and (not isinstance(url, str) or not url.startswith(("https://", "http://", "gs://"))):
        raise ValueError("test_set_url must be http(s):// or gs://")
    if payload.get("evaluated_at") is not None:
        from datetime import datetime
        value = payload["evaluated_at"]
        if not isinstance(value, str):
            raise ValueError("evaluated_at must be a timezone-aware ISO timestamp or null")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("evaluated_at requires a timezone")
    return metrics


def publish_test_results(path, bucket_name, prefix, project_id, client=None):
    """Compare-and-swap latest pointer; a concurrent edit is a visible conflict."""
    from google.api_core.exceptions import NotFound
    payload = read_json(path); validate_test_results(payload); safe_id(project_id)
    bucket = (client or get_gcs_client()).bucket(bucket_name)
    base = "/".join(p for p in (normalize_prefix(prefix), "annotations", project_id, payload["source_run_id"], "test") if p)
    pointer = bucket.blob(base + "/latest.json")
    try:
        pointer.reload(); old_generation = int(pointer.generation)
        previous = json.loads(pointer.download_as_text(if_generation_match=old_generation))
    except NotFound:
        old_generation, previous = 0, {"revision": 0}
    # Idempotent replay of the latest identical annotation.
    canonical = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    import hashlib
    digest = hashlib.sha256(canonical).hexdigest()
    if previous.get("payload_sha256") == digest:
        return previous
    revision = int(previous["revision"]) + 1
    with tempfile.TemporaryDirectory() as temporary:
        full = {**payload, "revision": revision, "previous_revision": previous["revision"], "payload_sha256": digest}
        file = write_json(Path(temporary) / "annotation.json", full)
        name = f"{base}/revision_{revision:06d}.json"
        generation = _upload_once(bucket, name, file)
        latest = {"revision": revision, "object": name, "generation": generation, "sha256": sha256(file), "payload_sha256": digest}
        pointer.upload_from_string(json.dumps(latest), content_type="application/json", if_generation_match=old_generation, checksum="crc32c")
    return latest


def pull_annotations(bucket_name, prefix, project_id, run_id, after=0, client=None):
    from google.api_core.exceptions import NotFound
    bucket = (client or get_gcs_client()).bucket(bucket_name)
    base = "/".join(p for p in (normalize_prefix(prefix), "annotations", safe_id(project_id), safe_id(run_id), "test") if p)
    pointer = bucket.blob(base + "/latest.json")
    try:
        pointer.reload(); latest = json.loads(pointer.download_as_text(if_generation_match=int(pointer.generation)))
    except NotFound:
        return []
    records = []
    # Ignore orphan revisions not reachable from a successfully updated pointer.
    for revision in range(int(after) + 1, int(latest["revision"]) + 1):
        blob = bucket.blob(f"{base}/revision_{revision:06d}.json"); blob.reload()
        if revision == int(latest["revision"]) and int(blob.generation) != int(latest["generation"]):
            raise ValueError("Annotation generation differs from latest pointer")
        raw = blob.download_as_bytes(if_generation_match=int(blob.generation), checksum="crc32c")
        import hashlib
        if revision == int(latest["revision"]) and hashlib.sha256(raw).hexdigest() != latest["sha256"]:
            raise ValueError("Latest annotation checksum mismatch")
        payload = json.loads(raw)
        if payload["source_run_id"] != run_id or payload["revision"] != revision:
            raise ValueError("Annotation revision identity mismatch")
        validate_test_results(payload); records.append(payload)
    return records
