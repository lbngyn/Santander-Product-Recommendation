"""Pre-host GCS sync and offline MLflow import; no direct MLflow SQL writes."""
from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from src.data.gcs_storage import get_gcs_client
from src.utils.run_files import now, read_json, write_json
from src.tracking.gcs_runs import inventory, pull_annotations, pull_bundle, validate_test_results
from src.tracking.run_bundle import safe_file, safe_id, verify_bundle


@contextmanager
def host_lock(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root / "sync.lock").open("a+b") as handle:
        handle.seek(0); handle.write(b"0"); handle.flush(); handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Ledger:
    def __init__(self, path):
        self.con = sqlite3.connect(path)
        self.con.execute("CREATE TABLE IF NOT EXISTS imports (identity TEXT PRIMARY KEY, state TEXT NOT NULL)")

    def get(self, identity):
        row = self.con.execute("SELECT state FROM imports WHERE identity=?", [identity]).fetchone()
        return json.loads(row[0]) if row else {}

    def put(self, identity, state):
        self.con.execute("INSERT OR REPLACE INTO imports VALUES (?,?)", [identity, json.dumps(state)])
        self.con.commit()

    def close(self):
        self.con.close()


def backend_uri(host_root):
    return "sqlite:///" + (Path(host_root).resolve() / "mlflow.db").as_posix()


def _milliseconds(timestamp):
    return int(datetime.fromisoformat(timestamp).timestamp() * 1000) if timestamp else None


def _numeric(values):
    return {k: float(v) for k, v in values.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}


def _metrics(client, run_id, values, timestamp, step=0):
    for key, value in _numeric(values).items():
        if not any(m.step == step and m.timestamp == timestamp and m.value == value for m in client.get_metric_history(run_id, key)):
            client.log_metric(run_id, key, value, timestamp=timestamp, step=step)


def _find_or_create(client, experiment, tags, start_time):
    query = " AND ".join(f"tags.`{key}` = '{tags[key]}'" for key in ("project_id", "source_run_id", "import_scope"))
    matches = client.search_runs([experiment], filter_string=query, max_results=3)
    if len(matches) > 1:
        raise ValueError("Duplicate imported source identity; reconcile before continuing")
    if matches:
        run = matches[0]
        if run.data.tags.get("bundle_digest") != tags["bundle_digest"]:
            raise ValueError("Source run ID already imported with a different digest")
        return run.info.run_id
    return client.create_run(experiment, start_time=start_time, tags=tags).info.run_id


def import_bundle(root, host_root, ledger):
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException
    root, host_root = Path(root).resolve(), Path(host_root).resolve()
    manifest = verify_bundle(root, cloud=True); marker = read_json(root / "COMMITTED.json")
    result = read_json(root / "run_result.json")
    identity = manifest["project_id"] + "/" + manifest["source_run_id"]
    state = ledger.get(identity); digest = marker["manifest_sha256"]
    if state and state["digest"] != digest:
        raise ValueError("Cloud content changed for an already imported source run")
    state.setdefault("digest", digest); state.setdefault("children", {}); state.setdefault("annotation_revision", 0)
    client = MlflowClient(tracking_uri=backend_uri(host_root))
    experiment = client.get_experiment_by_name(result["experiment_name"])
    if experiment is None:
        try:
            eid = client.create_experiment(result["experiment_name"], artifact_location=(host_root / "artifacts").as_uri())
        except MlflowException:
            experiment = client.get_experiment_by_name(result["experiment_name"])
            if experiment is None:
                raise
            eid = experiment.experiment_id
    else:
        eid = experiment.experiment_id
    tags = {"project_id": manifest["project_id"], "source_run_id": manifest["source_run_id"],
            "bundle_digest": digest, "import_scope": "parent", "runtime": result.get("runtime", "unknown"),
            "source_status": result.get("status", "unknown"), "approach": result.get("approach", "legacy"),
            "comparison_id": result.get("comparison_id", "legacy"), "sync_status": "importing"}
    run_id = _find_or_create(client, eid, tags, _milliseconds(result.get("started_at")))
    state["parent_run_id"] = run_id; state["status"] = "importing"
    ledger.put(identity, state)
    timestamp = _milliseconds(result.get("ended_at")) or client.get_run(run_id).info.start_time
    try:
        config = __import__("yaml").safe_load((root / "config_resolved.yaml").read_text(encoding="utf-8"))
        def flatten(value, prefix=""):
            output = {}
            for key, item in value.items():
                name = prefix + str(key)
                if isinstance(item, dict):
                    output.update(flatten(item, name + "."))
                elif item is not None and isinstance(item, (str, int, float, bool)):
                    output[name] = str(item)
            return output
        existing = client.get_run(run_id).data.params
        for key, value in flatten(config).items():
            if key in existing and existing[key] != value:
                raise ValueError(f"Imported immutable param differs: {key}")
            if key not in existing:
                client.log_param(run_id, key, value)
        for filename, prefix in (("training_metrics.json", "train."), ("validation_metrics.json", "validation.")):
            if (root / filename).is_file():
                _metrics(client, run_id, {prefix + k: v for k, v in read_json(root / filename).items()}, timestamp)
        index = read_json(root / "model_index.json") if (root / "model_index.json").is_file() else {}
        diagnostics = {d["product"]: d for d in read_json(root / "per_product_metrics.json")} if (root / "per_product_metrics.json").is_file() else {}
        for name, relative in index.items():
            safe_id(name)
            directory = safe_file(root, relative)
            metric_path = directory / "training_metrics.json"
            if not metric_path.is_file():
                continue
            child_tags = {**tags, "import_scope": name, "mlflow.parentRunId": run_id, "product": name}
            child = _find_or_create(client, eid, child_tags, _milliseconds(result.get("started_at")))
            state["children"][name] = child; ledger.put(identity, state)
            _metrics(client, child, {"train." + k: v for k, v in read_json(metric_path).items()}, timestamp)
            _metrics(client, child, {"validation." + k: v for k, v in diagnostics.get(name, {}).items()}, timestamp)
            client.set_tag(child, "sync_status", "complete")
            status = result.get("status", "unknown")
            client.set_terminated(child, status if status in {"FINISHED", "FAILED", "KILLED"} else "FINISHED", end_time=_milliseconds(result.get("ended_at")))
        # Copy once into a durable MLflow artifact directory; staging is disposable.
        client.log_artifacts(run_id, str(root))
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
        client.set_tag(run_id, "test.status", client.get_run(run_id).data.tags.get("test.status", "pending"))
        client.set_tag(run_id, "sync_status", "complete")
        status = result.get("status", "unknown")
        client.set_terminated(run_id, status if status in {"FINISHED", "FAILED", "KILLED"} else "FINISHED", end_time=_milliseconds(result.get("ended_at")))
        state["status"] = "complete"; ledger.put(identity, state)
        return state
    except Exception:
        client.set_tag(run_id, "sync_status", "failed")
        state["status"] = "failed"; ledger.put(identity, state)
        raise


def _import_annotation(client, run_id, payload, host_root):
    revision = int(payload["revision"])
    timestamp = _milliseconds(payload.get("evaluated_at")) or revision
    _metrics(client, run_id, {"test." + k: v for k, v in validate_test_results(payload).items()}, timestamp, revision)
    for key, value in {"test.status": payload["status"], "test_set_url": payload.get("test_set_url") or "",
                       "test.annotation_revision": revision, "test.notes": payload.get("notes") or ""}.items():
        client.set_tag(run_id, key, str(value))
    path = Path(host_root) / "annotations" / payload["source_run_id"] / f"revision_{revision:06d}.json"
    write_json(path, payload)
    client.log_artifact(run_id, str(path), artifact_path=f"annotations/test/revision_{revision:06d}")


def sync(config, *, dry_run=False):
    """Strict: any committed-run error propagates before server startup."""
    host = Path(config["host_root"]).expanduser().resolve()
    bucket, prefix, project = config["bucket"], config.get("prefix", "model-runs"), safe_id(config["project_id"])
    client = get_gcs_client(config.get("gcp_project"))
    with host_lock(host):
        entries = inventory(bucket, prefix, project, client)
        if dry_run:
            return {"status": "planned", "inventory": entries, "imports": False, "host_lock_created": True}
        ledger = Ledger(host / "sync_ledger.db")
        report = {"started_at": now(), "imported": [], "skipped": [], "annotations": 0}
        try:
            from mlflow import MlflowClient
            ml = MlflowClient(tracking_uri=backend_uri(host))
            for entry in entries:
                identity = project + "/" + entry["source_run_id"]
                state = ledger.get(identity)
                if state.get("status") == "complete":
                    # Still detect cloud conflicts and missing host artifacts.
                    marker_blob = client.bucket(bucket).blob(entry["prefix"] + "/COMMITTED.json", generation=entry["marker_generation"])
                    marker = json.loads(marker_blob.download_as_text())
                    if marker["manifest_sha256"] != state["digest"]:
                        raise ValueError("Committed run digest changed")
                    run = ml.get_run(state["parent_run_id"])
                    from urllib.parse import unquote, urlparse
                    parsed = urlparse(run.info.artifact_uri)
                    local = Path(unquote(parsed.path).lstrip("/") if os.name == "nt" else unquote(parsed.path))
                    if parsed.scheme != "file" or not (local / "run_result.json").is_file():
                        raise ValueError("Host artifacts missing; resolve/reimport before starting server")
                    verify_bundle(local, cloud=True)
                    report["skipped"].append(entry["source_run_id"])
                else:
                    stage = host / "staging" / project / entry["source_run_id"]
                    pull_bundle(bucket, entry, stage, client)
                    if read_json(stage / "manifest.json")["project_id"] != project:
                        raise ValueError("Cloud project identity mismatch")
                    state = import_bundle(stage, host, ledger)
                    report["imported"].append(entry["source_run_id"])
                    stage = stage.resolve(); staging_root = (host / "staging").resolve()
                    if not stage.is_relative_to(staging_root) or stage == staging_root:
                        raise ValueError("Unsafe staging cleanup path")
                    shutil.rmtree(stage)
                for payload in pull_annotations(bucket, prefix, project, entry["source_run_id"], state.get("annotation_revision", 0), client):
                    _import_annotation(ml, state["parent_run_id"], payload, host)
                    state["annotation_revision"] = payload["revision"]; ledger.put(identity, state)
                    report["annotations"] += 1
            report["completed_at"] = now(); report["status"] = "complete"
            write_json(host / "last_sync.json", report)
            return report
        finally:
            ledger.close()
