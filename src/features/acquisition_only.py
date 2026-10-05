"""Approved canonical processed snapshots and common, immutable split packs."""
from pathlib import Path

from src.features.acquisition_only_contract import (
    CATEGORICAL, COUNT_CANDIDATES, DATES, FEATURES, KEYS, LABELS, NUMERIC,
    PROFILES, parquet, quote,
)
from src.utils.modeling_runtime import connection
from src.utils.run_files import now, read_json, sha256, write_json
from src.features.customer_history import acquisition_label_sql
from src.products import PRODUCT_COLUMNS, PRODUCT_ID_MAP


def resolve_checkpoint(data_root, version=None):
    root = Path(data_root).resolve() / "processed/canonical_customer_month"
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("No main canonical checkpoint. Run notebooks/Data Pipeline.ipynb first.")
    for record in reversed(read_json(manifest)["versions"]):
        if version is not None and int(record["version"]) != int(version):
            continue
        if record.get("validation", {}).get("status") != "passed" or "train" not in record.get("artifacts", {}):
            continue
        path = (root / record["artifacts"]["train"]["path"].replace("\\", "/")).resolve()
        if not path.is_relative_to(root) or path.name != "train.parquet":
            raise ValueError("Unsafe canonical train path in manifest")
        if path.is_file():
            return path, record
    raise FileNotFoundError(f"No valid retained canonical train version: {version}")


def dataset_path(config, data_root):
    from src.tracking.run_bundle import safe_id
    return Path(data_root).resolve() / "processed/modeling/acquisition_only_baseline" / safe_id(config["pipeline"]["comparison_id"])


def _check_config(config, manifest):
    for key in ("comparison_id", "validation_start", "validation_end"):
        if manifest[key] != (config["pipeline"][key] if key == "comparison_id" else config["split"][key]):
            raise ValueError(f"Prepared dataset differs in {key}; choose a new comparison_id")
    if config["data"].get("checkpoint_version") is not None and int(config["data"]["checkpoint_version"]) != manifest["checkpoint_version"]:
        raise ValueError("Prepared data pins another checkpoint; choose a new comparison_id")


def preview(config, data_root, work):
    output = dataset_path(config, data_root)
    existing = output / "dataset_manifest.json"
    if existing.is_file():
        manifest = read_json(existing); _check_config(config, manifest)
        return {"action": "reuse_pinned_dataset", "dataset": str(output), "manifest": manifest}
    source, version = resolve_checkpoint(data_root, config["data"].get("checkpoint_version"))
    with connection(config["runtime"], work) as con:
        schema = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {parquet(source)}").fetchall()}
    missing = sorted(set([*KEYS, *PROFILES, *PRODUCT_COLUMNS]) - schema)
    if missing:
        raise ValueError(f"Canonical snapshot lacks original information: {missing}")
    return {"action": "prepare", "checkpoint_version": version["version"], "checkpoint": str(source),
            "dataset": str(output), "feature_names": FEATURES,
            "acquisition_count_column": next((c for c in COUNT_CANDIDATES if c in schema), None)}


def prepare(config, data_root, work):
    plan = preview(config, data_root, work); output = Path(plan["dataset"])
    if plan["action"] == "reuse_pinned_dataset":
        for name, checksum in plan["manifest"]["files"].items():
            if not (output / name).is_file() or sha256(output / name) != checksum:
                raise ValueError(f"Prepared dataset changed: {name}; use a new comparison_id")
        return output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Incomplete prepared dataset exists; use a new comparison_id or explicitly resolve it")
    output.mkdir(parents=True, exist_ok=True)
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    timeline = work / "baseline_timeline.parquet"
    start, end = config["split"]["validation_start"], config["split"]["validation_end"]
    from datetime import date
    date.fromisoformat(start); date.fromisoformat(end)
    if start >= end:
        raise ValueError("validation_start must precede validation_end")
    source = Path(plan["checkpoint"])
    with connection(config["runtime"], work) as con:
        bad_keys = con.execute(f"SELECT COUNT(*) != COUNT(DISTINCT (ncodpers, fecha_dato)) OR COUNT(*) != COUNT(ncodpers) OR COUNT(*) != COUNT(fecha_dato) FROM {parquet(source)}").fetchone()[0]
        if bad_keys:
            raise ValueError("Canonical snapshot violates unique non-null customer-month grain")
        invalid_states = " OR ".join(f"TRY_CAST({quote(p)} AS DOUBLE) NOT IN (0,1) OR ({quote(p)} IS NOT NULL AND TRY_CAST({quote(p)} AS DOUBLE) IS NULL)" for p in PRODUCT_COLUMNS)
        if con.execute(f"SELECT COUNT(*) FROM {parquet(source)} WHERE {invalid_states}").fetchone()[0]:
            raise ValueError("Canonical product states must be binary or missing")
        schema = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {parquet(source)}").fetchall()}
        count_column = plan["acquisition_count_column"]
        if count_column:
            numeric_count = f"TRY_CAST({quote(count_column)} AS DOUBLE)"
            if con.execute(f"SELECT COUNT(*) FROM {parquet(source)} WHERE CAST(fecha_dato AS DATE) < DATE '{end}' AND ({numeric_count} IS NULL OR {numeric_count} < 0 OR {numeric_count} != FLOOR({numeric_count}))").fetchone()[0]:
                raise ValueError("Stored acquisition count must be a nonnegative integer")
        profiles = []
        for name in PROFILES:
            if name in NUMERIC:
                expr = f"TRY_CAST({quote(name)} AS FLOAT)"
            elif name in DATES:
                expr = f"CAST(date_diff('day', DATE '1970-01-01', TRY_CAST({quote(name)} AS DATE)) AS FLOAT)"
            else:
                expr = f"COALESCE(CAST({quote(name)} AS VARCHAR), '__MISSING__')"
            profiles.append(f"{expr} AS {quote(name)}")
        states = [f"COALESCE(TRY_CAST({quote(p)} AS TINYINT), 0) AS {quote(p)}" for p in PRODUCT_COLUMNS]
        stored_labels = set(LABELS).issubset(schema)
        extras = ([f"{quote(c)} AS {quote('stored_' + c)}" for c in LABELS] if stored_labels else [])
        if count_column:
            extras.append(f"TRY_CAST({quote(count_column)} AS INTEGER) AS stored_count")
        projection = ", ".join([*KEYS, *profiles, *states, *extras])
        previous = ", ".join(f"LAG({quote(p)}) OVER customer_time AS {quote('prev_' + p)}" for p in PRODUCT_COLUMNS)
        labels = ", ".join(
            (f"CAST({quote('stored_' + c)} AS TINYINT)" if stored_labels else acquisition_label_sql(p, previous_prefix="prev_")) + f" AS {quote(c)}"
            for p, c in zip(PRODUCT_COLUMNS, LABELS)
        )
        label_sum = " + ".join(quote(c) for c in LABELS)
        count = "stored_count" if count_column else f"({label_sum})"
        history_query = f"""WITH states AS (SELECT {projection} FROM {parquet(source)}), history AS (
            SELECT *, {previous}, date_diff('month', LAG(CAST(fecha_dato AS DATE)) OVER customer_time, CAST(fecha_dato AS DATE)) AS record_gap_months
            FROM states WINDOW customer_time AS (PARTITION BY ncodpers ORDER BY fecha_dato)
        )"""
        if stored_labels:
            disagreements = " OR ".join(
                f"TRY_CAST({quote('stored_' + c)} AS DOUBLE) IS NULL OR TRY_CAST({quote('stored_' + c)} AS DOUBLE) != ({acquisition_label_sql(p, previous_prefix='prev_')})"
                for p, c in zip(PRODUCT_COLUMNS, LABELS)
            )
            if con.execute(f"{history_query} SELECT COUNT(*) FROM history WHERE CAST(fecha_dato AS DATE) < DATE '{end}' AND ({disagreements})").fetchone()[0]:
                raise ValueError("Canonical acquisition labels disagree with nearest previous observed record")
        query = f"""{history_query}, events AS (SELECT *, {labels} FROM history)
        SELECT {', '.join(quote(c) for c in [*KEYS, *FEATURES, *LABELS])},
               record_gap_months, {count} AS n_acquisitions
        FROM events WHERE CAST(fecha_dato AS DATE) < DATE '{end}'"""
        _copy(con, query, timeline)
        invalid = " OR ".join(f"{quote(c)} IS NULL OR {quote(c)} NOT IN (0,1)" for c in LABELS)
        count_bad = con.execute(f"SELECT COUNT(*) FROM {parquet(timeline)} WHERE {invalid} OR n_acquisitions IS NULL OR n_acquisitions != ({label_sum}) OR (record_gap_months IS NULL AND n_acquisitions != 0)").fetchone()[0]
        if count_bad:
            raise ValueError("Stored acquisition count/labels disagree or are invalid")
        eligible = " OR ".join(f"{quote('prev_' + p)} = 0" for p in PRODUCT_COLUMNS)
        base = f"record_gap_months IS NOT NULL AND ({eligible})"
        filters = {"train": f"{base} AND CAST(fecha_dato AS DATE) < DATE '{start}' AND n_acquisitions >= 1",
                   "validation": f"{base} AND CAST(fecha_dato AS DATE) >= DATE '{start}' AND CAST(fecha_dato AS DATE) < DATE '{end}'"}
        audit = {}
        for split, condition in filters.items():
            for kind, names in (("inputs", FEATURES), ("targets", LABELS)):
                _copy(con, f"SELECT {', '.join(quote(c) for c in [*KEYS, *names])} FROM {parquet(timeline)} WHERE {condition} ORDER BY ncodpers, fecha_dato", output / f"{split}_{kind}.parquet")
            pairs = " + ".join(f"CAST({quote('prev_' + p)} = 0 AS INTEGER)" for p in PRODUCT_COLUMNS)
            row = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT ncodpers), COALESCE(SUM({pairs}),0), COUNT(*) FILTER (WHERE n_acquisitions > 0) FROM {parquet(timeline)} WHERE {condition}").fetchone()
            if row[0] == 0:
                raise ValueError(f"Empty {split} population")
            products = []
            for p in PRODUCT_COLUMNS:
                pos, neg = con.execute(f"SELECT COUNT(*) FILTER (WHERE {quote('acq_' + p)}=1), COUNT(*) FILTER (WHERE {quote('acq_' + p)}=0) FROM {parquet(timeline)} WHERE {condition} AND {quote('prev_' + p)}=0").fetchone()
                products.append({"product": p, "positives": pos, "negatives": neg})
            audit[split] = dict(zip(("customer_months", "customers", "candidate_rows", "acquisition_positive_customer_months"), row)) | {"per_product": products}
        vocab = {}
        for name in CATEGORICAL:
            vocab[name] = [r[0] for r in con.execute(f"SELECT DISTINCT {quote(name)} FROM {parquet(timeline)} WHERE CAST(fecha_dato AS DATE) < DATE '{start}' ORDER BY 1").fetchall()]
        contract = {"schema_version": 1, "feature_names": FEATURES, "category_values": vocab,
                    "date_encoding": "days_since_1970-01-01", "product_id_map": PRODUCT_ID_MAP,
                    "income_policy": "accepted_canonical_static_income", "profile_policy": "preserve_canonical_preprocessing",
                    "transition_policy": "nearest_previous_observed_record_including_gaps"}
        write_json(output / "feature_contract.json", contract)
        write_json(output / "cohort_audit.json", audit)
    timeline.unlink()
    files = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output / "dataset_manifest.json", {"schema_version": 1, "comparison_id": config["pipeline"]["comparison_id"],
        "checkpoint_version": plan["checkpoint_version"], "checkpoint_path": str(source), "checkpoint_sha256": sha256(source),
        "validation_start": start, "validation_end": end, "acquisition_count_column": count_column,
        "labels_source": "canonical_acq_columns" if stored_labels else "nearest_record_transitions",
        "created_at": now(), "files": files, "audit": audit, "feature_contract": contract})
    return output


def _copy(con, query, destination):
    destination = Path(destination); temporary = destination.with_name(destination.name + ".tmp")
    escaped = str(temporary).replace("'", "''")
    con.execute(f"COPY ({query}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    temporary.replace(destination)
