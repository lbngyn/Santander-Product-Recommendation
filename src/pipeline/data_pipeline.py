"""Notebook-facing canonical customer-month pipeline (historical train data).

Delegate versioning and RUN/SKIP decisions to the existing checkpoint store;
adapt existing preprocessing and feature-store functions into full snapshots.
Competition test has no observed product states and is not this dataset.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Sequence

import duckdb

from src.data.gcs_storage import download_object_if_missing, object_name
from src.features.checkpoint_store import (
    KEY_COLUMNS, CheckpointStore, ProcessingStep, plan_processing_pipeline, resolve_forced_steps, run_processing_pipeline,
)
from src.features.customer_history import HISTORY_FEATURE_NAMES, quote
from src.features.feature_store import acquisition_feature_names, materialize_customer_month_feature_group
from src.features.persona import PERSONA_FEATURES, RAW_PERSONA_COLUMNS
from src.features.rfm_tiering import (
    CANONICAL_RFM_BOUNDARIES, CANONICAL_RFM_METHODS, CANONICAL_RFM_TIER_FEATURE_NAMES,
    CANONICAL_RFM_TIER_INPUTS, materialize_canonical_rfm_tiers,
)
from src.ingestion.checkpoint import build_interim_checkpoint
from src.preprocessing.customer_profile import preprocess_customer_profile_baseline
from src.preprocessing.renta_monetary import RENTA_OUTPUTS, preprocess_renta_monetary
from src.products import PRODUCT_COLUMNS


PREPROCESSING_MARKER = "profile_preprocessed"
PROFILE_COLUMNS = (*RAW_PERSONA_COLUMNS, "conyuemp", "nomprov")
ENGINEERED_FEATURES = tuple(dict.fromkeys((
    *PERSONA_FEATURES, *HISTORY_FEATURE_NAMES, *CANONICAL_RFM_TIER_FEATURE_NAMES,
    *("prev_" + name for name in PRODUCT_COLUMNS),
    *acquisition_feature_names(PRODUCT_COLUMNS),
)))

# Public notebook/config catalog: each name owns precisely these outputs.
# Age is both cleaned by profile preprocessing and projected by persona logic;
# selecting that shared output forces both producers. Renta has one owner.
PIPELINE_FUNCTION_OUTPUTS: dict[str, tuple[str, ...]] = {
    "customer_profile_preprocessing": (*tuple(c for c in PROFILE_COLUMNS if c != "renta"), PREPROCESSING_MARKER),
    "renta_monetary_preprocessing": (*RENTA_OUTPUTS, "renta"),
    "persona_features": PERSONA_FEATURES,
    "previous_product_states": tuple("prev_" + name for name in PRODUCT_COLUMNS),
    "acquisition_features": acquisition_feature_names(PRODUCT_COLUMNS),
    "record_gap_months_sql": ("record_gap_months",),
    "rfm_has_previous_record": ("rfm_has_previous_record",),
    "rfm_recency_months_sql": ("rfm_recency_months",),
    "rfm_never_acquired_before_sql": ("rfm_never_acquired_before",),
    "rfm_frequency_sql": ("rfm_frequency",),
    "rfm_monetary_sql": ("rfm_monetary",),
    "rfm_history_coverage_sql": ("rfm_observed_history_records",),
    "acquisitions_last_1m_sql": ("acquisitions_last_1m",),
    "acquisitions_last_3m_sql": ("acquisitions_last_3m",),
    "acquisitions_last_6m_sql": ("acquisitions_last_6m",),
    "cumulative_drops_sql": ("cumulative_drops",),
    "rfm_tiering": CANONICAL_RFM_TIER_FEATURE_NAMES,
}

# Actual persisted inputs: history functions currently compute their shared
# intermediates directly from product states, not from stored prev_/acq_ outputs.
PIPELINE_FUNCTION_INPUTS = {
    name: (*KEY_COLUMNS, *(RAW_PERSONA_COLUMNS if name == "persona_features" else PRODUCT_COLUMNS))
    for name in PIPELINE_FUNCTION_OUTPUTS
}
PIPELINE_FUNCTION_INPUTS.update({
    "rfm_tiering": CANONICAL_RFM_TIER_INPUTS,
    "customer_profile_preprocessing": (*KEY_COLUMNS, "age", "antiguedad", "renta"),
    "renta_monetary_preprocessing": (*KEY_COLUMNS, "renta", "cod_prov", "pais_residencia"),
})


def run_data_pipeline(
    *,
    required_features: Sequence[str] = (),
    force_process: bool = False,
    force_features: Sequence[str] = (),
    force_functions: Sequence[str] = (),
    publish_checkpoint: bool = True,
    dry_run: bool = False,
    data_root: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
    work_root: str | Path | None = None,
    raw_filename: str = "train_ver2.csv",
    download_raw: bool = True,
    chunksize: int = 150_000,
    show_progress: bool = True,
    memory_limit: str = "4GB",
    temp_directory: str | Path | None = None,
    renta_clip_upper_quantile: float = 0.99,
) -> dict[str, Any]:
    """Resolve a retained snapshot or compute and optionally publish a candidate.

    Empty requirements reuse the latest valid snapshot. ``force_process``
    bypasses reuse and recomputes on new files. Explicit feature/function
    selections bypass reuse and force their owners plus downstream consumers
    of updated inputs. Features outside executed contracts are retained;
    missing-output functions still run. ``dry_run`` previews without writes.
    Cleanup remains an explicit
    infrastructure operation; this entrypoint never deletes retained versions.
    """
    forced_steps = resolve_forced_steps(
        PIPELINE_FUNCTION_OUTPUTS, force_features=force_features, force_functions=force_functions,
    )
    root = Path(data_root or os.getenv("SANTANDER_DATA_ROOT", "data"))
    store = CheckpointStore(checkpoint_root or root / "processed/canonical_customer_month")
    required = tuple(dict.fromkeys(required_features))
    resolved = store.resolve(required, required_artifacts=("train",))
    if resolved is not None and not force_process and not forced_steps and not dry_run:
        path = store.root / resolved["artifacts"]["train"]["path"]
        print(f"[checkpoint] REUSE version={resolved['version']} | final validation=PASS")
        return {"status": "reused", "customer_month": path, "version": resolved,
                "processing": [], "published": False}

    parent = resolved or store.resolve([], required_artifacts=("train",))
    if dry_run:
        source = store.root / parent["artifacts"]["train"]["path"] if parent else root / "interim/train.parquet"
        columns = set()
        if source.is_file():
            with duckdb.connect() as con:
                columns = {r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]).fetchall()}
        preview_steps = [ProcessingStep(
            name,
            tuple(c for c in outputs if name != "customer_profile_preprocessing" or not columns
                  or c in columns or c == PREPROCESSING_MARKER),
            PIPELINE_FUNCTION_INPUTS[name], lambda current: current,
        ) for name, outputs in PIPELINE_FUNCTION_OUTPUTS.items()]
        processing = plan_processing_pipeline(preview_steps, columns, force_process=force_process,
                                              force_features=force_features, force_functions=force_functions)
        will_reuse = resolved is not None and not force_process and not forced_steps
        if will_reuse:
            # Match the resolver's minimum-schema reuse path exactly, even if
            # the retained snapshot does not contain every optional feature.
            for summary in processing:
                summary.update(decision="SKIP", force_process=False, upstream_functions=[],
                               reasons=["checkpoint_reuse"])
        for summary in processing:
            print(json_summary(summary))
        predicted = columns | {c for s in processing for c in s["declared_outputs"] if s["decision"] == "RUN"}
        return {"status": "planned", "customer_month": source if source.is_file() else None,
                "version": parent, "processing": processing, "published": False,
                "will_reuse": will_reuse,
                "needs_ingestion": parent is None,
                "unsatisfied_required_features": sorted(set(required) - predicted)}
    work = Path(work_root or root / ".pipeline_work")
    work.mkdir(parents=True, exist_ok=True)
    # Unique run directories also prevent accidental reuse of stale sidecars.
    run_dir = Path(mkdtemp(prefix="customer_month_", dir=work))
    spill = Path(temp_directory or os.getenv("SANTANDER_DUCKDB_TEMP_DIRECTORY", str(work / "duckdb_temp")))
    spill.mkdir(parents=True, exist_ok=True)

    def connect() -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(database=":memory:")
        con.execute("SET memory_limit = ?", [memory_limit])
        con.execute("SET temp_directory = ?", [str(spill)])
        return con

    def schema(path: Path) -> dict[str, str]:
        with connect() as con:
            return dict((row[0], row[1]) for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
            ).fetchall())

    def validate(artifacts: dict[str, Path]) -> None:
        for path in artifacts.values():
            columns = schema(path)
            if missing := set(KEY_COLUMNS).difference(columns):
                raise ValueError(f"Missing customer-month keys: {sorted(missing)}")
            with connect() as con:
                valid = con.execute(
                    "SELECT COUNT(*) = COUNT(DISTINCT (ncodpers, fecha_dato)) "
                    "AND COUNT(*) = COUNT(ncodpers) AND COUNT(*) = COUNT(fecha_dato) "
                    "FROM read_parquet(?)", [str(path)],
                ).fetchone()[0]
                if not valid:
                    raise ValueError(f"Non-null unique customer-month grain violated: {path}")
                counts = (
                    "rfm_frequency", "rfm_monetary", "rfm_recency_months",
                    "rfm_observed_history_records", "record_gap_months",
                    "acquisitions_last_1m", "acquisitions_last_3m",
                    "acquisitions_last_6m", "cumulative_drops",
                )
                binary = (
                    *PRODUCT_COLUMNS, *("prev_" + name for name in PRODUCT_COLUMNS),
                    *acquisition_feature_names(PRODUCT_COLUMNS),
                    "rfm_has_previous_record", "rfm_never_acquired_before",
                    "is_new_customer", "is_active_customer", "is_male",
                    "is_deceased", "is_domestic", "profile_missing_structural",
                )
                conditions = [f"{quote(name)} < 0" for name in counts if name in columns]
                conditions.extend(f"{quote(name)} NOT IN (0, 1)" for name in binary if name in columns)
                if PREPROCESSING_MARKER in columns:
                    conditions.append(f"{quote(PREPROCESSING_MARKER)} IS DISTINCT FROM TRUE")
                if conditions and con.execute(
                    f"SELECT COUNT(*) FROM {parquet(path)} WHERE {' OR '.join(conditions)}"
                ).fetchone()[0]:
                    raise ValueError(f"Feature-specific validation failed: {path}")

    def copy_query(query: str, destination: Path) -> Path:
        escaped = str(destination).replace("'", "''")
        with connect() as con:
            con.execute(f"COPY ({query}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        return destination

    def parquet(path: Path) -> str:
        return "read_parquet('" + str(path).replace("'", "''") + "')"

    def merge(source: Path, result: Path, destination: Path, outputs: Sequence[str]) -> Path:
        validate({"source": source, "result": result})
        with connect() as con:
            different_keys = con.execute(
                f"SELECT COUNT(*) FROM {parquet(source)} s FULL OUTER JOIN "
                f"{parquet(result)} r USING (ncodpers, fecha_dato) "
                "WHERE s.ncodpers IS NULL OR r.ncodpers IS NULL"
            ).fetchone()[0]
        if different_keys:
            raise ValueError("Processing changed the customer-month key set.")
        replaced = set(outputs)
        projection = ["s." + quote(name) for name in schema(source) if name not in replaced]
        projection.extend("r." + quote(name) for name in outputs)
        return copy_query(
            f"SELECT {', '.join(projection)} FROM {parquet(source)} s "
            f"JOIN {parquet(result)} r USING (ncodpers, fecha_dato)", destination,
        )

    def ingest() -> Path:
        raw, interim = root / "raw" / raw_filename, root / "interim/train.parquet"
        if not interim.is_file():
            if not raw.is_file() and download_raw:
                bucket = os.getenv("GCS_BUCKET")
                if not bucket:
                    raise ValueError("Set GCS_BUCKET or place the raw CSV under data/raw.")
                download_object_if_missing(
                    bucket, object_name(os.getenv("GCS_RAW_PREFIX", "raw"), raw_filename),
                    raw, os.getenv("GOOGLE_CLOUD_PROJECT"),
                )
            build_interim_checkpoint(raw, interim, chunksize=chunksize, show_progress=show_progress)
        validate({"train": interim})
        return interim

    working = {"train": store.root / parent["artifacts"]["train"]["path"]} if parent else {"train": ingest()}
    profile_outputs = tuple(name for name in PROFILE_COLUMNS if name != "renta" and name in schema(working["train"]))

    def preprocess(current: dict[str, Path]) -> dict[str, Path]:
        print("[processing] RUN customer_profile_preprocessing")
        # Fit from upstream values rather than already imputed/clipped columns.
        raw_source = ingest()
        result = preprocess_customer_profile_baseline(
            raw_source, None, run_dir / "profile", force_process=True,
            memory_limit=memory_limit, temp_directory=spill,
            renta_clip_upper_quantile=renta_clip_upper_quantile,
        )
        cleaned = copy_query(
            f"SELECT *, TRUE AS {quote(PREPROCESSING_MARKER)} "
            f"FROM {parquet(Path(result['train']))}", run_dir / "profile_marked.parquet",
        )
        return {"train": merge(current["train"], cleaned, run_dir / "preprocessed.parquet",
                               (*profile_outputs, PREPROCESSING_MARKER))}

    def feature_step(name: str, outputs: tuple[str, ...]) -> ProcessingStep:
        def engineer(current: dict[str, Path]) -> dict[str, Path]:
            # Outer orchestration has already decided RUN. Force precisely this
            # contract even if its current values exist in the source snapshot.
            features = materialize_customer_month_feature_group(
                current["train"], run_dir / f"{name}_features.parquet",
                outputs=outputs, force_process=True,
                memory_limit=memory_limit, temp_directory=spill,
            )
            destination = merge(current["train"], features, run_dir / f"{name}_candidate.parquet", outputs)
            # Only discard this run's earlier feature candidates, never retained
            # versions, raw data, preprocessing metadata, or external sources.
            previous = current["train"]
            if previous.parent.resolve() == run_dir.resolve() and previous.name.endswith("_candidate.parquet"):
                previous.unlink()
            return {"train": destination}

        return ProcessingStep(name, outputs, PIPELINE_FUNCTION_INPUTS[name], engineer, validate)

    def preprocess_income(current: dict[str, Path]) -> dict[str, Path]:
        # Fit the approved RFM rule on upstream income, before baseline fill/clip.
        result = preprocess_renta_monetary(
            ingest(), run_dir / "renta_monetary.parquet", force_process=True,
            memory_limit=memory_limit, temp_directory=spill,
        )
        income = copy_query(
            f"SELECT ncodpers, fecha_dato, {', '.join(RENTA_OUTPUTS)}, renta_filled AS renta "
            f"FROM {parquet(result)}", run_dir / "income.parquet",
        )
        return {"train": merge(current["train"], income, run_dir / "income_candidate.parquet",
                               (*RENTA_OUTPUTS, "renta"))}

    def tier_rfm(current: dict[str, Path]) -> dict[str, Path]:
        result = materialize_canonical_rfm_tiers(
            current["train"], run_dir / "rfm_tiering_candidate.parquet",
            force_process=True, memory_limit=memory_limit, temp_directory=spill,
        )
        return {"train": result}

    steps = [
        ProcessingStep("customer_profile_preprocessing", (*profile_outputs, PREPROCESSING_MARKER),
                       (*KEY_COLUMNS, "age", "antiguedad", "renta"), preprocess, validate),
        ProcessingStep("renta_monetary_preprocessing", (*RENTA_OUTPUTS, "renta"),
                       (*KEY_COLUMNS, "renta", "cod_prov", "pais_residencia"), preprocess_income, validate),
        *(feature_step(name, outputs) for name, outputs in PIPELINE_FUNCTION_OUTPUTS.items()
          if name not in {"customer_profile_preprocessing", "renta_monetary_preprocessing", "rfm_tiering"}),
        ProcessingStep("rfm_tiering", CANONICAL_RFM_TIER_FEATURE_NAMES,
                       CANONICAL_RFM_TIER_INPUTS, tier_rfm, validate),
    ]
    candidates, processing = run_processing_pipeline(
        working, steps, force_process=force_process, required_features=required,
        force_features=force_features, force_functions=force_functions,
    )
    validate(candidates)
    # Bounded disk-backed verification before publication: unrelated values and
    # their types must match the parent exactly, regardless of output ordering.
    if parent:
        before = working["train"]
        after = candidates["train"]
        replaced = {c for s in processing if s["decision"] == "RUN" for c in s["declared_outputs"]}
        old_schema, new_schema = schema(before), schema(after)
        preserved = [c for c in old_schema if c not in replaced]
        if any(new_schema.get(c) != old_schema[c] for c in preserved):
            raise ValueError("Processing changed unrelated column types")
        projection = ", ".join(quote(c) for c in preserved)
        with connect() as con:
            changed = con.execute(
                f"SELECT EXISTS ((SELECT {projection} FROM {parquet(before)} EXCEPT ALL "
                f"SELECT {projection} FROM {parquet(after)}) UNION ALL "
                f"(SELECT {projection} FROM {parquet(after)} EXCEPT ALL "
                f"SELECT {projection} FROM {parquet(before)}))"
            ).fetchone()[0]
        if changed:
            raise ValueError("Processing changed unrelated feature values")
    for summary in processing:
        print(json_summary(summary))
    print(f"[pipeline] required_features={list(required)} | final validation=PASS | publish_checkpoint={publish_checkpoint}")
    configuration = {
        "required_features": list(required), "force_process": force_process,
        "force_features": list(force_features), "force_functions": list(force_functions),
        "forced_functions_resolved": sorted(forced_steps),
        "functions_to_run": [s["step"] for s in processing if s["decision"] == "RUN"],
        "publish_checkpoint": publish_checkpoint, "raw_filename": raw_filename,
        "chunksize": chunksize, "memory_limit": memory_limit,
        "renta_clip_upper_quantile": renta_clip_upper_quantile,
        "rfm_tiering": {"boundaries": CANONICAL_RFM_BOUNDARIES, "methods": CANONICAL_RFM_METHODS},
        "preprocessing_stats": (
            json.loads((run_dir / "profile/preprocessing_stats.json").read_text(encoding="utf-8"))
            if processing[0]["decision"] == "RUN"
            else (parent or {}).get("configuration", {}).get("preprocessing_stats")
        ),
    }
    version = store.publish(
        candidates, parent_version=parent["version"] if parent else None,
        processing=processing, configuration=configuration,
        feature_definition_metadata={step.name: {"outputs": list(step.outputs), "inputs": list(step.required_inputs)} for step in steps},
    ) if publish_checkpoint else None
    path = store.root / version["artifacts"]["train"]["path"] if version else candidates["train"]
    return {"status": "published" if version else "candidate", "customer_month": path,
            "version": version, "processing": processing, "published": version is not None}


def json_summary(summary: dict[str, Any]) -> str:
    """Keep execution decisions and declared-output validation visible."""
    return "[processing] " + json.dumps(summary, ensure_ascii=False)
