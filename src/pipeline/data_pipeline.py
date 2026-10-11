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
from src.features.rfm.monetary_solver import solve_monetary_proxy
from src.features.rfm.monetary_rarity_rental import FEATURE_NAME as RARITY_FEATURE, FORMULA_VERSION, build_rarity_monetary_proxy


PREPROCESSING_MARKER = "profile_preprocessed"
PROFILE_COLUMNS = (*RAW_PERSONA_COLUMNS, "conyuemp", "nomprov")
ENGINEERED_FEATURES = tuple(dict.fromkeys((
    *PERSONA_FEATURES, *HISTORY_FEATURE_NAMES, *CANONICAL_RFM_TIER_FEATURE_NAMES, RARITY_FEATURE,
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
    "monetary_rarity_rental": (RARITY_FEATURE,),
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
    "monetary_rarity_rental": (*KEY_COLUMNS, "renta_filled", *PRODUCT_COLUMNS),
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
    memory_limit: str = "1GB",
    threads: int = 2,
    validation_column_batch_size: int = 16,
    temp_directory: str | Path | None = None,
    renta_clip_upper_quantile: float = 0.99,
    monetary_proxy: str = "porfolio_deep",
    monetary_boundaries: Sequence[float] | None = None,
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
    proxy = solve_monetary_proxy(monetary_proxy)
    if proxy.name == "rental" and monetary_boundaries is None:
        raise ValueError("rental tiering requires frozen monetary_boundaries on the log1p scale")
    if monetary_boundaries is not None:
        proxy.tier_sql("value", monetary_boundaries)
    if proxy.name == "rarity_rental" and (
        set(required_features).intersection(CANONICAL_RFM_TIER_FEATURE_NAMES)
        or set(force_features).intersection(CANONICAL_RFM_TIER_FEATURE_NAMES)
        or "rfm_tiering" in force_functions
    ):
        proxy.tier_sql(proxy.column, ())
    forced_steps = resolve_forced_steps(
        PIPELINE_FUNCTION_OUTPUTS, force_features=force_features, force_functions=force_functions,
    )
    root = Path(data_root or os.getenv("SANTANDER_DATA_ROOT", "data"))
    if threads < 1 or validation_column_batch_size < 1:
        raise ValueError("threads and validation_column_batch_size must be positive")
    runtime = {"memory_limit": memory_limit, "threads": threads}
    if temp_directory is not None and not dry_run:
        runtime["temp_directory"] = str(temp_directory)
    store = CheckpointStore(checkpoint_root or root / "processed/canonical_customer_month", runtime=runtime)
    required = tuple(dict.fromkeys((*required_features, *((proxy.column,) if proxy.name != "porfolio_deep" else ()))))
    monetary_contract = {"monetary_proxy": proxy.name,
                         "monetary_boundaries": list(monetary_boundaries) if monetary_boundaries is not None else None}
    resolved = store.resolve(required, required_artifacts=("train",), required_configuration=monetary_contract,
                             configuration_defaults={"monetary_proxy": "porfolio_deep", "monetary_boundaries": None})
    if resolved is not None and not force_process and not forced_steps and not dry_run:
        path = store.root / resolved["artifacts"]["train"]["path"]
        print(f"[checkpoint] REUSE version={resolved['version']} | final validation=PASS")
        return {"status": "reused", "customer_month": path, "version": resolved,
                "processing": [], "published": False, "monetary_proxy": proxy.name,
                "monetary_tiering_available": proxy.name != "rarity_rental"}

    parent = resolved or store.resolve([], required_artifacts=("train",))
    parent_config = (parent or {}).get("configuration", {})
    selector_changed = parent is not None and (
        parent_config.get("monetary_proxy", "porfolio_deep") != proxy.name
        or parent_config.get("monetary_boundaries") != monetary_contract["monetary_boundaries"])
    effective_force_functions = tuple(dict.fromkeys((*force_functions,
        *(("rfm_tiering",) if selector_changed and proxy.name != "rarity_rental" else ()))))
    parent_features = set((parent or {}).get("features", ()))
    include_rarity = (proxy.name == "rarity_rental" or RARITY_FEATURE in required
                      or RARITY_FEATURE in force_features or RARITY_FEATURE in parent_features
                      or "monetary_rarity_rental" in force_functions)
    active_outputs = {name: outputs for name, outputs in PIPELINE_FUNCTION_OUTPUTS.items()
                      if (name != "monetary_rarity_rental" or include_rarity)
                      and (name != "rfm_tiering" or proxy.name != "rarity_rental")}
    active_inputs = dict(PIPELINE_FUNCTION_INPUTS)
    active_inputs["rfm_tiering"] = (*CANONICAL_RFM_TIER_INPUTS[:-1], proxy.column)
    if dry_run:
        source = store.root / parent["artifacts"]["train"]["path"] if parent else root / "interim/train.parquet"
        columns = set()
        if source.is_file():
            with duckdb.connect(config={"memory_limit": memory_limit, "threads": threads}) as con:
                columns = {r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]).fetchall()}
        preview_steps = [ProcessingStep(
            name,
            tuple(c for c in outputs if name != "customer_profile_preprocessing" or not columns
                  or c in columns or c == PREPROCESSING_MARKER),
            active_inputs[name], lambda current: current,
        ) for name, outputs in active_outputs.items()]
        processing = plan_processing_pipeline(preview_steps, columns, force_process=force_process,
                                              force_features=force_features, force_functions=effective_force_functions)
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
                "needs_ingestion": parent is None, "monetary_proxy": proxy.name,
                "monetary_tiering_available": proxy.name != "rarity_rental",
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
        con.execute("SET threads = ?", [threads])
        con.execute("SET preserve_insertion_order = false")
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
                if RARITY_FEATURE in columns:
                    invalid = con.execute(
                        f"SELECT COUNT(*) FROM {parquet(path)} WHERE {quote(RARITY_FEATURE)} < 0 "
                        f"OR {quote(RARITY_FEATURE)} > {2 * len(PRODUCT_COLUMNS)} "
                        f"OR (NOT isfinite({quote(RARITY_FEATURE)}) AND {quote(RARITY_FEATURE)} IS NOT NULL)"
                    ).fetchone()[0]
                    if invalid:
                        raise ValueError("Invalid rarity monetary values")
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
            print(f"[processing] RUN {name}", flush=True)
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
            force_process=True, memory_limit=memory_limit, temp_directory=spill, threads=threads,
            monetary_proxy=proxy.name, monetary_boundaries=monetary_boundaries,
        )
        return {"train": result}

    def rarity_monetary(current: dict[str, Path]) -> dict[str, Path]:
        print("[processing] RUN monetary_rarity_rental", flush=True)
        result = build_rarity_monetary_proxy(
            current["train"], run_dir / "rarity_monetary_candidate.parquet",
            force_process=True, memory_limit=memory_limit, threads=threads, temp_directory=spill,
            weights_path=run_dir / "rarity_product_weights.parquet",
        )
        return {"train": result}

    steps = [
        ProcessingStep("customer_profile_preprocessing", (*profile_outputs, PREPROCESSING_MARKER),
                       (*KEY_COLUMNS, "age", "antiguedad", "renta"), preprocess, validate),
        ProcessingStep("renta_monetary_preprocessing", (*RENTA_OUTPUTS, "renta"),
                       (*KEY_COLUMNS, "renta", "cod_prov", "pais_residencia"), preprocess_income, validate),
        *(feature_step(name, outputs) for name, outputs in active_outputs.items()
          if name not in {"customer_profile_preprocessing", "renta_monetary_preprocessing", "rfm_tiering", "monetary_rarity_rental"}),
        *([ProcessingStep("monetary_rarity_rental", (RARITY_FEATURE,),
                          active_inputs["monetary_rarity_rental"], rarity_monetary, validate)] if include_rarity else []),
        *([ProcessingStep("rfm_tiering", CANONICAL_RFM_TIER_FEATURE_NAMES,
                          active_inputs["rfm_tiering"], tier_rfm, validate)] if proxy.name != "rarity_rental" else []),
    ]
    candidates, processing = run_processing_pipeline(
        working, steps, force_process=force_process, required_features=required,
        force_features=force_features, force_functions=effective_force_functions,
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
        print("[validation] Checking preserved values by snapshot and column batch", flush=True)
        with connect() as con:
            validate_preserved_values(
                con, before, after, preserved,
                column_batch_size=validation_column_batch_size,
            )
    for summary in processing:
        print(json_summary(summary))
    print(f"[pipeline] required_features={list(required)} | final validation=PASS | publish_checkpoint={publish_checkpoint}")
    rarity_ran = any(step["step"] == "monetary_rarity_rental" and step["decision"] == "RUN"
                     for step in processing)
    configuration = {
        **monetary_contract,
        "monetary_tiering_available": proxy.name != "rarity_rental",
        "rarity_income_column": "renta_filled" if include_rarity else None,
        "rarity_definition_version": (FORMULA_VERSION if rarity_ran else parent_config.get("rarity_definition_version")) if include_rarity else None,
        "required_features": list(required), "force_process": force_process,
        "force_features": list(force_features), "force_functions": list(force_functions),
        "forced_functions_resolved": sorted(forced_steps),
        "functions_to_run": [s["step"] for s in processing if s["decision"] == "RUN"],
        "publish_checkpoint": publish_checkpoint, "raw_filename": raw_filename,
        "chunksize": chunksize, "memory_limit": memory_limit,
        "threads": threads, "validation_column_batch_size": validation_column_batch_size,
        "renta_clip_upper_quantile": renta_clip_upper_quantile,
        "rfm_tiering": {
            "boundaries": ({**CANONICAL_RFM_BOUNDARIES,
                            "monetary": list(monetary_boundaries) if monetary_boundaries is not None
                            else CANONICAL_RFM_BOUNDARIES["monetary"]}
                           if proxy.name != "rarity_rental" else None),
            "methods": ({**CANONICAL_RFM_METHODS,
                         "monetary": "frozen_explicit" if monetary_boundaries is not None
                         else CANONICAL_RFM_METHODS["monetary"]}
                        if proxy.name != "rarity_rental" else None),
            "monetary_proxy": proxy.name, "available": proxy.name != "rarity_rental",
        },
        "preprocessing_stats": (
            json.loads((run_dir / "profile/preprocessing_stats.json").read_text(encoding="utf-8"))
            if processing[0]["decision"] == "RUN"
            else (parent or {}).get("configuration", {}).get("preprocessing_stats")
        ),
    }
    sidecars = {}
    weights = run_dir / "rarity_product_weights.parquet"
    if include_rarity and weights.is_file():
        sidecars["rarity_product_weights"] = weights
    elif include_rarity and parent:
        prior = parent.get("sidecars", {}).get("rarity_product_weights", {}).get("path")
        if prior and (store.root / prior).is_file():
            sidecars["rarity_product_weights"] = store.root / prior
    definitions = {step.name: {"outputs": list(step.outputs), "inputs": list(step.required_inputs)} for step in steps}
    if include_rarity:
        definitions["monetary_rarity_rental"].update(
            income_column="renta_filled", ownership="nullable lag of original product states",
            temporal_rule="fit target t from customer-month rows tau<t; ownership at nearest prior record of tau",
            income_policy="canonical offline filled income; may use future observations during preprocessing",
            formula="sum(prev_product*(1+normalized_IDF*C)); L=2*mu-1; P=2*STDDEV_POP(percentile)**2; C=L-P; no clipping",
            percentile_population="eligible historical customer-months with observed nullable lagged ownership and valid renta_filled",
            weights_grain=["target_month", "product"], definition_version=FORMULA_VERSION,
        )
        if not rarity_ran and parent:
            previous_definition = parent.get("feature_definition_metadata", {}).get("monetary_rarity_rental")
            if previous_definition is not None:
                definitions["monetary_rarity_rental"] = previous_definition
    version = store.publish(
        candidates, parent_version=parent["version"] if parent else None,
        processing=processing, configuration=configuration,
        feature_definition_metadata=definitions, sidecars=sidecars,
    ) if publish_checkpoint else None
    path = store.root / version["artifacts"]["train"]["path"] if version else candidates["train"]
    return {"status": "published" if version else "candidate", "customer_month": path,
            "version": version, "processing": processing, "published": version is not None,
            "monetary_proxy": proxy.name, "monetary_tiering_available": proxy.name != "rarity_rental"}


def json_summary(summary: dict[str, Any]) -> str:
    """Keep execution decisions and declared-output validation visible."""
    return "[processing] " + json.dumps(summary, ensure_ascii=False)


def validate_preserved_values(
    con: duckdb.DuckDBPyConnection,
    before: Path,
    after: Path,
    preserved: Sequence[str],
    *,
    column_batch_size: int = 16,
) -> None:
    """Compare exact values by unique customer-month key with bounded join width.

    Call after validating non-null unique keys and matching preserved types.
    Full outer joins detect missing/extra keys; IS DISTINCT FROM compares NULLs
    exactly. No hashes, sampling, row-order assumptions, or wide EXCEPT ALL.
    """
    if column_batch_size < 1:
        raise ValueError("column_batch_size must be positive")
    values = [name for name in preserved if name not in KEY_COLUMNS]
    batches = [values[i:i + column_batch_size] for i in range(0, len(values), column_batch_size)] or [[]]
    snapshots = con.execute(
        "SELECT fecha_dato FROM read_parquet(?) UNION SELECT fecha_dato FROM read_parquet(?) ORDER BY 1",
        [str(before), str(after)],
    ).fetchall()
    for (snapshot,) in snapshots:
        for batch in batches:
            projection = ", ".join(quote(c) for c in (*KEY_COLUMNS, *batch))
            conditions = ["s.ncodpers IS NULL", "r.ncodpers IS NULL"]
            conditions.extend(f"s.{quote(c)} IS DISTINCT FROM r.{quote(c)}" for c in batch)
            changed = con.execute(
                f"SELECT EXISTS (SELECT 1 FROM "
                f"(SELECT {projection} FROM read_parquet(?) WHERE fecha_dato = ?) s "
                f"FULL OUTER JOIN "
                f"(SELECT {projection} FROM read_parquet(?) WHERE fecha_dato = ?) r "
                "USING (ncodpers, fecha_dato) "
                f"WHERE {' OR '.join(conditions)} LIMIT 1)",
                [str(before), snapshot, str(after), snapshot],
            ).fetchone()[0]
            if changed:
                raise ValueError(f"Processing changed unrelated feature values or keys at snapshot {snapshot}")
