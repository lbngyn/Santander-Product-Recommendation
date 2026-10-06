"""Approved engineered acquisition features and immutable split checkpoints.

Income uses the user's accepted canonical static policy. All product history
inputs are prior-only; frozen RFM boundaries are supplied, never fitted here.
"""
from pathlib import Path
import shutil

from src.features import acquisition_only as original
from src.features.acquisition_only_contract import FEATURES, KEYS, parquet, quote
from src.features.persona import MODEL_PERSONA_FEATURES, CATEGORICAL_PERSONA_FEATURES
from src.features.customer_history import HISTORY_FEATURE_NAMES
from src.features.rfm_tiering import RFM_TIERING_FEATURE_NAMES, materialize_rfm_tiering_features, _tier_case_sql
from src.features.feature_store import materialize_customer_month_feature_group
from src.features.checkpoint_store import CheckpointStore, ProcessingStep, plan_processing_pipeline
from src.utils.modeling_runtime import connection
from src.utils.run_files import read_json, write_json, sha256, now
from src.tracking.run_context import create_run_id
from src.tracking.run_bundle import safe_id

PERSONA = tuple(n for n in MODEL_PERSONA_FEATURES if n not in FEATURES)
TIERS = tuple(n for n in RFM_TIERING_FEATURE_NAMES if n not in ('renta_filled', 'renta_imputation_method'))
MODEL_FEATURES = list(dict.fromkeys([*FEATURES, *PERSONA, *HISTORY_FEATURE_NAMES, *TIERS]))
FUNCTION_OUTPUTS = {'original_inputs': tuple(FEATURES), 'persona_features': PERSONA,
                    'history_features': HISTORY_FEATURE_NAMES, 'rfm_tiering': TIERS}
TIER_CATEGORIES = ('RFM_portfolio_score', 'RFM_renta_score', 'RFM_portfolio_tier', 'RFM_renta_tier')


def frozen_boundaries(config, root):
    """Read the exact EDA export (selected_tiering), or explicit frozen values."""
    spec = config['features'].get('rfm_boundaries')
    if spec is None:
        declared = config['features'].get('rfm_spec_path')
        if not declared:
            raise ValueError('Set features.rfm_spec_path to the approved RFM metadata JSON, or supply rfm_boundaries')
        path = Path(declared).expanduser()
        if not path.is_absolute():
            path = Path(root) / path
        if not path.is_file():
            raise FileNotFoundError(f'Frozen RFM metadata missing: {path}. Supply the exact approved EDA export or features.rfm_boundaries; tier selection is never rerun by this pipeline.')
        payload = read_json(path)
        spec = payload.get('selected_tiering', payload)
    result = {}
    for key in ('recency', 'frequency', 'monetary_portfolio', 'monetary_renta'):
        value = spec[key]
        if isinstance(value, dict):
            field = 'boundaries_log' if key == 'monetary_renta' else 'boundaries'
            value = value[field]
        if not value:
            raise ValueError(f'Empty frozen RFM boundaries: {key}')
        _tier_case_sql('value', value, lower_is_better=key == 'recency')
        result[key] = list(map(float, value))
    return result


def _root(config, root):
    return Path(root) / 'processed/modeling/acquisition_only_engineered' / safe_id(config['pipeline']['comparison_id'])


def checkpoint_store(output, split, runtime, work):
    return CheckpointStore(Path(output) / 'checkpoints' / split,
        runtime={**runtime, 'temp_directory': str(Path(work) / 'checkpoint_spill')})


def _steps():
    return [ProcessingStep(name, outputs,
                tuple(FEATURES) if name in ('persona_features', 'history_features') else
                ('renta', 'rfm_recency_months', 'rfm_never_acquired_before', 'rfm_frequency', 'rfm_monetary') if name == 'rfm_tiering' else (),
                lambda current: current)
            for name, outputs in FUNCTION_OUTPUTS.items()]


def _plan(config, store, split, runtime, work):
    version = store.resolve([], required_artifacts=(split,))
    path = store.root / version['artifacts'][split]['path'] if version else None
    with connection(runtime, work) as con:
        columns = {r[0] for r in con.execute(f'DESCRIBE SELECT * FROM {parquet(path)}').fetchall()} if path else set(KEYS)
    processing = config.get('processing', {})
    selected = split in processing.get('splits', ['train', 'validation', 'test'])
    plan = plan_processing_pipeline(_steps(), columns,
        force_process=selected and bool(processing.get('force_process', False)),
        force_features=processing.get('force_features', []) if selected else (),
        force_functions=processing.get('force_functions', []) if selected else ())
    return version, path, plan


def preview(config, root, work):
    boundaries = frozen_boundaries(config, root)
    output = _root(config, root)
    _check_identity(output, config, boundaries)
    splits = {}
    for split in ('train', 'validation', 'test'):
        store = checkpoint_store(output, split, config['runtime'], work)
        version, path, plan = _plan(config, store, split, config['runtime'], work)
        splits[split] = {'version': version['version'] if version else None,
                         'checkpoint': str(path) if path else None, 'processing': plan}
    return {'dataset_root': str(output), 'feature_names': MODEL_FEATURES,
            'rfm_boundaries': boundaries, 'splits': splits,
            'source': original.preview(config, root, work)}


def _check_identity(output, config, boundaries):
    path = output / 'identity.json'
    identity = {'validation_start': config['split']['validation_start'],
                'validation_end': config['split']['validation_end'], 'rfm_boundaries': boundaries,
                'income_policy': 'accepted_canonical_static',
                'checkpoint_version': config['data'].get('checkpoint_version')}
    if path.exists() and read_json(path) != identity:
        raise ValueError('Prepared feature identity differs; use a new comparison_id')
    return identity


def _merge(con, current, narrow, outputs, destination):
    columns = [r[0] for r in con.execute(f'DESCRIBE SELECT * FROM {parquet(current)}').fetchall()]
    kept = [n for n in columns if n not in outputs]
    projection = ', '.join([*(f'b.{quote(n)}' for n in kept), *(f'f.{quote(n)}' for n in outputs)])
    # Reject missing or duplicate feature rows before publishing an input.
    bad = con.execute(f'''SELECT COUNT(*) != COUNT(DISTINCT (b.ncodpers,b.fecha_dato))
        OR COUNT(*) FILTER (WHERE f.ncodpers IS NULL) > 0
        FROM {parquet(current)} b LEFT JOIN {parquet(narrow)} f USING (ncodpers,fecha_dato)''').fetchone()[0]
    if bad:
        raise ValueError('Feature join changed customer-month grain or omitted inputs')
    original._copy(con, f'SELECT {projection} FROM {parquet(current)} b LEFT JOIN {parquet(narrow)} f USING (ncodpers,fecha_dato) ORDER BY b.ncodpers,b.fecha_dato', destination)


def _split_checkpoint(config, output, split, base, source, boundaries, work, test_sources=None):
    runtime = config['runtime']
    store = checkpoint_store(output, split, runtime, work)
    version, current, plan = _plan(config, store, split, runtime, work)
    if all(p['decision'] == 'SKIP' for p in plan):
        return current, version
    work = Path(work) / create_run_id(split); work.mkdir(parents=True)
    generated_test = None
    for row in plan:
        if row['decision'] == 'SKIP':
            continue
        name, outputs = row['step'], row['declared_outputs']
        if name == 'original_inputs':
            if split == 'test':
                from src.ingestion.checkpoint import build_interim_checkpoint
                from src.submission.prepare import materialize_acquisition_only_test_input
                raw, history = test_sources
                test = work / 'raw_test.parquet'
                build_interim_checkpoint(raw, test, chunksize=int(runtime['batch_customer_months']), force_rebuild=True,
                    show_progress=bool(runtime.get('show_progress', True)))
                base = materialize_acquisition_only_test_input(test, history, work / 'original_test.parquet', runtime, work / 'original_work')
            if current is None:
                current = base
                continue
            narrow = base
        elif name in ('persona_features', 'history_features'):
            narrow = work / f'{name}.parquet'
            if split == 'test':
                if generated_test is None:
                    from src.ingestion.checkpoint import build_interim_checkpoint
                    from src.submission.prepare import materialize_competition_test_input
                    from src.products import PRODUCT_COLUMNS
                    raw, history = test_sources
                    test = work / 'raw_test.parquet'
                    if not test.is_file():
                        build_interim_checkpoint(raw, test, chunksize=int(runtime['batch_customer_months']), force_rebuild=True,
                            show_progress=bool(runtime.get('show_progress', True)))
                    generated_test = materialize_competition_test_input(test, history, work / 'engineered_test.parquet',
                        product_names=PRODUCT_COLUMNS, feature_names=[*PERSONA, *HISTORY_FEATURE_NAMES],
                        history_date='2016-05-31', memory_limit=runtime['memory_limit'], temp_directory=work / 'spill')
                narrow = generated_test
            else:
                with connection(runtime, work) as con:
                    available = {r[0] for r in con.execute(f'DESCRIBE SELECT * FROM {parquet(source)}').fetchall()}
                # Reuse canonical declared outputs; explicit force recomputes their producer.
                if set(outputs).issubset(available) and not row['force_process']:
                    narrow = source
                else:
                    materialize_customer_month_feature_group(source, narrow, outputs=outputs, force_process=True,
                        memory_limit=runtime['memory_limit'], temp_directory=work / 'spill')
            if name == 'persona_features':
                # Income representation follows the accepted original-input policy in every split.
                with connection(runtime, work) as con:
                    others = ', '.join(f'f.{quote(n)}' for n in outputs if n != 'income_log')
                    income_persona = work / 'persona_income.parquet'
                    original._copy(con, f'SELECT b.ncodpers,b.fecha_dato,{others},LN(1+b.renta) AS income_log FROM {parquet(current)} b JOIN {parquet(narrow)} f USING (ncodpers,fecha_dato)', income_persona)
                    narrow = income_persona
        else:
            proxy = work / 'income_proxy.parquet'
            with connection(runtime, work) as con:
                original._copy(con, f"SELECT ncodpers,fecha_dato,renta AS renta_filled,'accepted_canonical' AS renta_imputation_method FROM {parquet(current)}", proxy)
            narrow = materialize_rfm_tiering_features(current, proxy, work / 'tiers.parquet', boundaries=boundaries,
                force_process=True, memory_limit=runtime['memory_limit'], temp_directory=work / 'spill')
        destination = work / f'merged_{name}.parquet'
        with connection(runtime, work) as con:
            _merge(con, current, narrow, outputs, destination)
        previous = current; current = destination
        if previous.parent == work and previous.name.startswith('merged_'):
            previous.unlink()
    with connection(runtime, work) as con:
        final = work / 'inputs.parquet'
        original._copy(con, f"SELECT {', '.join(map(quote,[*KEYS,*MODEL_FEATURES]))} FROM {parquet(current)} ORDER BY ncodpers,fecha_dato", final)
    version = store.publish({split: final}, parent_version=version['version'] if version else None,
        processing=plan, configuration={'rfm_boundaries': boundaries, 'income_policy': 'accepted_canonical_static'})
    # Candidate files are disposable; published versions are never mutated.
    for path in work.glob('*.parquet'):
        path.unlink()
    return store.root / version['artifacts'][split]['path'], version


def prepare(config, root, work):
    boundaries = frozen_boundaries(config, root); output = _root(config, root)
    identity = _check_identity(output, config, boundaries)
    base = original.prepare(config, root, Path(work) / 'original')
    manifest = read_json(base / 'dataset_manifest.json'); source = Path(manifest['checkpoint_path'])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'identity.json', identity)
    checkpoints = {}
    for split in ('train', 'validation'):
        path, version = _split_checkpoint(config, output, split, base / f'{split}_inputs.parquet', source,
                                          boundaries, Path(work) / 'features')
        checkpoints[split] = {'path': str(path), 'version': version['version']}
    pointer = output / 'prepared.json'
    if pointer.is_file():
        previous = read_json(pointer)
        if previous['checkpoints'] == checkpoints:
            pack = output / previous['pack']
            for name, digest in read_json(pack / 'dataset_manifest.json')['files'].items():
                if sha256(pack / name) != digest:
                    raise ValueError(f'Prepared pack changed: {name}')
            return pack
    pack = output / 'packs' / create_run_id('prepared'); pack.mkdir(parents=True)
    for split, record in checkpoints.items():
        shutil.copy2(record['path'], pack / f'{split}_inputs.parquet')
        shutil.copy2(base / f'{split}_targets.parquet', pack / f'{split}_targets.parquet')
    contract = dict(read_json(base / 'feature_contract.json'))
    contract.update(feature_names=MODEL_FEATURES, feature_scope='approved_engineered', rfm_boundaries=boundaries,
        forbidden_inputs=['current_product_states','acq_*','n_acquisitions','sampled_for_*','lifecycle_*','CLV_*'],
        income_policy='accepted_canonical_static_income',
        feature_availability={'product_history': 'strictly_before_prediction_snapshot',
            'profile': 'prediction_snapshot', 'income': 'user_approved_offline_static_policy'})
    vocabulary = dict(contract['category_values'])
    with connection(config['runtime'], work) as con:
        for name in [*(n for n in CATEGORICAL_PERSONA_FEATURES if n in PERSONA), *TIER_CATEGORIES]:
            vocabulary[name] = [r[0] for r in con.execute(f'SELECT DISTINCT {quote(name)} FROM {parquet(pack / "train_inputs.parquet")} WHERE {quote(name)} IS NOT NULL ORDER BY 1').fetchall()]
    contract['category_values'] = vocabulary
    write_json(pack / 'feature_contract.json', contract)
    write_json(pack / 'cohort_audit.json', read_json(base / 'cohort_audit.json'))
    manifest.update(feature_contract=contract, feature_checkpoints=checkpoints, prepared_at=now(),
                    files={p.name: sha256(p) for p in pack.iterdir() if p.is_file()})
    write_json(pack / 'dataset_manifest.json', manifest)
    write_json(pointer, {'pack': pack.relative_to(output).as_posix(), 'checkpoints': checkpoints})
    return pack


def prepare_test(config, root, work, run, *, raw_test=None, history_path=None):
    run = Path(run); manifest = read_json(run / 'dataset_manifest.json')
    contract = read_json(run / 'feature_contract.json')
    history = Path(history_path or manifest['checkpoint_path'])
    if sha256(history) != manifest['checkpoint_sha256']:
        raise ValueError('Pinned canonical history changed')
    output = _root(config, root)
    _check_identity(output, config, contract['rfm_boundaries'])
    raw = Path(raw_test or Path(root) / config['competition']['raw_test'])
    test_identity = output / 'test_identity.json'
    identity = {'raw_test': str(raw.resolve()), 'history_sha256': manifest['checkpoint_sha256']}
    if test_identity.is_file() and read_json(test_identity) != identity:
        raise ValueError('Test sources differ; use a new comparison_id')
    path, version = _split_checkpoint(config, output, 'test', None, history,
        contract['rfm_boundaries'], Path(work) / 'test', test_sources=(raw, history))
    write_json(test_identity, identity)
    return path, version
