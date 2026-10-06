"""Shared notebook/CLI orchestration for engineered acquisition-only models."""
from pathlib import Path
import csv
import yaml
import pyarrow as pa
import pyarrow.parquet as pq

from src.features import acquisition_engineered as features
from src.features.acquisition_only_contract import parquet
from src.pipeline.acquisition_only import (
    locations, train_run, validate_run, finalize_run, publish_run, comparison,
)
from src.tracking.run_bundle import safe_id
from src.inference.acquisition_only import rank_prepared_batch
from src.utils.modeling_runtime import connection
from src.utils.run_files import read_json, write_json, sha256, now


def load_config(path, *, comparison_id=None, checkpoint_version=None):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for section in ('pipeline','data','split','features','model','runtime','tracking','processing','competition'):
        if not isinstance(config.get(section), dict):
            raise ValueError(f'Missing config section: {section}')
    if comparison_id is not None:
        config['pipeline']['comparison_id'] = comparison_id
    if checkpoint_version is not None:
        config['data']['checkpoint_version'] = int(checkpoint_version)
    for key in ('project_id','comparison_id'):
        safe_id(config['pipeline'][key])
    if config['pipeline']['approach'] not in ('independent','joint'):
        raise ValueError('Expected independent or joint')
    if config['features']['scope'] != 'approved_engineered' or config['features']['income_policy'] != 'accepted_canonical_static':
        raise ValueError('Unsupported feature or income policy')
    if config['features']['acquisition_policy'] != 'nearest_previous_observed_record':
        raise ValueError('Unsupported acquisition policy')
    if int(config['model']['n_estimators']) < 1 or min(int(config['runtime'][k]) for k in ('threads','batch_customer_months')) < 1:
        raise ValueError('Positive training/runtime budgets required')
    if set(config['processing'].get('splits', [])) - {'train','validation','test'}:
        raise ValueError('Unknown processing split')
    # Validate force selectors even when no checkpoints exist.
    from src.features.checkpoint_store import resolve_forced_steps
    resolve_forced_steps(features.FUNCTION_OUTPUTS,
        force_features=config['processing'].get('force_features', []),
        force_functions=config['processing'].get('force_functions', []))
    return config


def preview(config):
    root, work = locations(config)
    return features.preview(config, root, work / config['pipeline']['comparison_id'])


def prepare_dataset(config):
    root, work = locations(config)
    return features.prepare(config, root, work / config['pipeline']['comparison_id'])


def prepare_test(config, run, **kwargs):
    root, work = locations(config)
    # Separate preprocessing flags may change for a submission retry. Modeling settings may not.
    trained = yaml.safe_load((Path(run) / 'config_resolved.yaml').read_text(encoding='utf-8'))
    if {k:v for k,v in config.items() if k != 'processing'} != {k:v for k,v in trained.items() if k != 'processing'}:
        raise ValueError('Test preparation must use this run\'s saved config')
    return features.prepare_test(config, root, work / config['pipeline']['comparison_id'], run, **kwargs)


def create_submission(run, *, config=None, prepared_test=None, sample_submission=None):
    """Batch score Parquet and disk-join ranks to the official template order."""
    run = Path(run).resolve(); result = read_json(run / 'run_result.json')
    if result['status'] != 'FINISHED' or not result.get('inference_ready'):
        raise ValueError('Submission requires a FINISHED inference-ready run')
    config = config or yaml.safe_load((run / 'config_resolved.yaml').read_text(encoding='utf-8'))
    root, work_root = locations(config)
    path, version = prepare_test(config, run) if prepared_test is None else prepared_test
    contract = read_json(run / 'feature_contract.json')
    store = features.checkpoint_store(features._root(config, root), 'test', config['runtime'], work_root)
    expected = store.root / version['artifacts']['test']['path']
    if (Path(path).resolve() != expected.resolve()
            or version['configuration'].get('rfm_boundaries') != contract['rfm_boundaries']
            or not store.validate_version(version, required_features=contract['feature_names'])):
        raise ValueError('Prepared test does not match this run\'s approved feature checkpoint')
    template = Path(sample_submission or root / config['competition']['sample_submission'])
    if not template.is_file():
        raise FileNotFoundError(f'Official sample submission missing: {template}')
    work = work_root / result['source_run_id'] / 'submission'; work.mkdir(parents=True, exist_ok=True)
    batch_size = int(config['runtime']['batch_customer_months'])
    scores = work / 'recommendations.parquet'
    schema = pa.schema([('ncodpers', pa.int64()), ('added_products', pa.string())])
    required = ['ncodpers','fecha_dato',*contract['feature_names']]
    with pq.ParquetWriter(scores, schema, compression='zstd') as writer:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=required):
            ranked = rank_prepared_batch(batch.to_pandas(), run, max_batch_customers=batch_size)
            writer.write_table(pa.Table.from_pandas(ranked[['ncodpers','added_products']], schema=schema, preserve_index=False))
    output = root / 'artifacts/submissions' / result['source_run_id'] / 'submission.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with template.open(encoding='utf-8-sig', newline='') as handle:
        if next(csv.reader(handle), None) != ['ncodpers','added_products']:
            raise ValueError('Template columns must be ncodpers,added_products')
    temporary = output.with_suffix('.csv.tmp')
    with connection(config['runtime'], work) as con:
        # CSV template order is contractual, including with parallel scans.
        con.execute('SET preserve_insertion_order = true')
        escaped = str(template).replace("'", "''")
        con.execute(f"CREATE TABLE template AS SELECT ROW_NUMBER() OVER () AS ordinal,ncodpers FROM read_csv('{escaped}', header=true, columns={{'ncodpers':'BIGINT','added_products':'VARCHAR'}})")
        bad = con.execute(f'''SELECT (SELECT COUNT(*) != COUNT(DISTINCT ncodpers) OR COUNT(*) != COUNT(ncodpers) FROM template)
            OR (SELECT COUNT(*) != COUNT(DISTINCT ncodpers) OR COUNT(*) = 0 FROM {parquet(scores)})
            OR EXISTS(SELECT ncodpers FROM template EXCEPT SELECT ncodpers FROM {parquet(scores)})
            OR EXISTS(SELECT ncodpers FROM {parquet(scores)} EXCEPT SELECT ncodpers FROM template)''').fetchone()[0]
        if bad:
            raise ValueError('Test and official template must have identical unique customer IDs')
        escaped = str(temporary).replace("'", "''")
        con.execute(f"COPY (SELECT t.ncodpers,r.added_products FROM template t JOIN {parquet(scores)} r USING(ncodpers) ORDER BY t.ordinal) TO '{escaped}' (FORMAT CSV,HEADER TRUE)")
    temporary.replace(output)
    write_json(output.parent / 'submission_manifest.json', {
        'source_run_id': result['source_run_id'], 'approach': result['approach'], 'created_at': now(),
        'prepared_test': str(path), 'feature_checkpoint_version': version['version'],
        'feature_contract_sha256': sha256(run / 'feature_contract.json'),
        'test_input_sha256': sha256(path), 'sample_submission_sha256': sha256(template),
        'submission_sha256': sha256(output), 'top_k': 7, 'tie_break': 'canonical_product_order',
        'model_scope': 'same_pre_may_models_as_validation', 'income_policy': 'accepted_canonical_static'})
    scores.unlink()
    return output


def run_pipeline(config, *, publish=False):
    dataset = prepare_dataset(config)
    run = train_run(config, dataset)
    validation = validate_run(config, dataset, run)
    submission = create_submission(run, config=config)
    bundle = finalize_run(run)
    result = {'dataset': str(dataset), 'run': str(run), 'training': read_json(run / 'training_metrics.json'),
              'validation': validation, 'submission': str(submission), 'bundle': str(bundle)}
    if publish:
        result['published'] = publish_run(bundle, config)
    return result
