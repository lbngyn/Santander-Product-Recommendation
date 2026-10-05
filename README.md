# Santander Product Recommendation

# Overview dataset: 
- **Provider**: Santander Bank → Product là các dịch vụ liên quan tới tài chính (như là các loại tài khoản, các dịch vụ vay, thuế, credit card)
- **Dataset**:
    - Thời gian: `28-01-2015` → `28-05-2016`
    - Bản ghi các dịch vụ mà người dùng đã đăng kí của cty cùng với các thông tin cá nhân của người dùng.
- **Target**: tìm các **additional product** của từng customer tại thời điểm `28-06-2016`

## Data Pipeline: canonical customer-month

[Data Pipeline.ipynb](notebooks/Data%20Pipeline.ipynb) là notebook chuẩn bị dataset lịch sử dùng chung cho EDA và các tác vụ downstream. Mỗi hàng tương ứng một customer-month `(ncodpers, fecha_dato)`, gồm customer profile, product states và các feature persona, history, RFM. Pipeline dùng `train_ver2.csv`; competition test được xử lý riêng vì không có product states.

### Cấu trúc chính

| Thành phần | Vai trò |
| --- | --- |
| `notebooks/Data Pipeline.ipynb` | Bootstrap, khai báo feature cần dùng, cấu hình và gọi pipeline. |
| `src/pipeline/data_pipeline.py` | Điều phối ingest → preprocessing → feature engineering bằng các hàm hiện có. |
| `src/features/checkpoint_store.py` | Resolve version phù hợp, kiểm tra checkpoint và quản lý manifest/version bất biến. |
| `<SANTANDER_DATA_ROOT>/processed/canonical_customer_month/` | Lưu `manifest.json` và các full snapshot theo version, ví dụ `v000001/train.parquet`. |

### Cách sử dụng

1. Mở notebook trên Colab và chạy Bootstrap từ `notebooks/template.ipynb`. Branch được cấu hình trong Bootstrap cần có source mới nhất; thông tin runtime có thể cung cấp qua Colab Secrets hoặc bundle bên dưới.
2. Điền `REQUIRED_FEATURES` nếu cần một tập feature cụ thể. Giữ `[]` để dùng toàn bộ feature của checkpoint hợp lệ mới nhất; nếu chưa có checkpoint, pipeline tạo dataset lần đầu.
3. Chỉnh `FORCE_PROCESS`, `PUBLISH_CHECKPOINT` và `PIPELINE_CONFIG` khi cần, rồi chạy cell gọi pipeline. Mặc định là `FORCE_PROCESS=False`, `PUBLISH_CHECKPOINT=True`.
4. Dùng `CUSTOMER_MONTH_PATH` làm đường dẫn Parquet đầu vào cho tác vụ tiếp theo. `CHECKPOINT_VERSION` chứa metadata của version; giá trị là `None` nếu kết quả chỉ là candidate chưa publish.

### Cách hoạt động

```text
Bootstrap → REQUIRED_FEATURES → Resolve checkpoint
                                  ├─ Có version hợp lệ đủ feature → Reuse
                                  └─ Chưa có / force → Ingest → Preprocessing
                                                     → Feature engineering
                                                     → Validation → Optional publish
```

Pipeline reuse checkpoint mới nhất còn giữ lại và đáp ứng requirement. Khi cần xử lý, mỗi step chạy nếu thiếu declared output hoặc được force; step đủ output được skip. Candidate phải giữ đúng grain và đầy đủ required features trước khi được trả về.

`FORCE_PROCESS=True` tính lại outputs trên candidate mới. `PUBLISH_CHECKPOINT=True` lưu toàn bộ snapshot thành version mới và append manifest; `False` chỉ dùng candidate cho lần chạy hiện tại. Version đã publish không bị ghi đè, và notebook không tự cleanup version cũ.

Trên Colab, canonical checkpoints được lưu trên Drive qua `SANTANDER_DATA_ROOT`, còn working files và DuckDB spill nằm trên fast disk của runtime. Các version này có đường dẫn riêng với shared feature store của LightGBM V2/V3 được mô tả bên dưới; các pipeline model hiện vẫn dùng cấu hình feature store của chúng.

## Memory-safe ingestion

The Santander training CSV is too large to assume it fits in a 4 GB local machine. Ingestion is therefore streaming-only: it never loads or concatenates the complete dataset in memory.

What the pipeline does:

- Reads CSV or single-file ZIP input in bounded row chunks.
- Reads each raw chunk as strings. This prevents pandas from inferring a different type for sparse columns in different chunks (for example, an all-null `conyuemp` chunk versus a later chunk containing `S`/`N`).
- Normalises each chunk in place, then explicitly casts semantic numeric fields to nullable `Int64`, `Float32`, or `Int8`; all remaining fields are canonical nullable strings.
- Builds one canonical Arrow schema from the input header (therefore preserving the different train/test column sets) and forces every chunk to that schema before appending it to a Snappy Parquet checkpoint.
- Validates the finished Parquet schema against the canonical schema, then releases the pandas/Arrow temporary objects. Garbage collection runs periodically.
- Supports `usecols` in `read_csv_chunks()` for future consumers that truly need only a subset. Full ingest deliberately keeps all source columns to preserve the current checkpoint semantics.
- Keeps raw/checkpoint paths configurable through `SANTANDER_DATA_ROOT`; Colab bootstrap points this at mounted Drive while local defaults to `data/`.

Default chunk sizes are conservative for the target RAM budgets:

| Runtime | Default | Override |
| --- | ---: | --- |
| Local (4 GB) | 50,000 rows | `SANTANDER_CHUNKSIZE` or `config_from_environment(chunksize=...)` |
| Colab (12 GB) | 150,000 rows | `SANTANDER_CHUNKSIZE` or `config_from_environment(chunksize=...)` |

This bounds peak RAM roughly to the active chunk plus its short-lived pandas/Arrow conversion objects, rather than the full multi-GB CSV. Reading raw values as strings costs some additional memory within one chunk, but prevents schema drift and makes failures deterministic. Exact memory depends on column text length and missingness, so start with the defaults and decrease the chunk size if the runtime approaches its memory limit.

Trade-off: streaming performs more disk I/O and has per-chunk overhead, so it can be slower than a full in-memory load on a high-memory machine. It is intentionally preferred here for reliable ingestion on both 4 GB local CPU and 12 GB Colab CPU.

## Reusable customer-month feature checkpoint

Persona and history features are materialised with DuckDB into
`data/processed/customer_month_features.parquet`, keyed by
`(ncodpers, fecha_dato)`.  The companion `*_persona.parquet` and
`*_history.parquet` files are feature-group checkpoints.  V2/V3 model panels
read this common checkpoint and only add model-specific labels, temporal split,
and sampling flags.  This avoids rebuilding the same full persona/history
windows for each model version and bounds RAM to DuckDB's disk-backed query
execution.  The trade-off is extra Parquet disk space and an initial one-time
feature-store materialisation.  With `force_process=false`, existing feature
columns are reused; adding a feature to a group rebuilds that group and updates
the common checkpoint while preserving the other group.

The history group also contains the reusable `rfm_*` features extracted from
`notebooks/EDA/RFM_Tiering_EDA.ipynb`: prior-only recency and frequency, prior
portfolio size, and observation coverage. The per-snapshot acquisition count
is used internally to derive those features but is not persisted as a model
input, because it would reveal the current snapshot's outcome. RFM keeps an
acquisition across a calendar gap when the nearest previous product state is
observed. Model labels, training selection, and validation now use the same
nearest-record acquisition definition; adjacency is an explicit opt-in.

`src/features/rfm_tiering.py` additionally materialises the EDA's dual
Monetary RFM tiers when an experiment supplies frozen tier boundaries. Its
income inputs are preprocessed once by the canonical pipeline using the approved
RFM rule: observed, past fill, guarded backward fill (population persistence
at least 99%, customer at most one distinct income), geographic median of
customer medians, then global customer median. No P99 clipping applies to this
proxy. `renta_raw`, `renta_filled`, and `renta_imputation_method` preserve provenance;
RFM and CLV consume these fields without imputing again. This is an offline
proxy rule: backward fill and full-history statistics can use future observations. Both the proxy and tiering output are built with DuckDB and Parquet, so
peak RAM remains bounded rather than loading the customer-month table into
pandas. The trade-off is extra temporary/output disk I/O; callers can set a
DuckDB memory limit and spill directory through the feature functions.

### Using the shared feature store with LightGBM V2/V3

Set `SANTANDER_DATA_ROOT` to the directory containing `interim/train.parquet`
(or leave it unset to use `data/`). Both configs declare the same common
checkpoint:

```yaml
data:
  interim_train: interim/train.parquet
  feature_store: processed/customer_month_features.parquet
```

Run a training pipeline from a notebook or Python session:

```python
from src.pipeline.lightgbm_v2 import run_lightgbm_v2_from_config

result = run_lightgbm_v2_from_config("configs/baselines/lightgbm_v2.yaml")
print(result["model_panel"])
print(result["artifacts"])
```

For V3, change only the import/function. V3 reuses the same persona/history
feature store but creates its own panel and deterministic negative-sampling
flags:

```python
from src.pipeline.lightgbm_v3 import run_lightgbm_v3_from_config

result = run_lightgbm_v3_from_config("configs/baselines/lightgbm_v3.yaml")
```

On the first run, the following files are created. Later V2/V3 runs reuse
them while their required feature columns are already present:

```text
data/processed/
  customer_month_features_persona.parquet  # persona feature group
  customer_month_features_history.parquet  # history + product-state group
  customer_month_features.parquet          # shared model-agnostic checkpoint
```

To add a persona or history feature, add its definition to the relevant
feature module, then run once with `features.force_process: true` in the V2
or V3 config. This rebuilds the owned feature group and the common checkpoint;
set it back to `false` afterwards to reuse the checkpoint. Labels (`acq_*`),
validation decisions, and `sampled_for_*` columns are intentionally absent
from the shared checkpoint: they are created only in the named model panel.

To materialise or refresh only the shared checkpoint without training models:

```python
from pathlib import Path
from src.features.feature_store import ensure_customer_month_feature_store

root = Path("data")
store = ensure_customer_month_feature_store(
    root / "interim/train.parquet",
    root / "processed/customer_month_features.parquet",
    force_process=False,
    memory_limit="4GB",
    temp_directory=root / ".duckdb_tmp",
)
print(store)
```

## Acquisition target features

`src/features/acquisition.py` builds `acq_<product_column>` targets in
`data/processed/train.parquet`.  For every customer record, a target is `1`
only when the product changes from `0` in that customer's nearest earlier
record to `1` in the current record; all other cases, including a customer's
first record, are `0`.  Each target is stored as Parquet `TINYINT`.

The feature step uses DuckDB window functions and writes Parquet directly to a
temporary file before an atomic replace. This avoids loading the full panel
into pandas/RAM, at the cost of a disk-backed sort and one processed Parquet
write. `data/processed/test.parquet` is also copied once as the canonical test
checkpoint; it has no acquisition targets because Santander test data has no
product-state columns.

## VS Code → Colab CPU: runtime config bundle

Khi notebook chạy từ VS Code nhưng dùng Colab CPU, Colab Secrets không truy cập được. Bootstrap hỗ trợ một bundle Base64 duy nhất thay cho việc nhập từng credential.

Tạo `secrets/colab_runtime_config.json` (thư mục `secrets/` đã được Git ignore) với các key bắt buộc:

```json
{
  "GITHUB_TOKEN": "github_pat_...",
  "GCP_SERVICE_ACCOUNT_JSON": { "type": "service_account" },
  "GOOGLE_CLOUD_PROJECT": "your-project-id",
  "GCS_BUCKET": "your-bucket",
  "GCS_RAW_PREFIX": "santander/raw",
  "GCS_CHECKPOINT_PREFIX": "santander/interim"
}
```

`GCP_SERVICE_ACCOUNT_JSON` phải là toàn bộ JSON object trong service-account key, không chỉ phần `type` trong ví dụ.

Tạo Base64 một dòng và copy thẳng vào clipboard trên Windows PowerShell:

```powershell
$bundle = Get-Content -Raw "secrets\colab_runtime_config.json"
[Convert]::ToBase64String(
  [System.Text.Encoding]::UTF8.GetBytes($bundle)
) | Set-Clipboard
```

Chạy bootstrap cell trong notebook Colab rồi paste tại prompt `COLAB_RUNTIME_CONFIG_B64`. Bundle chỉ được giữ trong RAM của runtime hiện tại. Sau khi paste, xoá clipboard:


```powershell
Set-Clipboard -Value ""
```

Không commit file config, service-account JSON, GitHub PAT, Base64 bundle, hoặc output cell có secret. Nếu extension hỗ trợ environment variable cho remote kernel, đặt `COLAB_RUNTIME_CONFIG_B64` để bootstrap không cần hỏi prompt.

Skeleton để tổ chức side project Santander Product Recommendation. Repository hiện chỉ chứa cấu trúc thư mục — chưa có pipeline, model, dependency hay notebook implementation.

## LightGBM v1 competition submission

After completing a LightGBM v1 training run, create its submission with:

```powershell
python scripts/run_lightgbm_v1_submission.py data/artifacts/runs/<run_id>/model_artifacts.json
```

The command streams `raw/test_ver2.csv` into `interim/test.parquet`, joins each
test customer to product ownership from `interim/train.parquet` at `2016-05-28`,
and invokes the schema-validated generic `predict()` interface for every model.
It writes the final CSV to:

```text
<SANTANDER_DATA_ROOT>/artifacts/runs/<run_id>/submission.csv
```

For the initial Colab smoke run, each of the 24 LightGBM models uses 20 trees,
DuckDB has a 10 GB memory limit, and LightGBM uses four CPU threads. The larger
DuckDB allocation speeds disk-backed panel/sample queries, but leaves little
headroom on a standard 12 GB Colab runtime; lower `runtime.memory_limit` if the
runtime also has other large in-memory workloads.

The baseline is configured with `device_type: cpu`. For a later GPU optimisation
experiment, a CUDA-enabled LightGBM package is required; the standard pip wheel
is not sufficient. After selecting a GPU runtime in Colab and after the regular
bootstrap, run:

```bash
bash scripts/install_lightgbm_cuda_colab.sh
```

Then run the non-bootstrap training cells. If LightGBM was already imported in
the current kernel, restart the runtime, run bootstrap once, and then run the
installer before importing the training pipeline. The installer follows
LightGBM's documented source build with `USE_CUDA=ON`.

## Cấu trúc

```text
.
├── configs/                 # YAML/JSON cho data path, experiment, model
├── data/
│   ├── raw/                 # Dữ liệu Kaggle gốc — không commit
│   ├── interim/             # Dữ liệu tạm sau làm sạch — không commit
│   └── processed/           # Feature/model-ready data — không commit
├── docs/                    # Project plan, data dictionary, báo cáo kỹ thuật
├── notebooks/               # EDA, thử nghiệm, trình bày kết quả
├── reports/
│   └── figures/             # Biểu đồ và hình cho báo cáo
├── src/
│   ├── data/                # Ingest, validation schema, cleaning
│   ├── features/            # Feature engineering và lag features
│   ├── models/              # Baseline, training, inference, evaluation
│   ├── visualization/       # Hàm vẽ dùng lại được
│   └── utils/               # Config, logging, helpers dùng chung
├── tests/
│   ├── unit/                # Test từng hàm/module
│   └── integration/         # Test flow end-to-end với sample nhỏ
├── scripts/                 # Entry point chạy ingest/train/evaluate
└── artifacts/               # Model, metric, prediction sinh ra — không commit
```

Các thư mục rỗng được giữ trong Git bằng `.gitkeep`.

## Cách sử dụng codebase

1. Đặt file CSV Kaggle vào `data/raw/`. Dữ liệu này đã được ignore, không commit lên Git.
2. Khi bắt đầu Checkpoint 1, tạo code đọc dữ liệu trong `src/data/`, feature trong `src/features/`, và notebook khám phá trong `notebooks/`.
3. Đặt đường dẫn dữ liệu và tham số chạy vào `configs/`; không hard-code path hoặc secret trong source code.
4. Bắt đầu Checkpoint 2, để baseline/model tại `src/models/`, script chạy tại `scripts/`, và test tương ứng trong `tests/`.
5. Lưu biểu đồ/báo cáo tĩnh ở `reports/`; lưu output có thể tạo lại (Parquet, metrics, model) ở `data/processed/` hoặc `artifacts/`.

Khi sẵn sàng viết implementation, hãy bổ sung dependency manifest (`requirements.txt` hoặc `pyproject.toml`) dựa trên thư viện thực sự dùng, rồi bổ sung lệnh chạy vào README.


The CLV benchmark reads a retained canonical checkpoint through
`CheckpointStore.resolve`, or an explicitly supplied validated Parquet path.
Prepare missing upstream features in Data Pipeline first; CLV does not run raw
preprocessing. Its new feature functions remain inside the notebook. Intermediate
behavioral summaries and final CLV outputs are saved as full customer-month
candidates under `processed/CLV/<run>/`; future targets remain in separate
benchmark artifacts. DuckDB windows and joins spill to disk with a default
4 GB memory limit, avoiding a full-history Pandas load. LightGBM reads only its
four behavioral inputs and two labels. Extra Parquet snapshots cost disk space
and I/O. Existing checkpoints follow column-existence reuse: after changing
upstream acquisition semantics, refresh the canonical checkpoint in Data Pipeline
first. Notebook-local `FORCE_PROCESS`, `FORCE_PROCESS_FEATURES` and
`FORCE_PROCESS_FUNCTIONS` select CLV contracts; downstream consumers of refreshed
inputs run automatically. These flags do not recompute canonical preprocessing.
`checkpoint_metadata.json` records source lineage, contracts, schema, processing
decisions and benchmark configuration. Published versions remain unchanged.

Selective checkpoint recomputation is configured in `notebooks/Data Pipeline.ipynb`:

```python
REQUIRED_FEATURES = []
FORCE_PROCESS = False
FORCE_PROCESS_FEATURES = ["rfm_frequency", "rfm_recency_months"]
FORCE_PROCESS_FUNCTIONS = []  # alternatively: ["rfm_frequency_sql"]
PUBLISH_CHECKPOINT = True
```

These lists are passed as `force_features` and `force_functions` to
`run_data_pipeline`. `PIPELINE_FUNCTION_OUTPUTS` exposes the valid function names
and their output contracts. Selecting any output forces its owning function's
whole contract. Other functions skip when their outputs already exist; functions
sharing a selected output are all selected (for example, `age` is cleaned by
profile preprocessing and projected by persona logic). Other functions with
missing outputs still run. `force_process=True` overrides the selection
and recomputes all functions. Unknown names fail before processing starts.
`required_features` controls the minimum output schema, not what gets forced.
The manifest records selections, per-function RUN/SKIP, recomputed columns, and
reused columns. Reset the selection lists to `[]` after publishing the update.

Declared inputs in `PIPELINE_FUNCTION_INPUTS` propagate refreshes downstream:
if an earlier RUN replaces a consumer's input, that consumer runs too, transitively.
Selecting `renta_filled`, for example, also refreshes persona's `income_log`.
History builders currently read raw product states directly and calculate their
SQL intermediates internally; stored `prev_`/`acq_` columns are not their inputs.
Changes to shared SQL helpers require selecting their consuming functions;
there is no automatic code-change detection. Contracts run in upstream order.

Use `run_data_pipeline(..., dry_run=True)` or the notebook's preview cell to
inspect RUN/SKIP, reasons, replaced outputs and upstream functions. Preview reads
checkpoint metadata/schema and validates retained snapshots; it never ingests,
computes features, writes or publishes.
With no checkpoint, it plans the initial build; execution still validates source
availability. Execution recalculates the plan against the current checkpoint.
Before publishing, a disk-backed comparison checks that unrelated parent columns
retain both their types and values. This adds one full comparison scan, with the
configured DuckDB memory limit and spill-to-disk, rather than loading snapshots
into Pandas. The manifest retains the reasons, input contracts and executed list.

Each RFM/history output now has its own processing contract. A selected function
materializes a narrow Parquet result, then merges only its owned outputs into a
full candidate. DuckDB retains the same memory limit and spill configuration;
unselected feature builders are skipped. Earlier feature candidates created by
the current run are discarded after the next validated merge, bounding temporary
snapshot disk usage. Fine-grained first-time builds or forcing many functions
require more Parquet I/O and repeated common SQL windows than the former combined
step; normal selective updates avoid recomputing unrelated feature functions.

The sample-selection notebook at
`notebooks/EDA/sample_selection_eda_main_checkpoint.ipynb` uses a single DuckDB
connection with a default 2 GB memory limit and two threads, leaving room for
Python on a 4 GB local runtime. Override these with `SANTANDER_EDA_MEMORY_LIMIT`
and `SANTANDER_EDA_THREADS`; `SANTANDER_DUCKDB_TEMP_DIRECTORY` controls disk spill.
Acquisition EDA projects only keys and product states and aggregates all products
in one query, avoiding 24 repeated window scans and full-panel Pandas copies.
Only small aggregate tables enter Pandas. Disk spill can increase I/O and runtime.
Events compare the nearest previous observed record, including calendar gaps;
first observations have no transition. The notebook resolves the latest valid
train snapshot from `processed/canonical_customer_month/manifest.json` through
`CheckpointStore`; it never falls back to baseline, interim or EDA artifacts.
If absent, run `notebooks/Data Pipeline.ipynb` with `PUBLISH_CHECKPOINT=True`.
Set `SANTANDER_MAIN_CHECKPOINT` to a retained canonical `train.parquet` to lock
an explicit version.

The acquisition-only comparison is implemented as two regular pipelines:
`src/pipeline/lightgbm_acquisition_only_independent.py` and
`src/pipeline/lightgbm_acquisition_only_joint.py`, with YAML under
`configs/baselines/` and matching notebooks under `notebooks/models/`.
Both reuse one immutable prepared train/full-May-validation pack, preserve the
approved canonical static-income policy, and exclude owned products from
candidates. The 48 original columns define the information scope; independent
uses 46 prepared features and joint adds categorical product ID.

To limit memory, preparation uses narrow DuckDB projections, separate Parquet
inputs/targets and disk spill; training writes native LightGBM candidate caches
in batches instead of concatenating the full long dataset in Pandas. Validation
scores in batches and calculates exact average precision one product at a time.
This adds disk I/O and temporary storage. DuckDB defaults to 2 GB/two threads;
its limit does not bound native LightGBM RAM, which is measured by stage. Actual
local/Colab peak RAM has not yet been verified. Configure work storage with
`SANTANDER_MODELING_WORK_ROOT`; Colab defaults to local `/content` storage.

Finalized bundles contain inference models/contracts and versioning/tracking
metadata, excluding raw data, full split datasets and candidate caches. Publish
the bundle to GCS, then run `python scripts/sync_training_runs.py --host` to
pull, verify and import committed runs before starting local MLflow. Test score
and test-set URL start as null and can be published later as revisioned
annotations. Existing pipelines retain their current tracking behavior.
See [the implementation and migration report](docs/acquisition_only_pipeline_implementation.md)
for every added file, storage locations, notebook steps and legacy adaptation.
The new notebooks and semantic tests have been authored but not executed.
