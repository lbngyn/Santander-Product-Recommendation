# Santander Product Recommendation

# Overview dataset: 
- **Provider**: Santander Bank → Product là các dịch vụ liên quan tới tài chính (như là các loại tài khoản, các dịch vụ vay, thuế, credit card)
- **Dataset**:
    - Thời gian: `28-01-2015` → `28-05-2016`
    - Bản ghi các dịch vụ mà người dùng đã đăng kí của cty cùng với các thông tin cá nhân của người dùng.
- **Target**: tìm các **additional product** của từng customer tại thời điểm `28-06-2016`

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

## Acquisition target features

## Customer-profile preprocessing

`src/data/preprocessing.py` implements the cleaning decisions from
`docs/customer_profile_feature_deep_eda.md` before any encoding or feature
engineering. It is intentionally limited to preprocessing:

- keeps observed `age` values unchanged, including values outside 18--90;
- converts negative `antiguedad` sentinels and negative `renta` to null;
- uses only an earlier row of the same customer for history filling;
- fits residual numeric medians and the upper `renta` clipping threshold on a
  caller-supplied training partition;
- normalises `indrel_1mes` values `1` and `1.0`, retains categorical null as
  `__MISSING__`, and drops baseline columns `conyuemp`, `nomprov`,
  `ult_fec_cli_1t`, and `tipodom`.

Use `fit_profile_preprocessing()` and `transform_customer_profiles()` inside
each time split. `preprocess_customer_profiles()` is the convenience function
for producing the final `data/processed/train.parquet` and
`data/processed/test.parquet` artifacts. It persists the fitted statistics to
`profile_preprocessing_stats.json` beside the outputs.

The implementation uses DuckDB window functions and Parquet `COPY`, including
disk spill through a configurable temporary directory. It never materialises
the customer panel in pandas/RAM. This keeps peak memory bounded by DuckDB's
configured memory limit at the cost of a disk-backed sort and an additional
Parquet write. Do not use statistics fit on the full train file to evaluate an
earlier time-validation fold; fit them only on that fold's historical train
partition.

## Baseline v0 codebase and reproducibility

The source is organised by responsibility so that alternative preprocessing and
feature strategies can coexist without duplicating an entire pipeline:

```text
src/
├── ingestion/       # public ingest namespace; legacy src/data remains compatible
├── preprocessing/   # cleaning and missing-value strategies
├── features/        # acquisition labels and future feature groups
├── splits/          # time-based split contracts
├── models/          # training implementations
├── evaluation/      # ranking metrics and evaluation protocols
├── tracking/        # Git/config/data lineage and optional MLflow logging
└── pipeline/        # thin composition of the stages above
```

The first runnable experiment is [baseline v0](configs/baselines/v0.yaml).
It starts at the raw files: **raw CSV → interim Parquet → preprocessing**. It
retains product and categorical columns
without encoding or feature engineering, cleans negative `antiguedad`/`renta`,
clips only the upper tail of `renta`, and fills profile values using records of
the same customer. Numeric gaps with an observed record on each side use the
arithmetic mean; categorical/date values use the closest customer record;
residual categories become `__MISSING__` and residual numerics use a train-fit
median. Its exact stage descriptions are stored alongside its parameters in
the YAML config.

Run it after interim checkpoints exist:

```powershell
pip install -r requirements.txt
python scripts/run_baseline_v0.py --config configs/baselines/v0.yaml
```

It writes `data/interim/baseline_v0/`, `data/processed/baseline_v0/`, and a per-run lineage manifest to
`artifacts/runs/<run_id>/lineage_manifest.json`. The manifest contains the
resolved config, SHA-256 fingerprints of inputs/outputs, and Git commit,
branch, and dirty-worktree state. Enable `tracking.mlflow.enabled` in the
config to log the same config and manifest to local MLflow (`mlruns/`). DVC can
reproduce the declared stage with `dvc repro` after `dvc init`; configure a
remote (for example the existing GCS bucket) before sharing large artifacts.

Warning: `allow_future_values_within_train: true` is an offline data-repair
setting requested for baseline v0. It averages a missing training record with
an observation after it. Set it to `false` before time-based validation or
production scoring; otherwise it leaks future customer information. Test data
is always transformed past-only.

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

## Cấu trúc

```text
.
├── configs/                 # Config versioned theo baseline/experiment
├── data/
│   ├── raw/                 # Dữ liệu Kaggle gốc — không commit
│   ├── interim/             # Dữ liệu tạm sau làm sạch — không commit
│   └── processed/           # Feature/model-ready data — không commit
├── docs/                    # Project plan, data dictionary, báo cáo kỹ thuật
├── notebooks/               # EDA, thử nghiệm, trình bày kết quả
├── reports/
│   └── figures/             # Biểu đồ và hình cho báo cáo
├── src/
│   ├── data/                # Legacy-compatible ingest/schema implementation
│   ├── ingestion/           # Public namespace của ingestion
│   ├── preprocessing/       # Cleaning và missing-value strategies
│   ├── features/            # Acquisition labels và feature engineering
│   ├── splits/              # Time-based data splits
│   ├── models/              # Model training (chưa có model ở v0)
│   ├── evaluation/          # Evaluation protocol/metrics
│   ├── tracking/            # Git/config/data lineage + MLflow adapter
│   └── pipeline/            # Composition của các domain stage
├── tests/
│   ├── unit/                # Test từng hàm/module
│   └── integration/         # Test flow end-to-end với sample nhỏ
├── scripts/                 # Entry point chạy ingest/train/evaluate
└── artifacts/               # Model, metric, prediction sinh ra — không commit
```

Các thư mục rỗng được giữ trong Git bằng `.gitkeep`.

## Cách sử dụng codebase

1. Đặt file CSV Kaggle vào `data/raw/`. Dữ liệu này đã được ignore, không commit lên Git.
2. Đặt đường dẫn dữ liệu và tham số chạy vào `configs/`; không hard-code path hoặc secret trong source code.
3. Đặt strategy mới trong domain phù hợp (ví dụ `src/preprocessing/`), chọn strategy qua config, rồi để manifest/MLflow ghi lại run.
4. Lưu biểu đồ/báo cáo tĩnh ở `reports/`; lưu output có thể tạo lại ở `data/processed/` hoặc `artifacts/`.
