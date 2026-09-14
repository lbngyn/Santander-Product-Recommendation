# Santander Product Recommendation

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
