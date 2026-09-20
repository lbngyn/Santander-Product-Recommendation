# Santander Product Recommendation

## Mục tiêu

Xây dựng hệ thống gợi ý sản phẩm ngân hàng từ dữ liệu panel Santander. Target là **additional products**: sản phẩm chuyển từ `0` ở X-1 sang `1` ở X.

Baseline v0 là popularity baseline: xếp hạng sản phẩm theo lượt mua mới trong lịch sử, loại sản phẩm customer đã sở hữu ở X-1, rồi gợi ý tối đa 7 sản phẩm.

## Cấu trúc

```text
configs/       # Hydra config
notebooks/     # Bootstrap và gọi pipeline
scripts/       # Entry point
src/           # Logic ingest, split, preprocessing, model, evaluation
data/          # Cache local/Drive, không commit
artifacts/     # Output run, không commit
```

## Cài đặt local

```powershell
conda activate vsf-side-project
pip install -r requirements.txt
```

Tạo `.env` từ `.env.example`. Không commit `.env`, credential, PAT hoặc Base64 runtime bundle.

```dotenv
GCP_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
GOOGLE_CLOUD_PROJECT=your-project
GCS_BUCKET=your-bucket
GCS_RAW_PREFIX=santander/raw
GCS_CHECKPOINT_PREFIX=santander/checkpoints
SANTANDER_DATA_ROOT=data
SANTANDER_DUCKDB_TEMP_DIRECTORY=data/.duckdb_tmp
MLFLOW_TRACKING_URI=sqlite:///absolute/path/to/mlflow.db
MLFLOW_EXPERIMENT_NAME=santander-baselines
DVC_REMOTE_URL=gs://your-bucket/santander/dvc
```

Nếu dùng `GCP_SERVICE_ACCOUNT_JSON`, không đặt `GOOGLE_APPLICATION_CREDENTIALS` thành một đường dẫn giả. Chỉ dùng biến đó khi file service-account thực sự tồn tại.

## Chạy baseline v0

```powershell
conda run -n vsf-side-project python scripts/run_baseline_v0.py
```

Hydra override, ví dụ ép tạo lại fixed split:

```powershell
conda run -n vsf-side-project python scripts/run_baseline_v0.py split.force_split=true
```

Pipeline chạy theo thứ tự:

```text
GCS/raw cache → ingest → fixed temporal split → preprocessing
→ popularity train/evaluate → artifact local/Drive → upload GCS
```

CSV được ingest theo chunk; DuckDB xử lý Parquet và window function với memory limit/spill directory. Popularity v0 chạy CPU/DuckDB; GPU không được dùng ở baseline này.

## Fixed temporal split

Validation cố định tại `2016-05-28`, dùng chung cho tất cả pipeline:

```text
<SANTANDER_DATA_ROOT>/interim/splits/validation_2016-05-28/
├── train.parquet
├── validation_input.parquet
└── validation_target.parquet
```

- `train.parquet`: record trước validation date.
- `validation_input.parquet`: profile tại X và `prev_<product>` từ X-1; không có 24 product columns tại X.
- `validation_target.parquet`: 24 product columns thật tại X.

Model sau phải fit transform/model trên `train.parquet`, sau đó transform `validation_input.parquet`; không fit statistics trên validation.

## Output run

```text
<SANTANDER_DATA_ROOT>/artifacts/runs/<run_id>/
├── config_resolved.yaml
├── lineage_manifest.json
├── popularity_ranking.parquet
├── validation_predictions.parquet
└── metrics.json
```

Pipeline in tiến độ theo stage. Lịch sử timing được lưu tại `artifacts/runs/pipeline_timing.json`; ETA chỉ có sau ít nhất một run hoàn tất.

## Colab

Chạy bootstrap trong `notebooks/baseline_v0_popularity.ipynb`. Bootstrap lấy config từ Colab Secrets hoặc `COLAB_RUNTIME_CONFIG_B64`, mount Drive, đặt `SANTANDER_DATA_ROOT`, sync source và cài dependency. Branch được bootstrap chọn phải được commit/push trước khi Colab sync.

## Hydra, MLflow và DVC

- Hydra: config chính là `configs/baselines/v0.yaml`; override bằng command line.
- MLflow: log config, metric và artifact tại `MLFLOW_TRACKING_URI`. Dùng SQLite backend, ví dụ `sqlite:///E:/path/to/mlflow.db`; mở UI bằng `mlflow ui --backend-store-uri <URI> --port 5000`.
- DVC: cần khởi tạo/config remote riêng trước khi dùng: `dvc init`, sau đó cấu hình `DVC_REMOTE_URL`.

## Lỗi GCS OAuth `WinError 10013`

Nếu log báo `WinError 10013` khi gọi `oauth2.googleapis.com/token`, Windows, proxy, VPN, antivirus hoặc firewall đang chặn Python kết nối HTTPS outbound. Đây là lỗi network/permission, không phải lỗi dataset.

Kiểm tra:

```powershell
Test-NetConnection oauth2.googleapis.com -Port 443
Test-NetConnection storage.googleapis.com -Port 443
```

Cho phép Python của conda outbound HTTPS/443 hoặc cấu hình proxy theo chính sách mạng. Khi raw/checkpoint đã có local, có thể debug không upload GCS:

```powershell
conda run -n vsf-side-project python scripts/run_baseline_v0.py `
  storage.upload_checkpoints=false storage.upload_artifacts=false
```
