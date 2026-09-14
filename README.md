# Santander Product Recommendation

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
