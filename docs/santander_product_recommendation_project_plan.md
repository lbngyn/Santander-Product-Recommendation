# Side Project — Santander Product Recommendation

## TRACK DS — Santander Product Recommendation

---

## Checkpoint 1 (Tuần 1–2): Ingest → EDA → Preprocess

**Mục tiêu:** Hiểu dữ liệu, có pipeline load/clean sạch sẽ, sẵn sàng cho modeling.

### Tuần 1

- Setup môi trường:
  - Python `venv` / `conda`
  - Git repository
  - Jupyter / VS Code
- Tải dataset từ Kaggle, upload lên S3 (free tier).
- Viết script ingest:
  - Đọc CSV
  - Parse dtype
  - Xử lý encoding
  - Load vào DataFrame / Parquet
- EDA sơ bộ:
  - Shape
  - Missing values
  - Cardinality của categorical
  - Distribution của target (24 product columns)

### Tuần 2

- EDA sâu:
  - Correlation giữa các sản phẩm
  - Phân tích theo thời gian (`fecha_dato`)
  - Phân tích theo segment khách hàng:
    - Age
    - Income
    - Seniority
    - `canal_entrada`
- Preprocess:
  - Xử lý missing (`age`, `income`, `renta`)
  - Encode categorical
  - Xử lý imbalance giữa các product label
- Feature engineering cơ bản:
  - Tenure
  - Lag features — khách đã mua sản phẩm gì tháng trước
- Lưu processed data dưới dạng Parquet trên S3.

### Tech dùng

- pandas
- numpy
- matplotlib / seaborn hoặc plotly
- pyarrow
- boto3
- AWS S3

### Deliverable checkpoint 1

- Notebook EDA report:
  - Insight về data
  - Có ít nhất 5–7 finding đáng chú ý
- Data pipeline script tái sử dụng được:
  - `ingest → clean → feature`
  - Chạy được từ đầu đến cuối
- Processed dataset lưu trên S3
- Data dictionary

---

## Checkpoint 2 (Tuần 3–4): Baseline (rule-based) → ML Model đầu tiên

**Mục tiêu:** Có baseline để so sánh, và model ML đầu tiên hoạt động end-to-end.

### Tuần 3

- Định nghĩa metric đánh giá phù hợp bài toán recommendation:
  - MAP@7 — đúng metric gốc của competition
  - Hoặc Precision@K, Recall@K
- Xây baseline rule-based, ví dụ:
  - Recommend sản phẩm phổ biến nhất theo segment
  - Recommend sản phẩm khách chưa có nhưng người cùng segment có nhiều nhất
- Đo baseline trên validation set để có số benchmark.

### Tuần 4

- Chuyển bài toán thành:
  - Multi-label classification
  - Hoặc binary classification per-product
- Prototype model
- Validation
- So sánh với baseline

### Tech dùng

- scikit-learn
- XGBoost / LightGBM
- MLflow
  - Tracking experiment
  - Có thể chạy local hoặc trên EC2 free tier

### Deliverable checkpoint 2

- Báo cáo baseline (rule-based) với số liệu benchmark rõ ràng
- Model ML đầu tiên có MAP@7 (hoặc metric đã chọn) tốt hơn baseline
- Có log qua MLflow
- Bảng so sánh:
  - Baseline
  - Model v1

---

## Checkpoint 3 (Tuần 5–6): Optimize Model + Tổng kết

**Mục tiêu:** Cải thiện model về accuracy và tối ưu về latency/resource, trình bày kết quả.

### Tuần 5

- Feature engineering nâng cao:
  - Interaction features
  - Embedding cho categorical
- Hyperparameter tuning:
  - Optuna
  - GridSearch
- Thử model thứ 2 để so sánh:
  - Ví dụ: LightGBM vs Neural Net đơn giản
- Đo latency inference:
  - Batch
  - Single request

### Tuần 6

- Tối ưu:
  - Giảm số feature bằng feature importance / SHAP
  - Giảm model size
  - Thử quantization / pruning nếu dùng Neural Network
- Đo lại accuracy vs latency trade-off
- Chọn model final
- Viết báo cáo tổng kết + demo:
  - Notebook
  - Hoặc slide

### Tech dùng

- SHAP
- Optuna
- MLflow

### Deliverable checkpoint 3

- Model final với báo cáo so sánh đầy đủ:
  - Baseline
  - Model v1
  - Model v2 optimized
- SHAP explain
- Báo cáo latency / throughput benchmark
- Slide / notebook trình bày toàn bộ quá trình:
  - `ingest → EDA → preprocess → model → optimize`
