## TRACK DS — Customer Segmentation: RFM, CLV, Affinity, Persona (Track A)
 
### Checkpoint 1 (Tuần 1-2): Ingest → EDA → RFM-proxy + Customer Lifecycle
 
**Mục tiêu:** Hiểu dữ liệu, có pipeline load/clean sạch sẽ; dựng được phân khúc RFM-proxy và gán trạng thái lifecycle.
 
**Lưu ý dataset:** Santander Product Recommendation là dữ liệu **snapshot hàng tháng** (mỗi dòng = 1 khách hàng tại 1 tháng `fecha_dato`, kèm 24 cột nhị phân thể hiện đang sở hữu sản phẩm nào), **không phải dữ liệu giao dịch/invoice**. Vì vậy RFM/CLV không áp dụng trực tiếp như retail mà cần định nghĩa qua **"sự kiện phát sinh sản phẩm mới"** (product acquisition event — khi một cột sản phẩm chuyển từ 0 → 1 giữa 2 tháng liên tiếp), dùng làm proxy cho "transaction".
 
| Tuần | Task |
| -------- | -------- |
| Tuần 1 | - Setup môi trường (Python venv/conda, Git repo, Jupyter/VS Code)<br>- Tải dataset từ Kaggle, upload lên **S3 (free tier)**<br>- Viết script ingest: đọc `train_ver2.csv` (~13GB) → tối ưu dtype, xử lý encoding, cân nhắc đọc theo chunk/`dask` nếu máy yếu, load vào DataFrame/Parquet<br>- EDA sơ bộ: shape, missing values (`renta`, `age`, `cod_prov`, `indrel_1mes`...), cardinality categorical (`canal_entrada`, `segmento`, `pais_residencia`), phân bố số sản phẩm sở hữu/khách |
| Tuần 2 | - EDA sâu: biến đổi số sản phẩm theo thời gian (`fecha_dato`), theo segment (`segmento`, `canal_entrada`, `antiguedad`), tỷ lệ khách rời đi (không còn xuất hiện ở tháng sau)<br>- **Tạo bảng "product acquisition event"**: diff 24 cột sản phẩm giữa 2 tháng liên tiếp/khách để xác định thời điểm phát sinh sản phẩm mới<br>- **RFM-proxy**: Recency = số tháng từ lần phát sinh sản phẩm gần nhất; Frequency = số lần phát sinh sản phẩm trong lịch sử; Monetary = proxy (số sản phẩm đang sở hữu hiện tại hoặc tier theo `renta`) — so sánh 3 cách chia tier (quantile, k-means, rule-based) và đánh giá độ ổn định<br>- **Lifecycle**: gán trạng thái new → active → at-risk → churned → win-back dựa trên `ind_actividad_cliente`, tần suất phát sinh sản phẩm, và việc khách biến mất khỏi dữ liệu (rời ngân hàng) hoặc `indfall` |
 
**Tech dùng:** pandas, numpy, dask hoặc chunked read (cho file lớn), matplotlib/seaborn hoặc plotly, pyarrow, boto3, AWS S3, scikit-learn (k-means).
 
**Deliverable checkpoint 1:**
 
- Notebook EDA report (ít nhất 5-7 finding đáng chú ý)
- Data pipeline script tái sử dụng được (ingest → clean → product-event → RFM-proxy/lifecycle) chạy được từ đầu đến cuối, xử lý được file dung lượng lớn
- RFM-proxy tiering + lifecycle assignment, kèm tài liệu định nghĩa "product acquisition event", định nghĩa segment + rule chuyển trạng thái (viết dưới dạng đặc tả feature cho Feature Store)
- Processed dataset lưu trên S3, có data dictionary
 
---
 
### Checkpoint 2 (Tuần 3-4): CLV-proxy + Affinity Matrix
 
**Mục tiêu:** Ước lượng giá trị vòng đời khách hàng (qua proxy) và xây dựng affinity matrix giữa các sản phẩm cho cross-sell.
 
| Tuần | Task |
| -------- | -------- |
| Tuần 3 | - **CLV probabilistic**: dùng "product acquisition event" làm input tần suất/thời điểm cho BG/NBD (thư viện `lifetimes`/`pymc-marketing`); vì thiếu monetary thật, thử Gamma-Gamma với monetary proxy (`renta` hoặc số sản phẩm), nêu rõ giới hạn của proxy này trong báo cáo<br>- **CLV ML**: gradient boosting regression dự báo số sản phẩm/giá trị proxy trong 6-12 tháng tới; so sánh với mô hình xác suất, đặc biệt ở nhóm khách mới (ít lịch sử)<br>- Phân tích sai số CLV-proxy theo segment RFM (kết nối với Checkpoint 1) |
| Tuần 4 | - **Affinity matrix** sản phẩm × sản phẩm: co-occurrence giữa 24 sản phẩm (khách hàng nào sở hữu sản phẩm này thường sở hữu thêm sản phẩm nào)<br>- Association rules (Apriori/FPGrowth, `mlxtend`) so với embedding (item2vec) để gợi ý sản phẩm tiếp theo<br>- Đo chất lượng affinity (support/confidence/lift hoặc recall@k cho bài toán dự đoán sản phẩm mới khách sẽ mở — sát với mục tiêu gốc của competition) |
 
**Tech dùng:** scikit-learn, XGBoost/LightGBM, `lifetimes`/`pymc-marketing`, `mlxtend`, MLflow (tracking experiment — có thể chạy local hoặc trên EC2 free tier).
 
**Deliverable checkpoint 2:**
 
- Báo cáo CLV-proxy: so sánh BG/NBD + Gamma-Gamma vs gradient boosting, có benchmark trên validation set, phân tích sai số theo segment RFM, nêu rõ giới hạn của monetary proxy
- Affinity matrix (sản phẩm × sản phẩm) + association rules/item2vec, đánh giá chất lượng cho gợi ý sản phẩm tiếp theo
- Bảng so sánh và log experiment qua MLflow
 
---
 
### Checkpoint 3 (Tuần 5-6): Persona + Tổng kết
 
**Mục tiêu:** Cụm hóa thành persona có thể diễn giải được cho business, và trình bày kết quả end-to-end.
 
| Tuần | Task |
| -------- | -------- |
| Tuần 5 | - **Persona**: gom cụm khách hàng từ RFM-proxy + CLV-proxy + affinity/product portfolio + demographics (`age`, `renta`, `antiguedad`, `canal_entrada`, `segmento`); chọn số cụm và phương pháp (k-means/hierarchical)<br>- Đặt tên và diễn giải persona để đội business dùng được (5-8 persona card) kèm đặc trưng định lượng<br>- Kiểm tra độ ổn định và tính diễn giải được của persona |
| Tuần 6 | - Tổng hợp toàn bộ pipeline: ingest → EDA → RFM-proxy/lifecycle → CLV-proxy → affinity → persona<br>- Đóng gói feature spec (naming convention, definition, TTL) theo chuẩn Feature Store, như một bài tập best-practice để bàn giao cho hệ thống production sau này<br>- Viết báo cáo tổng kết + demo (notebook hoặc slide) |
 
**Tech dùng:** scikit-learn, SHAP (cho CLV model), pandas, matplotlib/seaborn.
 
**Deliverable checkpoint 3 (cuối track):**
 
- 5-8 Persona card kèm đặc trưng định lượng, có diễn giải business
- Báo cáo tổng kết end-to-end (RFM-proxy → lifecycle → CLV-proxy → affinity → persona), kèm benchmark và hạn chế (đặc biệt là giới hạn của các proxy dùng thay transaction/monetary thật)
- Feature spec theo chuẩn Feature Store (schema, naming, versioning) — tài liệu mẫu, không phụ thuộc hệ thống nào cụ thể trong batch này
- Slide/notebook trình bày toàn bộ quá trình trước hội đồng (15-20 phút)
 
---
 
### Ghi chú về dữ liệu và scope
 
- **Dataset chính:** Santander Product Recommendation (Kaggle) — dữ liệu snapshot hàng tháng theo khách hàng, 24 sản phẩm ngân hàng, không có transaction/invoice/monetary thật. Track A đòi hỏi tự định nghĩa proxy (product acquisition event, monetary proxy) để áp dụng RFM/CLV — đây cũng là một phần kỹ năng cần rèn (data modeling khi thiếu dữ liệu lý tưởng).
- **Dung lượng dữ liệu:** `train_ver2.csv` khá lớn (~13GB) — cần cân nhắc sampling, đọc theo chunk, hoặc dùng `dask`/Parquet ngay từ Checkpoint 1 để pipeline chạy được trên máy cá nhân/free tier.
- **Affinity:** vì bài toán đã có sẵn 24 cột sản phẩm dạng basket, affinity matrix ở đây tự nhiên và sát với mục tiêu gốc của competition (dự đoán sản phẩm mới) hơn so với market-basket truyền thống.
- **Tính chuyển giao:** toàn bộ code + tài liệu đóng gói để đội chính thức chạy lại trên dữ liệu thật sau này (chỉ cần đúng schema là thay thế được). Nghiêm cấm dùng dữ liệu khách hàng thật.
- **Không phụ thuộc track MLE:** batch này track DS (customer analytics) và track MLE (GeoLife — home/office/POI) dùng 2 dataset độc lập, không có pipeline nối tiếp nhau. Feature spec ở Checkpoint 3 là bài tập rèn kỹ năng đóng gói/bàn giao theo chuẩn Feature Store, không phải input thực tế cho track MLE B1.
 
 
## TRACK MLE — Home/Office/POI Inference (Track B1)
 
### Checkpoint 1 (Tuần 1-2): Stay-point detection → Home/Office classifier + API Spec
 
**Mục tiêu:** Có model location cơ bản chạy được, và API contract rõ ràng — ưu tiên tốc độ, không tối ưu accuracy.
 
| Tuần | Task |
| -------- | -------- |
| Tuần 1 | - Setup repo, môi trường, terraform<br>- Load dataset Microsoft GeoLife, clean GPS trace (filter noise, speed-based filtering)<br>- Xử lý timezone: convert timestamp trong file `.plt` (lưu theo GMT) sang local time (đa số dữ liệu thu thập tại Bắc Kinh, UTC+8) trước khi áp bất kỳ heuristic theo giờ nào<br>- Stay-point detection: implement thuật toán phát hiện điểm dừng (time-threshold + distance-threshold)<br>- Baseline heuristic: đêm = home, giờ hành chính = office |
| Tuần 2 | - Định nghĩa API spec (OpenAPI/Swagger): endpoint `/classify/{user_id}`, input/output schema (lat/lng sequence → home/office/POI + confidence), error handling, versioning trong URL (`/v1/classify`)<br>- Định nghĩa rõ cách tính `confidence` (vd: tỷ lệ thời gian ở tại location trong khung giờ kỳ vọng, hoặc mật độ điểm/số lần ghé của cluster từ DBSCAN) — vì bài toán không có nhãn nên confidence chỉ mang tính heuristic, không phải xác suất mô hình học có giám sát<br>- Viết doc spec đầy đủ (request/response example, status code)<br>- Review spec cùng mentor trước khi code |
 
**Tech dùng:** pandas, numpy, scikit-learn (DBSCAN), h3/geohash, FastAPI (để định nghĩa spec dễ tự sinh Swagger docs), OpenAPI.
 
**Deliverable checkpoint 1:**
 
- Stay-point detection + baseline heuristic chạy được (không cần tối ưu)
- File OpenAPI spec (`.yaml`) hoàn chỉnh, review được
- Repo có cấu trúc rõ ràng (model/, api/, tests/)
- Ghi chú đánh giá: GeoLife không có nhãn home/office chuẩn, nên đánh giá bằng cách lấy mẫu một số user, kiểm tra thủ công tính hợp lý (địa điểm home/office suy luận có khớp với pattern di chuyển không) thay vì đo accuracy tuyệt đối
 
---
 
### Checkpoint 2 (Tuần 3-4): Deploy Model + Versioning + Deploy Strategy
 
**Mục tiêu:** Model chạy như service thật trên AWS, có version control và hiểu 3 chiến lược deploy.
 
| Tuần | Task |
| -------- | -------- |
| Tuần 3 | - Implement API theo spec đã định nghĩa bằng FastAPI<br>- Đóng gói bằng Docker<br>- Deploy lên AWS free tier: EC2 (t2.micro) hoặc Lambda + API Gateway (nếu model nhẹ)<br>- Setup model versioning: MLflow Model Registry hoặc đơn giản là naming convention + S3 (model-v1, model-v2)<br>- Model v1 = heuristic theo giờ áp trực tiếp trên từng stay-point riêng lẻ; v2 = dùng DBSCAN gộp các stay-point gần nhau thành 1 location trước khi áp heuristic (giảm nhiễu do GPS lệch vị trí giữa các lần ghé) |
| Tuần 4 | - Học và mô phỏng 3 chiến lược deploy:<br>- **Shadow**: traffic gửi đến cả model cũ + mới, chỉ log kết quả model mới, không trả về user<br>- **Canary**: route % nhỏ traffic (vd 10%) sang model mới<br>- **Blue-green**: 2 environment riêng, switch traffic toàn bộ khi model mới pass test<br>- Implement được ít nhất 1 trong 3 (khuyến nghị Canary vì dễ mô phỏng với API Gateway/ALB weighted routing hoặc đơn giản là random routing logic trong code) |
 
**Tech dùng:** Docker, FastAPI, MLflow, AWS EC2/Lambda/API Gateway, GitLab CI/CD hoặc GitHub Actions cho CI/CD pipeline deploy.
 
**Deliverable checkpoint 2:**
 
- API đang chạy thật trên AWS, có thể gọi qua public/internal endpoint
- Có ít nhất 2 version model (heuristic vs clustering), quản lý qua registry
- Demo được 1 deploy strategy (canary/shadow/blue-green) hoạt động thực tế, kèm giải thích 3 chiến lược (điểm khác biệt, khi nào dùng)
 
---
 
### Checkpoint 3 (Tuần 5-6): Serving Strategy + Monitoring/Drift/Latency
 
**Mục tiêu:** Hiểu sync/async serving, và có hệ thống giám sát model trong production.
 
| Tuần | Task |
| -------- | -------- |
| Tuần 5 | - So sánh Sync (request-response trực tiếp, dùng cho real-time classify) vs Async (queue-based, dùng SQS + worker, phù hợp batch classify cho tập user lớn)<br>- Implement thử 1 flow async đơn giản: request → SQS → Lambda/worker xử lý → lưu kết quả → client poll hoặc callback<br>- Đo latency của cả 2 approach |
| Tuần 6 | - Setup monitoring: log request/response, latency (p50/p95/p99), error rate<br>- Setup drift detection cơ bản (so sánh distribution location/trajectory theo thời gian — location drift); vì GeoLife là dữ liệu tĩnh (2007-2012), mô phỏng drift bằng cách replay trajectory theo thứ tự thời gian như traffic thật, không phải drift từ traffic sản xuất thực tế<br>- Dashboard: CloudWatch (AWS free tier) hoặc Grafana + Prometheus nếu tự host<br>- Tổng kết + demo toàn bộ hệ thống |
 
**Tech dùng:** AWS SQS, CloudWatch, Prometheus/Grafana (nếu muốn tự host, docker compose).
 
**Deliverable checkpoint 3 (cuối track):**
 
- So sánh sync vs async có số liệu latency thực tế
- Dashboard monitoring hiển thị latency + basic drift alert
- Demo end-to-end: từ request → model serving → log → monitor, kèm slide tổng kết toàn bộ 6 tuần (API spec → deploy → serving → monitoring)
 
---
 
### Bổ sung cho Track B1 (nằm trong các checkpoint trên)
 
| Component | Vị trí | Deliverable thêm |
| -------- | -------- | -------- |
| Stay-point detection | Checkpoint 1 | Thuật toán + threshold tuning |
| Home/office heuristic vs DBSCAN | Checkpoint 1-2 | Benchmark 2 method |
| POI categorization (bonus) | Checkpoint 2 | Reverse geocode / h3 cell mapping |
| Privacy risk analysis | Xuyên suốt | Mục trong báo cáo (geohash làm thô, k-anonymity) — bám theo yêu cầu của doc de-bai-nghien-cuu |
 
**Lưu ý scope:**
 
- Microsoft GeoLife làm dataset chính (GPS trace dày, phù hợp stay-point detection).
- Gowalla/Brightkite chỉ dùng ở mức so sánh (chứng minh check-in data không suy luận được home/office), không làm end-to-end.
- Kèm mục phân tích rủi ro quyền riêng tư và kỹ thuật giảm nhạy cảm (geohash làm thô, k-anonymity).
 