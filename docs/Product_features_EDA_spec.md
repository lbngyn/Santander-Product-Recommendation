# SPEC — Product History EDA

## 0. Mục tiêu chung

Phần này **chỉ tập trung vào 24 product columns và lịch sử product theo `ncodpers × fecha_dato`**.

Mục tiêu cuối cùng:

$$
\text{Product history up to }t-1
\rightarrow
\text{Product acquisition/state at }t
$$

EDA phải phục vụ ba việc:

1. Hiểu behavior của 24 product theo thời gian.
2. Tạo và kiểm chứng các representation/feature mới từ product history.
3. Xác định feature nào có khả năng hữu ích cho việc dự đoán/recommend **product mới**.

Không đưa demographic/profile như `age`, `renta`, `segmento`, `canal_entrada`, `ind_actividad_cliente` vào phần này. Những relationship đó để dành cho **Cross EDA**.

---

# I. FORMAT BẮT BUỘC CHO MỖI EDA CELL

Không được tạo một cell chỉ có code/chart mà không giải thích mục đích.

Mỗi analysis unit trong notebook phải có cấu trúc:

```text id="3e23my"
### [EDA title]

Hypothesis
----------
Giả thuyết đang muốn kiểm tra là gì?

Objective
---------
EDA này muốn trả lời câu hỏi cụ thể nào?

Logic
-----
- Input nào được sử dụng?
- Population/sample nào được xét?
- Derived metric được tính như thế nào?
- Temporal cutoff là gì?
- Missing/gap được xử lý như thế nào?

Code
----
<code cell>

Result / Visualization
----------------------
<table / chart>

Insight
-------
- Quan sát trực tiếp từ kết quả.
- Không suy diễn causal.
- Có số liệu cụ thể nếu phù hợp.

Impact
------
Insight này ảnh hưởng thế nào tới:
- target/model;
- imbalance;
- preprocessing;
- feature engineering;
- hoặc validation.

Next step / Feature direction
-----------------------------
- Có candidate feature nào không?
- Cần EDA tiếp gì để kiểm chứng?
- Keep / investigate / no action.
```

Ví dụ không được viết:

> Recency rất quan trọng.

Mà phải viết kiểu:

> Future acquisition rate giảm theo `months_since_last_acquisition` từ ... đến ... trong vùng có đủ sample support. Điều này cho thấy Recency có association với future acquisition và là candidate historical feature. Cần kiểm chứng incremental value trên temporal validation.

**Insight phải được sinh từ output thực tế**, agent không được viết conclusion cố định trước khi chạy cell.

---

# II. GLOBAL RULES

### Rule 1 — Temporal direction

Đây là rule quan trọng nhất.

Đối với sample tại `t`:

$$
X^{product}_t
=
f(ProductHistory_{\leq t-1})
$$

và:

$$
Y_t=ProductState_t
$$

hoặc khi EDA acquisition:

$$
Y_t=Acquisition_t
$$

Không được dùng product information tại `t` để tạo predictor cho acquisition/state tại `t`.

Ví dụ:

**Đúng**

$$
Recency_{t-1}\rightarrow Acquisition_t
$$

**Sai**

$$
Recency_t\rightarrow Acquisition_t
$$

vì `Recency_t=0` có thể được tạo trực tiếp bởi acquisition tại `t`.

---

### Rule 2 — Chỉ coi là monthly transition khi gap = 1

Với:

```text id="47xg7q"
previous observed record = March
current record = June
```

không được coi March là `lag1` của June.

Acquisition/drop chỉ hợp lệ khi:

$$
date\_diff(month,t_{prev},t)=1
$$

---

### Rule 3 — Phân biệt state và event

Phải dùng terminology nhất quán:

```text id="iw08xk"
state:
product_p = 0/1

acquisition:
0 → 1

drop:
1 → 0

stable non-owner:
0 → 0

stable owner:
1 → 1
```

Không gọi raw product state là acquisition.

---

### Rule 4 — Không tự động coi missing product state = 0

`NULL` không mặc định đồng nghĩa với "not owned".

Nếu một analysis cần `COALESCE(...,0)`, agent phải:

* giải thích assumption;
* kiểm tra missing trước;
* hoặc chứng minh dataset cho phép assumption đó.

---

### Rule 5 — Effect luôn đi cùng Support

Mọi rate/probability phải đi kèm denominator/count.

Không được kết luận từ:

```text id="gqlujf"
P(B|A) = 60%
```

mà không biết:

```text id="dmlg1b"
A acquisition count = 5
```

Mọi table relationship nên có:

```text id="0jvymh"
count
rate
```

và visualization phải kiểm tra sparse tail/outlier.

---

### Rule 6 — Association ≠ causation

Chỉ dùng:

> associated with / có relationship / có xu hướng

Không viết:

> A làm customer mua B

chỉ dựa trên observational EDA.

---

### Rule 7 — Không tạo feature chỉ vì có thể tạo

Quy trình:

$$
Hypothesis
\rightarrow EDA
\rightarrow Evidence
\rightarrow CandidateFeature
\rightarrow Validation
$$

Không:

$$
Idea\rightarrow Create100Features
$$

---

# KHỐI 1 — PRODUCT STATE EDA

## Mục tiêu

Hiểu **trạng thái sở hữu của 24 product** trước khi biến chúng thành event/history features.

Trả lời:

> Product nào phổ biến? Product ownership thay đổi thế nào theo thời gian? Portfolio của customer có cấu trúc như thế nào? Product state có persistence cao không?

---

## EDA 1.1 — Product ownership/popularity over time

### Hypothesis

Mức độ sở hữu các product không đồng đều và có thể thay đổi theo thời gian.

### Objective

Đo:

$$
P(State_{p,t}=1)
$$

cho từng:

```text id="wt8qor"
product × checkpoint_month
```

### Logic

Mỗi `fecha_dato`:

* customer count;
* holders từng product;
* ownership rate;
* net holder change so với tháng trước.

### Output

Table:

```text id="1x5jq7"
month
product
customer_count
customers_holding
ownership_rate
net_holders_change
```

Visualization:

* top product ownership trends;
* acquisition-independent net holder changes;
* latest product popularity ranking.

### Impact

Xác định:

* imbalance giữa products;
* temporal trend;
* product popularity prior;
* liệu time features/product-level priors đáng investigate không.

---

# EDA 1.2 — Portfolio depth distribution

### Hypothesis

Customers có độ rộng portfolio rất khác nhau.

### Objective

Phân tích:

$$
M_{i,t}=\sum_p State_{i,p,t}
$$

### Logic

Tính số products owned/customer/checkpoint.

Phân tích:

* overall distribution;
* distribution theo checkpoint;
* mean/median/quantiles.

### Output

Histogram/bar + temporal trend.

### Feature direction

Candidate:

```text id="nvbfds"
products_owned_count_{t-1}
```

Nhưng chưa kết luận predictive cho tới Khối 3.

---

# EDA 1.3 — Product state transition/persistence

### Hypothesis

State tại `t-1` có association mạnh với state tại `t`.

### Objective

Với từng product tính:

```text id="en88s2"
0→0
0→1
1→0
1→1
```

chỉ trên consecutive months.

### Metrics

Đặc biệt:

$$
P(State_t=1|State_{t-1}=1)
$$

và:

$$
P(State_t=1|State_{t-1}=0)
$$

### Output

24-product transition table + visualization.

### Impact

Nếu persistence cao:

```text id="8eglf8"
24 × product_state_lag1
```

là candidate feature quan trọng.

---

# KHỐI 2 — PRODUCT EVENT EDA

Đây là bước chuyển:

$$
ProductState
\rightarrow
CustomerBehavior
$$

## EDA 2.1 — Build Product Acquisition Event table

Đây là **derived analytical dataset**, không chỉ là visualization.

### Definition

Với customer \(i\), product \(p\):

$$
Acq_{i,p,t}
=
I(State_{i,p,t-1}=0 \land State_{i,p,t}=1)
$$

và:

$$
Drop_{i,p,t}
=
I(State_{i,p,t-1}=1 \land State_{i,p,t}=0)
$$

### Logic

Chỉ tạo event khi:

$$
gap(t-1,t)=1
$$

### Agent phải validate

* số valid transitions;
* số excluded transitions do gap;
* missing product state;
* acquisition/drop count sanity checks.

Derived event table này được reuse cho toàn bộ EDA sau.

---

# EDA 2.2 — Acquisition imbalance by product

### Hypothesis

Probability `0→1` khác mạnh giữa 24 products và acquisition là rare event.

### Objective

Với mỗi product tính:

```text id="l8ujxj"
eligible = previous_state = 0
positive = 0→1
negative = 0→0

acquisition_rate
negative_to_positive_ratio
```

### Output

Table + ranked chart.

### Impact

Thông tin trực tiếp cho:

* class imbalance;
* product-specific weighting;
* sampling;
* loss function;
* model-per-product strategy.

---

# EDA 2.3 — Acquisition/drop over time

### Hypothesis

Acquisition behavior của từng product có temporal variation.

### Objective

Theo:

```text id="qxig2i"
product × month
```

tính:

```text id="h0vj7i"
acquisitions
eligible opportunities
acquisition_rate
drops
drop_rate
```

### Output

Temporal trends.

### Impact

Đánh giá candidate:

```text id="xqrr2r"
month
time_index
historical product popularity/acquisition prior
```

Không khẳng định seasonality chỉ từ 17 checkpoints.

---

# EDA 2.4 — Number of acquisitions per customer-month

### Hypothesis

Một số customer acquire nhiều products trong cùng một checkpoint.

### Objective

Tính:

$$
N_{i,t}^{acq}
=
\sum_p Acq_{i,p,t}
$$

### Output

Distribution:

```text id="1ad3jv"
0
1
2
3
...
```

### Impact

Nếu multi-product acquisitions tồn tại đáng kể → motivation trực tiếp cho **same-month product relationship EDA** ở Khối 4.

---

# KHỐI 3 — HISTORICAL BEHAVIOR EDA

Khối này chuyển event history thành behavioral features.

Mọi predictor phải tính **as-of `t-1`** khi so với outcome tại `t`.

---

# EDA 3.1 — Recency → future acquisition

### Hypothesis

Khoảng thời gian từ acquisition gần nhất có association với khả năng acquire tiếp.

Define:

$$
R_{i,t-1}
=
months\ since\ last\ acquisition
$$

### Test

$$
R_{t-1}
\rightarrow
AnyAcquisition_t
$$

và nếu khả thi:

$$
R_{t-1}
\rightarrow
Acquisition_{p,t}
$$

### Logic

Group theo exact recency:

```text id="7w5x86"
0
1
2
3
...
NULL
```

Không biến `NULL` thành một numeric arbitrary value.

### Output

Với từng recency:

```text id="f41yeu"
records
customers
future_acquisition_count
future_acquisition_rate
```

* line chart + support count.

### Feature direction

Nếu relationship ổn định:

```text id="4rcyqk"
months_since_last_acquisition
```

Không cần `never_acquired_flag` nếu model xử lý missing phù hợp.

---

# EDA 3.2 — Frequency → future acquisition

### Hypothesis

Customers historically acquire nhiều products có future behavior khác.

Define:

$$
F_{i,t-1}
=
\sum_{\tau\leq t-1}N^{acq}_{i,\tau}
$$

### Test

$$
F_{t-1}\rightarrow AnyAcquisition_t
$$

### Output

Exact/binned F vs future acquisition rate + support.

### Feature direction

```text id="3rj6uc"
cumulative_acquisitions
```

---

# EDA 3.3 — Portfolio depth → future acquisition

Bây giờ quay lại `M`, nhưng đúng temporal direction.

### Hypothesis

Số product customer đã sở hữu ảnh hưởng đến propensity acquire thêm.

### Test

$$
M_{t-1}\rightarrow AnyAcquisition_t
$$

### Output

```text id="j6mkfi"
products_owned_{t-1}
records
customers
future_acquisition_rate
```

### Feature direction

```text id="7a7cyv"
products_owned_count
```

---

# EDA 3.4 — Recent acquisition intensity

### Hypothesis

Recent behavior chứa signal khác với cumulative lifetime frequency.

### Candidate windows

```text id="h2fbg1"
acquisitions_last_1m
acquisitions_last_3m
acquisitions_last_6m
```

### Test

Mỗi feature:

$$
RecentAcq_{t-1}\rightarrow Acquisition_t
$$

### Objective

Không chỉ chứng minh rolling acquisition có relationship, mà còn trả lời:

> 1m, 3m và 6m có mang information khác nhau không?

### Impact

Nếu 3m và 6m cho pattern gần giống hoàn toàn → cân nhắc chỉ giữ một để giảm redundancy.

---

# EDA 3.5 — Recent drop intensity

Tương tự:

```text id="et3d89"
drops_last_1m
drops_last_3m
drops_last_6m
```

### Hypothesis

Customer đang thu hẹp portfolio gần đây có future acquisition behavior khác.

### Test

$$
RecentDrops_{t-1}\rightarrow Acquisition_t
$$

### Feature direction

Chỉ giữ window nào có evidence và sau này có incremental validation value.

---

# EDA 3.6 — History availability / continuity

### Hypothesis

Độ dài và continuity của observed history ảnh hưởng reliability của historical features.

### Analyze

```text id="lv4j0c"
customer_history_length
record_gap_months
```

### Objective

Phân biệt:

```text id="7l8y7v"
acquisitions_last_3m = 0
because genuinely no acquisition
```

với:

```text id="40qxej"
acquisitions_last_3m = 0
because historical observations are missing
```

### Impact

Candidate:

```text id="frkpy8"
customer_history_length
record_gap_months
```

Không cần `is_continuous_gap` nếu chỉ là deterministic transform:

```text id="d17fzw"
record_gap_months == 1
```

trừ khi model/pipeline sau này cần explicit indicator.

---

# EDA 3.7 — Redundancy among derived history features

Đây là bước mình muốn **bắt buộc có** trước khi kết thúc Khối 3.

### Hypothesis

Một số engineered historical features đang mô tả cùng behavior.

Ví dụ:

```text id="8x84up"
cumulative_acquisitions
acquisitions_last_3m
acquisitions_last_6m
products_owned_count
months_since_last_acquisition
```

### Objective

Tìm feature redundancy.

### Method

Không chỉ Pearson correlation.

Dùng:

* Spearman cho numeric monotonic relationships;
* distribution/cross-analysis;
* relationship của từng feature với same future target;
* kiểm tra pair feature có gần deterministic hay không.

### Output

Correlation matrix + selected pair plots/table.

### Impact

Đề xuất:

```text id="bs4k62"
KEEP
DROP candidate
INVESTIGATE
```

Không tự động drop chỉ vì correlation cao; đây mới là EDA recommendation.

---

# KHỐI 4 — PRODUCT RELATIONSHIP EDA

Mục tiêu:

> Product nào liên quan với product nào và relationship đó có temporal structure không?

Đây là phần bổ sung quan trọng vì feature set trước chủ yếu mô tả customer history; product relationship cần được phân tích riêng. 

---

# EDA 4.1 — Ownership co-occurrence

### Hypothesis

Một số products được sở hữu cùng nhau nhiều hơn mức kỳ vọng từ popularity riêng của chúng.

### Metrics

Cho A,B:

$$
P(B=1|A=1)
$$

và:

$$
Lift(A,B)
=
\frac{P(A=1,B=1)}
{P(A=1)P(B=1)}
$$

### Output

* conditional probability matrix;
* lift matrix;
* top product pairs;
* support count.

### Interpretation

Không chỉ rank theo conditional probability vì popular products có thể dominate.

### Impact

Đây là candidate source cho product affinity feature, nhưng acquisition relationships bên dưới sát target hơn.

---

# EDA 4.2 — Same-month co-acquisition

### Hypothesis

Một số products có xu hướng được acquire cùng nhau.

### Test

$$
P(Acq_{B,t}=1|Acq_{A,t}=1)
$$

và same-month acquisition lift.

### Output

24×24:

* conditional probability;
* lift;
* support.

Top pairs table:

```text id="w7d6qs"
A
B
A_count
B_count
A+B_count
P(B|A)
P(A|B)
lift
```

### Impact

Nếu relationship mạnh và đủ support → candidate pair-affinity feature.

---

# EDA 4.3 — Sequential acquisition A → B

### Hypothesis

Sau khi acquire A, probability acquire B trong tương lai thay đổi.

### Test

$$
P(Acq_{B,t+k}=1|Acq_{A,t}=1)
$$

với:

$$
k=1,2,...,6
$$

### Logic

Với mỗi acquisition A:

```text id="20eifp"
A at t
 ↓
search B acquisition:
t+1
t+2
...
t+6
```

Không được dùng "next observed record" thay cho calendar `t+k`.

### Output

```text id="qdwz9p"
A
B
gap_months
A_event_count
B_after_A_count
conditional_rate
```

* pair×gap visualization.

---

# EDA 4.4 — Sequential lift / baseline adjustment

Đây cũng nên bắt buộc.

### Hypothesis

Một sequential relationship có thể chỉ xuất hiện vì B vốn phổ biến.

Do đó so sánh:

$$
P(B_{t+k}|A_t)
$$

với baseline:

$$
P(B_{t+k})
$$

và tính:

$$
SequentialLift_{A\rightarrow B}(k)
=
\frac{P(B_{t+k}|A_t)}
{P(B_{t+k})}
$$

### Output

Ví dụ:

```text id="iyu5b4"
A → B

gap   P(B|A)   baseline(B)   lift   support
1       ...        ...        ...     ...
2       ...        ...        ...     ...
...
```

### Impact

Nếu:

```text id="dkebxi"
lift >> 1 ở gap 1–2
lift ≈ 1 sau gap 3
```

thì mới có evidence cho time-dependent product relationship.

---

# EDA 4.5 — Product-pair timing distribution

### Hypothesis

Các pair có characteristic acquisition gap khác nhau.

### Objective

Tìm:

$$
Gap(A\rightarrow B)
$$

distribution.

### Output

Cho selected high-support/high-lift pairs:

```text id="q27vzl"
same month
1 month
2 months
...
6 months
```

Visualization: line/distribution charts.

### Feature direction

Chỉ sau EDA này mới cân nhắc:

```text id="msuh0g"
A_to_B_affinity
A_to_B_gap_affinity
months_since_related_product_acquisition
```

Không tạo hàng trăm pairwise features ngay lập tức.

---

# III. BẮT BUỘC CÓ SUMMARY SAU MỖI KHỐI

Agent phải kết thúc mỗi khối bằng một markdown summary.

Ví dụ Khối 3:

```text id="itwj4l"
## Historical Behavior EDA — Findings

Supported hypotheses
--------------------
H3: ...
H4: ...

Weak / unsupported hypotheses
-----------------------------
H5: ...

Candidate features
------------------
KEEP / INVESTIGATE:
- months_since_last_acquisition
- products_owned_count
- acquisitions_last_3m
...

Potential redundancy
--------------------
- acquisitions_last_3m vs acquisitions_last_6m: ...
- cumulative_acquisitions vs ...: ...

Risks / limitations
-------------------
- Sparse high-recency samples
- Incomplete customer history
- Only 17 checkpoints
...

Next step
---------
...
```

Quan trọng: **summary phải dựa trên output vừa chạy**, không hard-code insight.

---

# IV. SUMMARY CUỐI PRODUCT EDA

Sau cả 4 khối, coding agent phải tạo một **Feature Candidate Registry**.

Format:

| Feature                         | Source EDA | Hypothesis           | Evidence | Leakage-safe definition | Redundancy concern | Decision    |
| ------------------------------- | ---------- | -------------------- | -------- | ----------------------- | ------------------ | ----------- |
| `product_state_lag1`            | 1.3        | State persistence    | ...      | state at t-1            | —                  | Investigate |
| `products_owned_count`          | 3.3        | Portfolio propensity | ...      | count at t-1            | product states     | Investigate |
| `months_since_last_acquisition` | 3.1        | Recency              | ...      | as-of t-1               | —                  | Investigate |
| `acquisitions_last_3m`          | 3.4        | Recent behavior      | ...      | window ending t-1       | 6m                 | Investigate |
| ...                             | ...        | ...                  | ...      | ...                     | ...                | ...         |

Agent **không được gọi một feature là "final" chỉ từ EDA**.

Decision ở giai đoạn này chỉ nên là:

```text id="5jjunb"
INVESTIGATE
LOW PRIORITY
DROP CANDIDATE
```

Sau đó mới:

$$
EDA
\rightarrow FeatureEngineering
\rightarrow TemporalValidation
\rightarrow FeatureImportance/SHAP
\rightarrow FinalFeatureSet
$$

Điều này cũng khớp roadmap project: feature engineering nâng cao được thực hiện trước, sau đó feature reduction/SHAP mới dùng để tối ưu feature set. 


**Không rewrite toàn bộ notebook nếu không cần thiết.** Agent nên xem các EDA nào đã tồn tại, giữ lại analysis đúng, sửa những phần vi phạm temporal logic/metric definition, rồi bổ sung các EDA còn thiếu theo 4 khối trên. Đồng thời ưu tiên DuckDB/out-of-core thay vì load toàn bộ Santander train vào pandas vì dataset lớn.
