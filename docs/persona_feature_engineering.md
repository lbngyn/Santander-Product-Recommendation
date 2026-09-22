# Bảng Feature Engineering cho Persona Data

| Raw feature → New feature | Loại | Kiểu / Valid values | Xử lý missing | Redundancy / Merge | Cách thực hiện | Ý nghĩa |
|---|---|---|---|---|---|---|
| `fecha_dato` → `time_idx` | Temporal | Integer `[0,+∞)` | Không missing | Raw date không feed trực tiếp vào model | Tính số tháng kể từ snapshot đầu tiên: `2015-01 → 0`, `2015-02 → 1`, ... | Xác định vị trí sample trên timeline và capture temporal trend. |
| `fecha_dato` → `snapshot_month` | Temporal | Integer categorical `[1,12]` | Không missing | Dẫn xuất cùng `time_idx`, nhưng không redundant hoàn toàn | Lấy `MONTH(fecha_dato)`. Có thể thử thêm sin/cos sau và quyết định bằng temporal validation. | Capture calendar/seasonal effect. |
| `fecha_alta` + `fecha_dato` → `account_age_months` | Tenure | Integer / `MISSING`, `[0,+∞)` | Past-only fill `fecha_alta`; nếu customer chưa từng có giá trị trước `t` thì giữ `MISSING` | Thay raw `fecha_alta`; ưu tiên thay `antiguedad` do hai feature rất redundant | Tính số tháng giữa `fecha_alta` và `fecha_dato` tại từng sample. | Tuổi tài khoản của customer tại đúng thời điểm sample. |
| `antiguedad` → DROP / fallback cho `account_age_months` | Tenure | — | Đổi sentinel âm như `-999999` thành `MISSING`; có thể past-fill trước khi dùng làm fallback | Redundant mạnh với `fecha_alta` (Pearson khoảng `-0.967`) | Ưu tiên `account_age_months`; chỉ dùng `antiguedad` để bổ sung khi `fecha_alta` thiếu nếu consistency được xác minh. | Tránh đưa hai representation gần như cùng một tenure signal vào model. |
| `age` → `age` | Demographic | Numeric / `MISSING` | Past-only fill; nếu chưa từng observed thì giữ `MISSING` | Giữ | Giữ numeric gốc; không tự động clip chỉ vì ngoài 18–90. | Tuổi customer tại sample. |
| `renta` → `income_log` | Financial | Float / `MISSING` | Past-only fill; nếu chưa từng observed thì giữ `MISSING` | Có thể thay raw `renta` bằng representation log | `log1p(renta)`; nếu clipping thì threshold phải fit trên train. | Income signal với distribution ít skew hơn. |
| `ind_nuevo` → `is_new_customer` | Binary | `{0,1,MISSING}` | Không past-fill vô điều kiện vì đây là state theo thời điểm; missing giữ `MISSING` | Có liên hệ với tenure nhưng không hoàn toàn redundant | Chuẩn hóa về `0/1/MISSING`. | Customer có được đánh dấu là customer mới tại `t` hay không. |
| `indrel` + `indrel_1mes` + `tiprel_1mes` → `customer_relationship_status` | Relationship | Categorical + `MISSING` | Không dùng tương lai để fill; normalize các state và giữ `MISSING` khi không xác định | Merge ba raw feature vì EDA cho thấy chúng bao hàm/chồng lấn mạnh về trạng thái quan hệ | Chuẩn hóa `1`/`1.0`; mapping các state tương ứng và tạo một canonical relationship status duy nhất. | Một representation thống nhất về relationship giữa customer và ngân hàng. |
| `ind_actividad_cliente` → `is_active_customer` | Binary | `{0,1,MISSING}` | Không past-fill vô điều kiện vì trạng thái có thể thay đổi; missing giữ `MISSING` | Có liên hệ với relationship status nhưng không hoàn toàn redundant | Chuẩn hóa thành `0/1/MISSING`. | Customer active hay inactive tại thời điểm `t`. |
| `ind_empleado` → `employee_status` | Categorical | `{A,B,F,N,P,MISSING}` | Past-only fill nếu status đã được quan sát trước đó; nếu không có history thì `MISSING` | Không merge với `conyuemp` | Native categorical hoặc one-hot tùy model. | Trạng thái employment của customer đối với ngân hàng. |
| `conyuemp` → DROP baseline | Binary | — | Không biến null thành `0` | Missing gần như toàn bộ nên không đủ coverage; không merge vào `employee_status` | Không đưa vào baseline; chỉ thử riêng nếu temporal validation chứng minh hữu ích. | Tránh feature cực sparse và không ổn định. |
| `sexo` → `is_male` | Binary | `{0,1,MISSING}` | Past-only fill; nếu chưa từng observed thì `MISSING` | Giữ | `H → 1`, `V → 0`, null → `MISSING`. | Chuẩn hóa gender thành binary. |
| `indfall` → `is_deceased` | Binary / eligibility | `{0,1,MISSING}` | Past-only fill nếu có lịch sử hợp lệ; nếu không thì `MISSING`; không biến missing thành `0` | Giữ; structural missing có ý nghĩa riêng | `S → 1`, `N → 0`, null → `MISSING`. Customer `1` có thể được route khỏi recommendation trước model. | Phân biệt deceased, not deceased và unknown. |
| `pais_residencia` → `country` | Geography | Categorical + `MISSING` | Past-only fill; nếu chưa từng observed thì `MISSING` | Canonical geography feature cấp country | Giữ country code; dùng native categorical hoặc frequency/count encoding. | Quốc gia cư trú của customer. |
| `cod_prov` + `nomprov` → `province` | Geography | Categorical + `MISSING` | Past-only fill `cod_prov`; nếu chưa có thì `MISSING`; không suy province chỉ từ country | Drop `nomprov`, giữ `cod_prov` vì hai cột gần redundant | Rename `cod_prov → province`; categorical/frequency encode. | Geographic information cấp province mà không duplicate code và name. |
| `indresi` + `pais_residencia` → `is_domestic` | Geography / Binary | `{0,1,MISSING}` | Past-only fill khi có thông tin lịch sử; nếu không xác định thì `MISSING` | `indresi` redundant/interactive mạnh với `country`; ưu tiên canonical representation | Domestic → `1`, foreign → `0`, unknown → `MISSING`; có thể derive/consistency-check từ `country`. | Binary abstraction của country. |
| `indext` → DROP / consistency check | Geography | — | — | Redundant với `country` / `is_domestic` | Không đưa baseline model; dùng để kiểm tra consistency trong preprocessing. | Tránh nhiều feature cùng biểu diễn domestic/foreign status. |
| `canal_entrada` → `entry_channel` | Acquisition | Categorical + `MISSING` | Ưu tiên past-only fill nếu đã từng observed; nếu chưa có history thì `MISSING` | Giữ | Normalize category; native categorical hoặc frequency/count encoding. | Channel customer gia nhập/được tiếp cận bởi ngân hàng. |
| `segmento` → `customer_segment` | Customer segment | Categorical + `MISSING` | Không past-fill vô điều kiện vì segment thực sự có thể thay đổi; missing giữ `MISSING` | Giữ | Normalize categorical; không ordinal encode nếu chưa có domain evidence. | Segment ngân hàng gán cho customer tại `t`. |
| `ult_fec_cli_1t` + relationship features → absorbed vào `customer_relationship_status` | Relationship | — | Không impute raw date | Raw date cực sparse và gắn chặt với relationship state | Không feed raw date; có thể dùng availability/state của cột khi xây `customer_relationship_status`. | Bổ sung relationship information mà không tạo predictor datetime cực sparse. |
| `tipodom` → DROP | Address | — | Không blanket-fill missing thành giá trị observed | Observed value gần như constant | Không feed raw value vào model. | Loại feature gần constant. |
| Common missing cohort → `profile_missing_structural` | Data quality | Boolean `{0,1}` | Không cần impute | Gộp pattern missing đồng thời của nhóm profile features thành một structural signal | `1` nếu record thuộc common historical profile-missing pattern, ngược lại `0`. | Capture historical/profile coverage thay vì coi các missing đồng thời là các quá trình độc lập. |

# Mapping tên feature từ cũ sang mới

| Feature cũ | Feature mới | Trạng thái |
|---|---|---|
| `fecha_dato` | `time_idx` | Dẫn xuất |
| `fecha_dato` | `snapshot_month` | Dẫn xuất |
| `ncodpers` | `customer_id` | Rename; key only, không feed model |
| `age` | `age` | Giữ |
| `fecha_alta` + `fecha_dato` | `account_age_months` | Dẫn xuất |
| `antiguedad` | `account_age_months` | Redundant / fallback |
| `renta` | `income_log` | Transform |
| `ind_nuevo` | `is_new_customer` | Rename + binary normalize |
| `indrel` | `customer_relationship_status` | Merge |
| `indrel_1mes` | `customer_relationship_status` | Merge |
| `tiprel_1mes` | `customer_relationship_status` | Merge |
| `ult_fec_cli_1t` | `customer_relationship_status` | Absorb / raw column drop |
| `ind_actividad_cliente` | `is_active_customer` | Rename + binary normalize |
| `ind_empleado` | `employee_status` | Rename |
| `conyuemp` | — | Drop baseline |
| `sexo` | `is_male` | Rename + binary normalize |
| `indfall` | `is_deceased` | Rename + binary normalize |
| `pais_residencia` | `country` | Rename |
| `pais_residencia` + `indresi` | `is_domestic` | Canonicalize / merge |
| `indext` | — | Drop / consistency check |
| `cod_prov` | `province` | Rename |
| `nomprov` | — | Drop; redundant với `cod_prov` |
| `canal_entrada` | `entry_channel` | Rename |
| `segmento` | `customer_segment` | Rename |
| `tipodom` | — | Drop |
| Common profile missing pattern | `profile_missing_structural` | Dẫn xuất |

# Rule cho Past Fill

| Rule | Cách thực hiện |
|---|---|
| **Past-only** | Với sample `(customer, t)`, chỉ được sử dụng giá trị của chính customer tại thời điểm `< t` để fill giá trị missing tại `t`. |
| **Ưu tiên giá trị gần nhất** | Nếu feature tại `t` missing, lấy giá trị non-missing gần `t` nhất trong lịch sử trước đó của cùng customer. |
| **Không có lịch sử** | Nếu customer chưa từng có giá trị hợp lệ trước `t`, giữ nguyên `MISSING`. |
| **Không backfill** | Tuyệt đối không dùng giá trị ở `t+1`, `t+2`, ... để fill cho sample tại `t`. |
| **Stable/profile features** | Ưu tiên past fill cho `sexo/is_male`, `fecha_alta`, `age`, `renta`, `country`, `province`, `entry_channel`, `employee_status`, `is_deceased`. |
| **Dynamic state features** | Không past-fill vô điều kiện cho `is_new_customer`, `is_active_customer`, `customer_segment`, `customer_relationship_status`; nếu thiếu tại `t`, mặc định giữ `MISSING` trừ khi có rule domain được xác minh riêng. |
| **Binary / Boolean** | Chuẩn hóa positive → `1`, negative → `0`, unknown/null → `MISSING`. Không chuyển `MISSING → 0`. |
| **Thứ tự dữ liệu** | Trước khi past fill, sort theo `ncodpers`, `fecha_dato`. |
| **Leakage safety** | Nguồn dùng để fill một row tại `t` bắt buộc có timestamp `< t`; không sử dụng thông tin tương lai. |

Ví dụ:

| Customer | Tháng | `sexo` raw | `sexo` sau past fill | Nguồn |
|---|---:|---|---|---|
| A | Jan | `H` | `H` | Current |
| A | Feb | `MISSING` | `H` | Jan |
| A | Mar | `MISSING` | `H` | Jan |
| A | Apr | `H` | `H` | Current |

Khi tạo sample February hoặc March, giá trị April không được phép tham gia vào quá trình fill.
