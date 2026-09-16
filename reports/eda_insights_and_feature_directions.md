# Các phát hiện EDA và định hướng chuẩn bị dữ liệu

## Tóm tắt điều hành

- Tập huấn luyện có **13,65 triệu dòng khách hàng-tháng**, **956,6 nghìn khách hàng** và 17 ảnh chụp dữ liệu từ **2015-01-28 đến 2016-05-28**. Mỗi `(ncodpers, fecha_dato)` là duy nhất, do đó độ hạt dùng để mô hình hóa là khách hàng tại một tháng.
- Mức sở hữu và mua mới sản phẩm mất cân bằng mạnh. Ở ảnh chụp cuối, 58,77% khách hàng có tài khoản thanh toán; nhiều sản phẩm có tỷ lệ dưới 1%. Theo nhãn tháng liền kề của EDA-v2, 96,48% khách hàng-tháng đủ điều kiện không mua sản phẩm mới nào.
- Tập kiểm tra là phần giữ lại theo thời gian tại 2016-06-28; cả 929.615 mã trong tập kiểm tra đều có lịch sử trong tập huấn luyện, nhưng không có các cột trạng thái sản phẩm. Cần xác thực và suy luận theo thời gian; không dùng trạng thái hiện tại hoặc tương lai làm đặc trưng.
- Ưu tiên: xử lý giá trị mã hóa đặc biệt/thiếu, loại các trường gần như rỗng, mã hóa biến phân loại an toàn và tạo lịch sử trễ trước khi huấn luyện mô hình dự đoán mua mới đa nhãn.

## Bằng chứng, phạm vi và thuật ngữ

Tài liệu tổng hợp các kết quả đã chạy từ `notebooks/EDA-v1.ipynb`, `notebooks/EDA-v2.ipynb` và Colab **EDA nhãn: Mua mới sản phẩm**; các số liệu là quan sát từ những kết quả đó.

EDA-v2 dùng nhãn **tháng liền kề nghiêm ngặt**: một sản phẩm được mua khi chuyển từ 0 ở `t-1` sang 1 ở `t`, và loại các bản ghi không có quan sát trước đó đúng một tháng. Điểm kiểm tra Colab hiện có thể dùng bản ghi trước gần nhất, kể cả khi có khoảng trống. Không được trộn hai định nghĩa này khi báo cáo tỷ lệ, tạo đặc trưng, xác thực hoặc chấm điểm.

## 1. Cấu trúc dữ liệu và train/test

| Hạng mục | Kết quả | Hàm ý |
|---|---:|---|
| Tập huấn luyện | 13.647.309 dòng; 48 cột | Không phù hợp để nạp toàn bộ bảng bằng Pandas. |
| Tập kiểm tra | 929.615 dòng; 24 cột | Thiếu 24 cột trạng thái sản phẩm. |
| Khách hàng trong tập huấn luyện | 956.645 | Có lịch sử lặp theo tháng. |
| Tập huấn luyện | 2015-01-28 đến 2016-05-28 | Dùng xác thực theo thời gian. |
| Tập kiểm tra | Chỉ 2016-06-28 | Dự đoán ngay sau ảnh chụp huấn luyện cuối. |
| ID train/test trùng nhau | 929.615 | Mọi ID kiểm tra đều có trạng thái lịch sử. |
| Dòng trùng / khóa trùng | 0 / 0 | Khách hàng-tháng là khóa hợp lệ. |

Không có giá trị phân loại trong tập kiểm tra chưa từng xuất hiện ở các trường chung đã kiểm tra; trung bình/trung vị số giữa train/test gần nhau, nhưng điều này không chứng minh phân phối đồng thời hay mối quan hệ với nhãn là ổn định.

## 2. Chất lượng dữ liệu

| Trường | Tỷ lệ thiếu | Cách xử lý đề xuất |
|---|---:|---|
| `conyuemp` | 99,9868% | Loại ở đường cơ sở. |
| `ult_fec_cli_1t` | 99,8183% | Loại hoặc chỉ dùng cờ hiện diện sau khi kiểm tra rò rỉ dữ liệu. |
| `renta` | 20,4756% | Trung vị theo từng phần huấn luyện + `renta_missing`. |
| `segmento`, `canal_entrada` | 1,3876%, 1,364% | Nhãn `Unknown`; `canal_entrada` cần mã hóa có kiểm soát. |
| `indrel_1mes`, `tiprel_1mes` | 1,098% | Nhãn thiếu, không xóa dòng. |
| Tỉnh | 0,686% | Giữ mã/nhãn thiếu. |

- `ncodpers` chỉ dùng để nối/nhóm/chia tập; `tipodom` là hằng số và cần loại.
- Đổi giá trị đặc biệt `antiguedad=-999999` thành thiếu; chuẩn hóa `indrel_1mes` như `1`/`1.0` về một cách biểu diễn.
- Chỉ điền giá trị thiếu ở `ind_nomina_ult1` và `ind_nom_pens_ult1` bằng 0 nếu từ điển dữ liệu xác nhận quy ước đó. Dùng một cách biểu diễn chuẩn trong `cod_prov`/`nomprov`.

## 3. Sở hữu sản phẩm

Ở ảnh chụp mới nhất: `ind_cco_fin_ult1` có 562.259 khách hàng (58,77%), `ind_recibo_ult1` 113.602 (11,88%), `ind_ctop_fin_ult1` 100.658 (10,52%), `ind_ecue_fin_ult1` 75.946 (7,94%), `ind_cno_fin_ult1` 73.110 (7,64%). `ind_ahor_fin_ult1` chỉ có 78 người sở hữu, `ind_aval_fin_ult1` có 16; mức sở hữu trên toàn bộ bảng cũng tập trung tương tự.

Giỏ sản phẩm ở ảnh chụp cuối: 26,81% không có sản phẩm, 47,71% có một sản phẩm, 12,62% có hai sản phẩm. Quy mô giỏ là đặc trưng hợp lý nhưng phải được tính từ trạng thái sở hữu trễ.

## 4. Nhãn mua mới

### Điều kiện hợp lệ và tính thưa

EDA-v2 có 12.682.421 bản ghi tháng liền kề (92,93%); 7,07% còn lại là quan sát đầu tiên hoặc có khoảng trống. Trong quần thể hợp lệ: 12.235.910 (96,4793%) không có lượt mua mới, 355.683 (2,8045%) có 1 lượt, và 90.828 (0,7162%) có từ 2 lượt trở lên.

Đây là bài toán xếp hạng đa nhãn thưa. Đánh giá bằng MAP@7, recall@K, precision/recall hoặc PR-AUC theo sản phẩm; báo cáo kết quả trung bình vĩ mô và có trọng số theo tần suất. Dùng cách chia cửa sổ mở rộng/điểm gốc cuộn theo thời gian, đồng thời khớp toàn bộ thống kê gộp/mã hóa trong từng phần huấn luyện.

### Mức tập trung và biến động

Nhãn lớn nhất: `ind_recibo_ult1` 153.146 (1,2075%), `ind_nom_pens_ult1` 84.709, `ind_nomina_ult1` 73.753, `ind_tjcr_fin_ult1` 69.118, `ind_cco_fin_ult1` 69.103. Hai nhãn hiếm nhất chỉ có 2 và 4 mẫu dương; dùng bộ xếp hạng chung có nhận biết sản phẩm, họ sản phẩm, làm trơn/tiên nghiệm phổ biến và phương án dự phòng thay vì mô hình riêng.

Tỷ lệ mua mới biến thiên từ 2,89% (2016-01) đến 5,30% (2015-06); 2016-02 là 3,95%. Đây là thay đổi mang tính mô tả, không chứng minh tính mùa vụ/chiến dịch, nhưng ủng hộ việc dùng đặc trưng lịch và theo dõi theo tháng.

### EDA vòng đời bổ sung từ Colab

| Quan sát tại điểm kiểm tra Colab | Hướng xử lý |
|---|---|
| 13.199.406/13.647.309 khách hàng-tháng (96,72%) không mua mới; trong 447.903 bản ghi dương, 356.918 chỉ mua một sản phẩm. | Xếp hạng ứng viên với K nhỏ; cân nhắc mô hình khuynh hướng mua riêng. |
| 956.645 khách hàng tạo 563.310 sự kiện; 21,15% từng mua, trung vị 0, tối đa 22. | Đánh giá riêng tất cả khách hàng và người mua đang hoạt động/đủ điều kiện. |
| `ind_recibo_ult1` có 153.205 lượt mua tại điểm kiểm tra; sản phẩm hiếm nhất có 2--4. | Tiên nghiệm sản phẩm có làm trơn, mô hình theo họ/chung, phương án dự phòng. |
| 2016-02 có 49.072 sự kiện từ 36.113 người mua. | Hiệu ứng tháng, kiểm tra tỷ lệ dự đoán theo từng phần. |
| `ind_cco_fin_ult1` dẫn đầu tồn lượng sở hữu (561.615), còn `ind_recibo_ult1` dẫn đầu dòng mua mới (10.163). | Tách tồn lượng trễ, tỷ lệ mua mới cuộn và thay đổi tồn lượng ròng. |
| `ind_deco_fin_ult1` có tỷ lệ đóng 26,7%; `ind_reca_fin_ult1` có tỷ lệ duy trì 99,998%. | Cờ mở/đóng, số tháng từ sự kiện, số lần cuộn 3/6 tháng; làm trơn sản phẩm hiếm. |
| Trung vị quy mô giỏ luôn là 1; P90 giảm từ 4 xuống 3 khi quần thể tăng từ khoảng 632 nghìn lên 830 nghìn. | Chuẩn hóa/hiệu ứng cố định theo tháng, nhóm quan sát đầu tiên. |
| Đỉnh theo sản phẩm: thẻ 2015-03; trích nợ tự động/khoản vay/chứng khoán 2015-10; tài khoản thanh toán/lương-hưu mạnh 2015-12--2016-02. | Sản phẩm × tháng, tiên nghiệm mùa vụ theo họ, tỷ lệ mua mới trễ. |

## 5. Tiền xử lý, rò rỉ dữ liệu và hợp đồng đặc trưng

1. Chia theo thời gian trước; chỉ khớp bộ điền thiếu, bộ mã hóa, ngưỡng cắt và thống kê nhãn trên giai đoạn huấn luyện.
2. Loại `ncodpers` khỏi đầu vào mô hình; loại `tipodom`, `conyuemp` và nhiều khả năng là `ult_fec_cli_1t` ở đường cơ sở.
3. Điền biến số bằng trung vị của phần huấn luyện, thêm chỉ báo thiếu; mã hóa một-nóng cho biến có ít giá trị. Với `canal_entrada`/`pais_residencia`, so sánh mã hóa tần suất, mã hóa nhãn có điều chuẩn an toàn theo phần và xử lý biến phân loại gốc của mô hình.
4. Tại tháng `t`, chỉ dùng hồ sơ sẵn sàng trong vận hành tại `t` và lịch sử sản phẩm từ `t-1` trở về trước. Không dùng nhãn, trạng thái sản phẩm tại `t` hay ảnh chụp tương lai.
5. Với tập kiểm tra 2016-06, nối trạng thái sản phẩm lịch sử tháng 2016-05 bằng `ncodpers`; không coi cột sản phẩm chỉ tồn tại trong tập huấn luyện là biến dự báo thông thường của tập kiểm tra.

| Nhóm đặc trưng | Ví dụ |
|---|---|
| Sở hữu trễ/giỏ sản phẩm | Cờ sở hữu từng sản phẩm, số sản phẩm, số lượng theo họ, giỏ rỗng tại `t-1`. |
| Chuyển đổi/vòng đời | Lịch sử mua mới/hủy, lần mở/đóng gần nhất, số tháng từ sự kiện, số lần cuộn 3/6 tháng. |
| Tồn lượng và dòng | Tỷ lệ mua mới sản phẩm cuộn, thay đổi tồn lượng ròng, tỷ lệ duy trì/đóng có làm trơn. |
| Gần đây/quỹ đạo | Số tháng từ `fecha_alta`, thay đổi hồ sơ/trạng thái gần nhất, số bản ghi lịch sử, thay đổi giỏ 1/3/6 tháng. |
| Lịch/nhóm thời điểm | Tháng ảnh chụp, tháng đã trôi qua, nhóm quan sát đầu tiên, danh mục chuẩn hóa theo tháng. |
| Tương tác/ứng viên | Sản phẩm × tháng, tương tác hồ sơ có điều chuẩn; không đề xuất sản phẩm đã sở hữu ở `t-1`. |

Ma trận chuyển đổi chỉ là hiện vật khám phá: nếu dùng việc sở hữu sản phẩm A để dự đoán mua mới B, mọi tỷ lệ phải được tính lại trong phần huấn luyện theo thời gian.

## 6. Quyết định bắt buộc về nhãn

Chọn một chính sách trước khi xây dựng đường cơ sở:

1. **Tháng liền kề**: loại bản ghi không có bản ghi trước đó đúng một tháng; đây là chính sách của EDA-v2.
2. **Bản ghi trước gần nhất**: so sánh với bản ghi trước gần nhất, kể cả khi có khoảng trống; đây là chính sách mà điểm kiểm tra hiện có thể dùng.

Nếu dùng bản ghi trước gần nhất, hãy chạy lại toàn bộ EDA nhãn theo định nghĩa đó. Nếu dùng tháng liền kề, hàm tái sử dụng cần sinh `is_consecutive_month` và lọc cờ này nhất quán trong huấn luyện/đánh giá.

## Nguồn tham chiếu

| Nguồn | Nội dung |
|---|---|
| `notebooks/EDA-v1.ipynb` | Cấu trúc/độ hạt, lược đồ, kiểm tra thiếu/trùng, sở hữu, giỏ sản phẩm, kiểm tra rò rỉ train/test. |
| `notebooks/EDA-v2.ipynb` | Kiểm tra khoảng trống, điều kiện hợp lệ của nhãn, tính thưa, dịch chuyển theo tháng, ma trận chuyển đổi. |
| Colab `Target EDA: Product Acquisitions` | Phân bố mua mới tại điểm kiểm tra, tồn lượng-dòng, chuyển đổi/duy trì, danh mục, mùa vụ và phần tóm tắt trực quan liên kết. |
