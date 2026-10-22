# Báo cáo giải thích CatBoostRegressor và SVR

**Đối tượng đọc:** thành viên nhóm nghiên cứu  
**Phạm vi:** hai mô hình hồi quy điểm đang có trong `CATBoost-SVR`  
**Ngày lập:** 22/10/2026

## 1. Tóm tắt cho nhóm

CatBoostRegressor và SVR đều là thuật toán **machine learning có giám sát** dùng được cho bài toán hồi quy: học từ các hồ sơ đã có đầu vào và điểm đích, sau đó dự đoán điểm cho hồ sơ mới.

Trong dự án này, hai mô hình đang dự đoán điểm Toán, Ngữ văn và Ngoại ngữ trên dữ liệu mô phỏng. Kết quả hiện tại cho thấy artifact CatBoost có sai số thấp hơn SVR trên ba môn, nhưng CatBoost được huấn luyện bằng 30.000 dòng còn SVR chỉ dùng 1.500 dòng. Vì vậy bảng kết quả hệ thống chưa phải phép so sánh công bằng để kết luận thuật toán CatBoost luôn tốt hơn.

Ở phép thử Toán trong notebook, hai mô hình có cùng 1.500 dòng train và cùng tập test; cấu hình CatBoost cho kết quả tốt hơn cấu hình SVR trong lần chạy đó. Đây là bằng chứng ban đầu để tiếp tục nghiên cứu, chưa phải kết luận tổng quát vì mới có một môn, một cấu hình mỗi mô hình, một seed và dữ liệu mô phỏng.

## 2. Bài toán machine learning trong dự án

Mỗi hồ sơ được biểu diễn thành các đặc trưng `x` và nhãn điểm `y`:

- `X`: khoảng 55 cột điểm học tập, giờ tự học, chuyên cần và điểm thi thử tương ứng.
- `y`: một trong các cột `diem_thi_toan_mo_phong`, `diem_thi_ngu_van_mo_phong`, `diem_thi_ngoai_ngu_mo_phong`.
- Đầu ra là giá trị số trên thang 0–10, nên đây là **hồi quy**, không phải phân loại.

Ba tập dữ liệu có vai trò khác nhau:

| Tập | Số dòng | Số nhóm nguồn | Dùng để |
|---|---:|---:|---|
| Train | 30.000 | 6.000 | Học quy luật từ dữ liệu |
| Validation | 10.000 | 2.000 | Chọn vòng dừng/tham số |
| Test | 10.000 | 2.000 | Đánh giá cuối sau khi chốt cách làm |

Mỗi hồ sơ gốc có các biến thể liên quan trong bộ dữ liệu 50.000 dòng. Các biến thể của cùng `nhom_ho_so_goc` cần nằm trong cùng một tập để tránh thông tin gần như trùng lặp lọt từ train sang test. Notebook kiểm tra và ghi nhận không có nhóm giao nhau giữa các tập.

## 3. CatBoostRegressor hoạt động như thế nào?

### 3.1. Trực giác

CatBoost là gradient boosting dùng cây quyết định. Cây thứ nhất tạo dự đoán ban đầu; các cây sau lần lượt học cách cải thiện dự đoán hiện tại. Tổng dự đoán gồm dự đoán ban đầu cộng đóng góp của nhiều cây.

Với mẫu thứ `i` sau `t` cây, có thể viết khái quát:

**ŷᵢ⁽ᵗ⁾ = ŷᵢ⁽ᵗ⁻¹⁾ + η · hₜ(xᵢ)**

Trong đó:

- `x_i`: các đặc trưng của hồ sơ thứ `i`.
- `ŷ_i^(t)`: dự đoán sau cây thứ `t`.
- `h_t`: cây mới được học ở vòng `t`.
- `η` (`learning_rate`): mức đóng góp của cây mới.

Ở mức toán học, cây mới được xây để xấp xỉ hướng làm giảm hàm mất mát, thường liên hệ với **gradient âm**:

**gᵢ⁽ᵗ⁾ = −∂L(yᵢ, ŷᵢ) / ∂ŷᵢ**

`g_i` là hướng và mức thay đổi dự đoán có thể giảm lỗi tại mẫu đó. Cây chia các mẫu thành vùng; mỗi vùng nhận một giá trị cập nhật. Vì thế boosting không phải chỉ lấy trung bình nhiều cây độc lập như trực giác về Random Forest.

### 3.2. Hàm mất mát RMSE

Mã dự án dùng `loss_function="RMSE"`. RMSE được định nghĩa:

**RMSE = √[(1/n) · Σᵢ₌₁ⁿ (yᵢ − ŷᵢ)²]**

Sai số lớn bị bình phương nên ảnh hưởng mạnh hơn sai số nhỏ. CatBoost điều chỉnh các cây để tối ưu hàm mục tiêu đã chọn; trên validation, `eval_metric="RMSE"` theo dõi chất lượng và hỗ trợ early stopping.

### 3.3. Những điểm đặc trưng

- **Boosting tuần tự:** cây mới được xây dựa trên lỗi của mô hình hiện thời.
- **Ordered Boosting:** dùng hoán vị và cách tính theo thứ tự để giảm một dạng prediction shift trong boosting.
- **Biến phân loại:** CatBoost hỗ trợ các cột dạng nhóm và tạo biểu diễn số/thống kê phù hợp; phải khai báo đúng cột phân loại.
- **Cây đối xứng mặc định:** cùng điều kiện chia được áp dụng ở cùng một tầng của cây theo cấu hình mặc định.

Trong mô hình điểm hiện tại, 55 đặc trưng là số. Do đó lợi thế xử lý biến phân loại của CatBoost chưa được khai thác trong thí nghiệm này. Ordered Boosting cũng không thay thế việc chia train/test đúng hoặc kiểm soát phụ thuộc giữa nhóm hồ sơ.

## 4. SVR hoạt động như thế nào?

SVR là Support Vector Regression, biến thể hồi quy của Support Vector Machine. Ý tưởng là tìm một hàm dự đoán phẳng nhất có thể, đồng thời cho phép sai số nằm trong một biên `epsilon` mà chưa bị phạt.

### 4.1. Công thức epsilon-SVR

Với hàm tuyến tính trong không gian đặc trưng `φ(x)`, dạng tối ưu hóa chuẩn có thể viết:

**Tối thiểu hóa:** ½‖w‖² + C · Σᵢ₌₁ⁿ(ξᵢ + ξᵢ*)

Với mọi mẫu `i`, các ràng buộc là:

- yᵢ − (wᵀφ(xᵢ) + b) ≤ ε + ξᵢ
- (wᵀφ(xᵢ) + b) − yᵢ ≤ ε + ξᵢ*
- ξᵢ ≥ 0 và ξᵢ* ≥ 0

Trong đó:

- `w, b`: tham số của hàm dự đoán.
- `ε` (`epsilon`): độ rộng vùng sai số không bị phạt.
- `ξ, ξ*`: phần sai số vượt khỏi vùng `epsilon`.
- `C`: mức phạt cho sai số vượt biên; `C` lớn làm mô hình cố gắng khớp train chặt hơn, nhưng có thể nhạy với nhiễu.

Có thể diễn giải phần sai số bằng hàm epsilon-insensitive:

**Lε(y, f(x)) = max(0, |y − f(x)| − ε)**

Sai số nhỏ hơn `epsilon` có loss bằng 0 theo hàm này; vượt biên thì bắt đầu bị phạt.

### 4.2. Kernel RBF

Mã dự án dùng `kernel="rbf"`, cho phép mô hình hóa quan hệ phi tuyến. Kernel RBF:

**K(x, x′) = exp(−γ‖x − x′‖²)**

`γ` (`gamma`) quyết định mức ảnh hưởng theo khoảng cách giữa hai hồ sơ:

- `gamma` lớn: ảnh hưởng tập trung gần mỗi điểm, mô hình có thể uốn cong phức tạp hơn.
- `gamma` nhỏ: ảnh hưởng lan rộng hơn, mô hình thường mượt hơn.

SVR dựa vào một tập các điểm train được gọi là **support vectors** để xác định hàm dự đoán. Trong artifact dự án có 1.373–1.390 support vectors từ 1.500 dòng train, tức phần lớn các điểm train đang tham gia xác định ranh giới hồi quy.

### 4.3. Vì sao chuẩn hóa?

Khoảng cách trong kernel RBF phụ thuộc thang đo các đặc trưng. Nếu một cột có miền giá trị lớn hơn nhiều cột khác, nó có thể chi phối khoảng cách. Pipeline dự án đặt `StandardScaler` trước SVR để chuẩn hóa đặc trưng. Scaler phải được fit trên train bên trong pipeline/fold, sau đó mới áp dụng lên validation/test để tránh rò rỉ thống kê.

## 5. Các tham số chính có thể hiệu chỉnh

### 5.1. CatBoostRegressor

| Tham số | Giá trị hiện tại | Tác động và hướng thử |
|---|---:|---|
| `iterations` | Tối đa 1.000 | Số cây tối đa. Dùng validation và early stopping để chọn số cây; không nhất thiết phải dùng hết 1.000. |
| `depth` | 8 | Độ sâu cây. Thử 4, 6, 8, 10; depth cao tăng khả năng học tương tác nhưng cũng tăng độ phức tạp/overfit. |
| `learning_rate` | 0,05 | Bước cập nhật của mỗi cây. Thử 0,03; 0,05; 0,1 đi cùng ngân sách `iterations`; learning rate thấp thường cần nhiều cây hơn. |
| `loss_function` | `RMSE` | Mục tiêu tối ưu trong train. Có thể nghiên cứu `MAE` nếu muốn giảm sai số tuyệt đối, nhưng cần chốt theo câu hỏi nghiên cứu. |
| `eval_metric` | `RMSE` | Metric theo dõi validation/early stopping. Có thể cân nhắc MAE nếu đó là metric chính đã chọn trước. |
| `early_stopping_rounds` | 100 | Số vòng chờ validation trước khi dừng. Thử một số giá trị định trước, ví dụ 50, 100, 150. |
| `l2_leaf_reg` | Mặc định thư viện | Regularization cho giá trị lá; nên ghi tường minh và có thể thử vài mức để kiểm soát overfit. |
| `random_strength` | Mặc định thư viện | Thêm ngẫu nhiên vào điểm đánh giá split; có thể thử như regularization phụ. |
| `bootstrap_type`, `bagging_temperature` hoặc `subsample` | Mặc định thư viện | Quy định cách lấy mẫu/trọng số quan sát. Tuning các tham số tương thích theo loại bootstrap, không chọn tùy tiện kết hợp. |
| `random_seed` | 42 | Cố định để tái lập; dùng thêm seed sau khi chọn cấu hình để đánh giá độ ổn định. |

### 5.2. SVR

| Tham số | Giá trị artifact | Tác động và hướng thử |
|---|---:|---|
| `kernel` | `rbf` | Dạng quan hệ. Có thể so sánh `linear` và `rbf`; thử `poly` chỉ khi có giả thuyết phù hợp. |
| `C` | 10,0 | Cân bằng độ phẳng với lỗi train. Tìm theo thang log, ví dụ 0,1; 1; 10; 100. |
| `epsilon` | 0,1 | Biên không phạt sai số. Thử 0,05; 0,1; 0,2; 0,5 tùy độ phân giải và mục tiêu điểm. |
| `gamma` | `scale` | Phạm vi ảnh hưởng RBF. So sánh `scale`, `auto` và một dải float theo log. Có tương tác mạnh với `C` và scaling. |
| `tol` | 0,01 | Ngưỡng dừng solver; giảm có thể đòi hỏi nghiệm chặt hơn và tốn thời gian hơn. |
| `max_iter` | 20.000 trong artifact | Giới hạn vòng lặp. Phải kiểm tra `fit_status_ == 0`; lần notebook thăm dò dùng 5.000 và ghi nhận chưa hội tụ. |
| `shrinking` | `True` (mặc định) | Heuristic của solver, có thể ảnh hưởng tốc độ hội tụ; ưu tiên tuning sau C/epsilon/gamma. |
| `StandardScaler` | bật mặc định | Cần giữ scaling cho SVR RBF; mọi bước fit scaler phải chỉ dùng train. |

Tuning nên tập trung trước vào `C`, `epsilon`, `gamma`, sau đó mới so sánh kernel và tham số solver. Tham số cần được chọn bằng validation hoặc cross-validation theo nhóm nguồn, không dựa trên test.

## 6. Đánh giá chất lượng mô hình

### 6.1. Chỉ số hồi quy

- **MAE:** **MAE = (1/n) · Σᵢ₌₁ⁿ |yᵢ − ŷᵢ|**. Diễn giải được trực tiếp theo điểm; thấp hơn tốt hơn.
- **RMSE:** căn trung bình bình phương sai số; cũng cùng đơn vị điểm và phạt lỗi lớn mạnh hơn MAE; thấp hơn tốt hơn.
- **R²:** so chất lượng với dự đoán bằng trung bình nhãn; cao hơn tốt hơn, có thể âm nếu kém hơn baseline trung bình.

Hồi quy không có một “accuracy %” duy nhất như phân loại. Khi cần phần trăm dễ hiểu, có thể báo **tỷ lệ dự đoán nằm trong ngưỡng sai số đã định**, chẳng hạn tỷ lệ có sai số tuyệt đối không quá 0,5 hoặc 1 điểm. Phải gọi đúng tên chỉ số và giải thích ngưỡng; không đổi MAE thành “độ chính xác”.

### 6.2. Kết quả artifact hệ thống

Metric dưới lấy từ `models/metadata.json` và `models/svr_metadata.json`. Mỗi môn được đánh giá trên 10.000 dòng test. CatBoost train 30.000 dòng; SVR train 1.500 dòng thuộc 300 nhóm.

| Môn | Mô hình | Train | MAE | RMSE | R² | Trong ±0,5 điểm | Trong ±1,0 điểm |
|---|---|---:|---:|---:|---:|---:|---:|
| Toán | CatBoost | 30.000 | 0,755 | 0,951 | 0,866 | 41,00% | 70,60% |
| Toán | SVR | 1.500 | 0,924 | 1,158 | 0,802 | 33,65% | 61,03% |
| Ngữ văn | CatBoost | 30.000 | 0,745 | 0,936 | 0,877 | 41,35% | 71,42% |
| Ngữ văn | SVR | 1.500 | 0,930 | 1,162 | 0,811 | 32,78% | 60,92% |
| Ngoại ngữ | CatBoost | 30.000 | 0,737 | 0,931 | 0,875 | 41,86% | 72,35% |
| Ngoại ngữ | SVR | 1.500 | 0,919 | 1,152 | 0,809 | 33,33% | 61,29% |

Các phần trăm được tính lại trên test từ artifact là tỷ lệ `|y - ŷ| ≤ ngưỡng`. Chúng là phần trăm nằm trong ngưỡng sai số, **không phải accuracy phân loại**. Ngưỡng ±0,5/±1,0 là lựa chọn báo cáo hiện tại, chưa phải chuẩn ngành hoặc ngưỡng đã được xác nhận có ý nghĩa giáo dục.

### 6.3. So sánh công bằng hiện có: Toán với cùng 1.500 dòng train

Trong `analys/svr.ipynb`, CatBoost được chạy thêm trên đúng mẫu train 1.500 dòng/300 nhóm dùng cho SVR, và cả hai được đánh giá trên cùng test:

| Mô hình/mốc | MAE test | RMSE test | R² test |
|---|---:|---:|---:|
| CatBoost, depth 8 | 0,810 | 1,009 | 0,849 |
| SVR RBF | 0,924 | 1,158 | 0,802 |
| Baseline `toan_lop9` | 1,065 | 1,352 | 0,730 |

Trong lần chạy ghép cặp này CatBoost giảm MAE khoảng 12,3%, RMSE khoảng 12,9% so với SVR và R² cao hơn 0,047. Hai cấu hình đều tốt hơn baseline trên Toán. Tuy nhiên, SVR trong notebook dùng `max_iter=5.000` và báo `fit_status=1` (chạm giới hạn, chưa hội tụ); artifact SVR riêng dùng `max_iter=20.000` và đã hội tụ. Vì vậy cần chạy lại phép so sánh ghép cặp với cấu hình SVR hội tụ trước khi xem đây là kết quả cuối.

### 6.4. Baseline và cách kết luận

Baseline điểm lớp 9 trên test:

| Môn | MAE | RMSE | R² |
|---|---:|---:|---:|
| Toán (`toan_lop9`) | 1,065 | 1,352 | 0,730 |
| Ngữ văn (`ngu_van_lop9`) | 0,924 | 1,165 | 0,810 |
| Ngoại ngữ (`ngoai_ngu_lop9`) | 1,036 | 1,315 | 0,751 |

SVR Ngữ văn có MAE 0,930, hơi cao hơn baseline 0,924; do đó SVR không cải thiện mọi metric/môn một cách đồng nhất. Không nên nói đơn giản rằng “cả hai mô hình đều chính xác X%” hoặc “CatBoost luôn tốt hơn”. Câu kết luận vững hơn là: **trên các artifact và phép chia mô phỏng hiện có, CatBoost có metric tốt hơn nhưng so sánh artifact bị lệch lượng train; ở phép thử Toán cùng lượng train, cấu hình CatBoost thắng lần chạy đó, còn cần xác nhận lại với SVR hội tụ.**

## 7. Hai mô hình này có phù hợp cho machine learning không?

Nếu câu hỏi là “đây có phải thuật toán machine learning phù hợp để học và dự báo điểm dạng bảng không?” thì câu trả lời là **có, cả hai đều phù hợp để đưa vào thực nghiệm**:

### CatBoostRegressor

- Phù hợp với dữ liệu dạng bảng và quan hệ phi tuyến; có hỗ trợ biến phân loại nếu khai báo đúng.
- Thường ít đòi hỏi chuẩn hóa thang đo so với SVR.
- Trong dự án hiện tại, artifact dùng được 30.000 dòng train và cho MAE/RMSE/R² tốt hơn baseline trên cả ba nhãn mô phỏng.
- Hạn chế: cấu hình hiện tại chưa được tuning có hệ thống; dữ liệu toàn số nên chưa kiểm tra lợi thế biến phân loại; kết quả có thể học lại công thức tạo nhãn mô phỏng.

### SVR RBF

- Có thể học quan hệ phi tuyến; phù hợp làm mô hình đối chứng cho hồi quy với số mẫu vừa phải.
- Cần chuẩn hóa đặc trưng; kết quả nhạy với `C`, `epsilon`, `gamma`.
- Trong dự án, 1.500 dòng train giúp thời gian thực nghiệm nhẹ hơn; nhiều support vectors khiến kích thước/chi phí dự đoán có thể tăng.
- Hạn chế về quy mô: tài liệu scikit-learn lưu ý SVR dựa trên libsvm có chi phí fit tăng nhanh hơn bậc hai theo số mẫu, nên khó mở rộng đến hàng chục nghìn mẫu; CatBoost thực tế phù hợp hơn khi muốn dùng đủ 30.000 dòng trong cấu hình hiện tại.

Nếu câu hỏi là “đã đủ để dùng tư vấn học sinh thật hay chọn môn tự chọn phù hợp chưa?” thì câu trả lời là **chưa**. Nhãn điểm đang mô phỏng và mục tiêu của hai mô hình là dự đoán ba điểm thi, không phải đo độ phù hợp với bốn môn tự chọn. Muốn đi tới tư vấn thực tế cần dữ liệu thật được phép sử dụng, mốc dự báo rõ, tiêu chí/nhãn phù hợp độc lập, đánh giá người dùng/chuyên gia và kiểm tra độ ổn định qua trường/năm.

## 8. Đánh giá mức độ sẵn sàng cho báo cáo khoa học

| Hạng mục | Trạng thái hiện tại | Cần làm để báo cáo chặt chẽ |
|---|---|---|
| Câu hỏi dự báo điểm | Ba nhãn điểm Toán, Ngữ văn, Ngoại ngữ đã có trong dữ liệu mô phỏng | Chốt đây là nghiên cứu dự báo điểm mô phỏng, không gọi là hiệu quả tư vấn thực tế |
| Chia dữ liệu | Nhóm nguồn không giao nhau; test là khóa mô phỏng 2026 | Giữ nguyên quy tắc nhóm, công bố tập và seed |
| Baseline | Có điểm lớp 9 tương ứng | Giữ baseline cho mọi mô hình/môn |
| So sánh CatBoost/SVR | Chỉ Toán có notebook so sánh cùng 1.500 dòng; lần notebook SVR chưa hội tụ | Chạy lại cả ba môn với cùng nhóm train, feature, validation/test và đảm bảo SVR hội tụ |
| Tuning | Mỗi mô hình đang có cấu hình đơn lẻ, chưa tìm kiếm có ngân sách cân bằng | Chọn bằng validation/CV nhóm, lưu toàn bộ cấu hình thử |
| Độ bất định | Chưa có khoảng tin cậy/chênh lệch ghép cặp | Bootstrap theo nhóm nguồn hoặc lặp fold nhóm; không coi biến thể cùng nguồn độc lập |
| Kết quả phần trăm | Đã tính tỷ lệ trong ±0,5/±1 điểm trên test | Chốt trước ngưỡng nếu dùng làm metric chính; gọi đúng tên chỉ số |
| Khái quát ngoài mô phỏng | Chưa có dữ liệu thật | Không suy rộng; nếu có dữ liệu thật cần đánh giá ngoài mẫu và tuân thủ quyền riêng tư |

**Khuyến nghị:** xem đây là một nghiên cứu so sánh mô hình hồi quy trong môi trường mô phỏng và bài thực hành machine learning tốt. Trước khi tuyên bố mô hình thắng, cần huấn luyện CatBoost và SVR trên đúng cùng các nhóm train (tối thiểu cho cả ba môn), đảm bảo hội tụ, chọn tham số chỉ bằng validation, rồi đánh giá test đã khóa. Chỉ sau đó mới kết luận về cấu hình cụ thể; để kết luận về thuật toán nói chung cần nhiều cấu hình, nhiều kịch bản hoặc dữ liệu kiểm chứng độc lập.

## 9. Nguồn và tệp nội bộ đã dùng

- CatBoost, [bài báo NeurIPS 2018 về Ordered Boosting và xử lý biến phân loại](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).
- Tài liệu CatBoost: [tham số tuning](https://catboost.ai/docs/en/concepts/parameter-tuning), [biến phân loại](https://catboost.ai/docs/en/features/categorical-features), [cấu trúc cây](https://catboost.ai/docs/en/concepts/algorithm-main-stages_choose-tree-structure).
- Tài liệu scikit-learn: [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html), gồm hàm kernel, tham số và lưu ý độ phức tạp theo số mẫu.
- Mã và kết quả dự án: [`train_models.py`](./train_models.py), [`train_svr_models.py`](./train_svr_models.py), [`models/metadata.json`](./models/metadata.json), [`models/svr_metadata.json`](./models/svr_metadata.json), [`analys/catboost.ipynb`](./analys/catboost.ipynb), [`analys/svr.ipynb`](./analys/svr.ipynb), [`review.md`](./review.md).
