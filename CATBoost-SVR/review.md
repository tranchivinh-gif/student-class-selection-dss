# Đánh giá CatBoostRegressor và SVR

**Phạm vi đánh giá:** các mô hình hồi quy điểm thi trong thư mục `CATBoost-SVR`  
**Ngày rà soát:** 22/10/2026  
**Nguồn bằng chứng:** mã huấn luyện, metadata mô hình và đầu ra đã lưu trong hai notebook. Đây là đánh giá thực nghiệm nội bộ trên bộ dữ liệu hiện có, không phải phản biện độc lập hay kiểm chứng trên dữ liệu học sinh thực tế.

## 1. Tóm tắt kết luận

Hệ thống đang có ba mô hình CatBoostRegressor và ba pipeline SVR để dự báo điểm Toán, Ngữ văn và Ngoại ngữ. Trên bộ test mô phỏng, CatBoost có chỉ số tốt hơn SVR ở cả ba môn trong các artifact hiện có. Tuy nhiên, CatBoost được huấn luyện trên 30.000 dòng, trong khi SVR chỉ dùng 1.500 dòng; các số liệu artifact này không cô lập được ảnh hưởng của thuật toán khỏi ảnh hưởng của lượng dữ liệu huấn luyện.

Phép so sánh kiểm soát được trong `analys/svr.ipynb` chỉ áp dụng cho Toán: hai mô hình dùng cùng 55 đặc trưng, cùng 1.500 dòng train và cùng 10.000 dòng test. Trong phép thử này, cấu hình CatBoost tốt hơn cấu hình SVR theo MAE, RMSE và R². Kết quả là bằng chứng ban đầu có ích, nhưng chưa đủ để kết luận CatBoost nói chung tốt hơn SVR: đây là một môn, một cấu hình mỗi thuật toán, một seed và một bộ dữ liệu mô phỏng.

Các kết quả hiện chứng minh mức độ khớp với **nhãn điểm thi được sinh bằng mô phỏng**. Chúng chưa chứng minh độ chính xác với học sinh thật, hiệu quả tư vấn tuyển sinh, hay khả năng chọn bốn môn tự chọn phù hợp. Bài toán chọn môn cần nhãn và tiêu chí phù hợp riêng; không thể suy ra trực tiếp từ các mô hình hồi quy ba môn thi này.

## 2. Bài toán và dữ liệu được đánh giá

- **Nhãn:** `diem_thi_toan_mo_phong`, `diem_thi_ngu_van_mo_phong`, `diem_thi_ngoai_ngu_mo_phong`, trên thang 0–10.
- **Đầu vào:** 55 đặc trưng cho mỗi môn, gồm 54 đặc trưng chung từ điểm học tập, giờ tự học và chuyên cần, cùng điểm thi thử tương ứng với môn đích. Mã học sinh, mã nhóm nguồn, kết quả tuyển sinh và thông tin tương lai không nằm trong danh sách đặc trưng của các mô hình hồi quy này.
- **Dữ liệu:** `dulieuhocsinh_50000_mo_phong.csv`, 50.000 dòng. Theo mô tả dữ liệu, đây là dữ liệu mô phỏng mở rộng từ 10.000 hồ sơ giả định; mỗi hồ sơ nguồn có năm biến thể có quan hệ phụ thuộc.
- **Phân vùng:** train 30.000 dòng/6.000 nhóm; validation 10.000/2.000 nhóm; test 10.000/2.000 nhóm. Các notebook kiểm tra và ghi nhận không có nhóm nguồn giao nhau giữa ba phần. Phân vùng gắn với các khóa mô phỏng: train 2022–2024, validation 2025, test 2026.
- **Ý nghĩa của test:** đây là kiểm tra trên các nhóm hồ sơ nguồn khác với train và trên một khóa mô phỏng muộn hơn. Vì cùng quy trình sinh dữ liệu, nó chưa tương đương đánh giá chuyển giao sang học sinh hoặc năm tuyển sinh thật.

Tài liệu mô tả nhãn điểm thi mô phỏng được sinh chủ yếu từ điểm học tập lớp 8–9, giờ tự học, chuyên cần và nhiễu. Do các đầu vào có liên hệ được thiết kế với nhãn, chỉ số cao có thể cho thấy mô hình học tốt quy tắc sinh dữ liệu. Đây là giới hạn trọng yếu khi diễn giải kết quả.

## 3. Thiết kế và cấu hình thực nghiệm

| Thành phần | CatBoostRegressor | SVR |
|---|---|---|
| Pipeline | Mô hình boosting trên cây; đầu vào số | `StandardScaler` rồi `SVR` kernel RBF |
| Tham số đã ghi | 1.000 vòng tối đa, depth 8, learning rate 0,05, loss/eval RMSE, seed 42, early stopping 100 vòng | `C=10`, `epsilon=0,1`, `gamma="scale"`, `tol=0,01`, `max_iter=20.000`, seed 42 cho lấy mẫu |
| Dữ liệu train trong artifact | 30.000 dòng, đủ 6.000 nhóm train | Lấy 300 nhóm train bằng seed 42, đủ 1.500 dòng (5 dòng/nhóm) |
| Validation | Dùng chọn vòng dừng sớm; metadata CatBoost hiện không lưu metric validation theo từng môn | 10.000 dòng; metric được lưu cho cả ba môn |
| Test | 10.000 dòng | Cùng 10.000 dòng test |

Pipeline SVR chuẩn hóa đặc trưng sau khi chia tập, nên scaler chỉ được fit trên train. Việc lấy mẫu theo toàn bộ nhóm giữ các biến thể của một hồ sơ trong cùng phần. Các thực hành này là điểm mạnh về tránh rò rỉ và tiền xử lý.

### 3.1. So sánh công bằng hiện có cho môn Toán

Notebook SVR còn huấn luyện CatBoost trên cùng mẫu 300 nhóm/1.500 dòng mà SVR dùng. Hai mô hình dùng chung tập validation/test và 55 đặc trưng. Đây là phép so sánh công bằng hơn để nhận xét về hai **cấu hình** trên môn Toán:

| Mô hình / mốc | Train | MAE test ↓ | RMSE test ↓ | R² test ↑ |
|---|---:|---:|---:|---:|
| CatBoost, mẫu so sánh chung | 1.500 | 0,810 | 1,009 | 0,849 |
| SVR RBF | 1.500 | 0,924 | 1,158 | 0,802 |
| Mốc `toan_lop9` | — | 1,065 | 1,352 | 0,730 |

Trong phép thử này, CatBoost giảm MAE khoảng 12,3% và RMSE khoảng 12,9% so với SVR; R² cao hơn 0,047. So với mốc điểm lớp 9, CatBoost giảm MAE khoảng 23,9%, SVR giảm khoảng 13,2%. Đây là chênh lệch chỉ số quan sát được trên đúng tập test này; chưa có khoảng tin cậy hay kiểm định để xác định độ ổn định của chênh lệch.

Notebook cho biết CatBoost chọn 256 cây (best iteration 255). Mẫu SVR notebook báo `fit_status=1` với `max_iter=5.000`, nghĩa là lần chạy notebook đó chạm giới hạn vòng lặp và chưa hội tụ. Vì thế kết quả notebook này cần được ghi rõ là lần chạy thăm dò, không dùng làm kết quả cuối nếu chưa chạy lại cho hội tụ.

## 4. Kết quả của các mô hình hiện lưu

Bảng sau chép từ `models/metadata.json` và `models/svr_metadata.json`. Mốc là điểm môn lớp 9 tương ứng. Mặc dù test giống nhau, quy mô train khác nhau đáng kể; bảng này mô tả hiệu năng artifact hiện tại, không phải so sánh cô lập thuật toán.

| Môn | Mô hình | Train rows | MAE ↓ | RMSE ↓ | R² ↑ | Mốc lớp 9: MAE / RMSE / R² |
|---|---|---:|---:|---:|---:|---:|
| Toán | CatBoost | 30.000 | 0,755 | 0,951 | 0,866 | 1,065 / 1,352 / 0,730 |
| Toán | SVR | 1.500 | 0,924 | 1,158 | 0,802 | 1,065 / 1,352 / 0,730 |
| Ngữ văn | CatBoost | 30.000 | 0,745 | 0,936 | 0,877 | 0,924 / 1,165 / 0,810 |
| Ngữ văn | SVR | 1.500 | 0,930 | 1,162 | 0,811 | 0,924 / 1,165 / 0,810 |
| Ngoại ngữ | CatBoost | 30.000 | 0,737 | 0,931 | 0,875 | 1,036 / 1,315 / 0,751 |
| Ngoại ngữ | SVR | 1.500 | 0,919 | 1,152 | 0,809 | 1,036 / 1,315 / 0,751 |

CatBoost trong bảng có sai số thấp hơn và R² cao hơn SVR cho cả ba môn, nhưng còn confound bởi lượng train gấp 20 lần. Ở Ngữ văn, SVR có MAE 0,930, cao hơn nhẹ mốc `ngu_van_lop9` 0,924; RMSE và R² của SVR chỉ nhỉnh hơn mốc một lượng nhỏ. Vì vậy chưa thể nói SVR cải thiện ổn định trên mọi môn chỉ dựa vào một chỉ số.

Mô hình CatBoost dùng đủ 30.000 dòng có các kết quả test lần lượt là Toán 0,755/0,951/0,866; Ngữ văn 0,745/0,936/0,877; Ngoại ngữ 0,737/0,931/0,875 (MAE/RMSE/R²). So với mốc lớp 9, MAE giảm lần lượt khoảng 29,1%, 19,4% và 28,9%. Những mức giảm này là trên nhãn mô phỏng và cùng giới hạn ngoại suy nêu trên.

Artifact SVR đã lưu khai báo hội tụ (`fit_status=0`) ở cả ba môn, với 1.373–1.390 support vectors trong 1.500 dòng train. Tỷ lệ support vector cao (khoảng 91,5–92,7%) cho thấy nhiều điểm train ảnh hưởng đến nghiệm; cần đo thời gian dự đoán thực tế nếu triển khai nhiều hồ sơ. Metadata hiện để `fit_seconds` là null nên chưa có bằng chứng định lượng về thời gian huấn luyện. Kết quả SVR đã lưu không được đồng nhất với lần notebook `max_iter=5.000` chưa hội tụ.

## 5. Đánh giá theo tiêu chí

| Tiêu chí | Nhận xét | Mức đánh giá |
|---|---|---|
| Định nghĩa bài toán | Nhãn, thang điểm, ba môn và bài toán hồi quy đã rõ trong mã | Đạt cho thử nghiệm dự báo điểm mô phỏng |
| Độ chính xác dự báo | Đánh giá bằng MAE/RMSE (điểm; càng thấp càng tốt) và R² (càng cao càng tốt), không quy đổi thành phần trăm chính xác. Trên phép thử Toán cùng 1.500 dòng train, CatBoost đạt MAE 0,810, RMSE 1,009, R² 0,849; SVR đạt 0,924, 1,158, 0,802. | CatBoost tốt hơn trong phép thử ghép cặp này; mới xác nhận trên một môn/cấu hình, dữ liệu mô phỏng và một lần chia |
| Chất lượng chia tập | Tách theo nhóm nguồn, test tách nhóm; dùng tập test chung | Tốt trong phạm vi dữ liệu mô phỏng |
| Nguy cơ rò rỉ | Không thấy mã định danh/nhóm hoặc nhãn thi cuối làm đầu vào; cần duy trì kiểm tra này khi thay dữ liệu | Khá |
| Baseline | Có mốc điểm lớp 9; hữu ích để xác định ML có giá trị hơn quy tắc đơn giản không | Tốt, nhưng chỉ một baseline |
| So sánh mô hình | Có so sánh Toán cùng 1.500 dòng; bảng artifact ba môn dùng 30.000 so với 1.500 | Chưa đủ công bằng cho kết luận tổng quát |
| Chọn tham số | Dùng một cấu hình cố định cho mỗi thuật toán; chưa có tìm kiếm/tuning với ngân sách cân bằng | Chưa đủ để đại diện cho thuật toán |
| Độ ổn định/thống kê | Một seed và một test; chưa có sai số ghép cặp, bootstrap theo nhóm hoặc khoảng tin cậy | Chưa được đánh giá |
| Phân tích lỗi | Có MAE/RMSE/R² và feature importance CatBoost; thiếu phân tích phần dư theo mức điểm/khóa/nhóm | Một phần |
| Tái lập | Seed, split, đặc trưng và phần lớn tham số đã có; cần thống nhất notebook và script, lưu phiên bản thư viện/thời gian | Khá, còn việc cần làm |
| Khả năng áp dụng thực tế | Nhãn mô phỏng và trường dữ liệu giả định; chưa có đánh giá người dùng/chuyên gia | Chưa được chứng minh |

## 6. Diễn giải và giới hạn

1. **Không gọi MAE là phần trăm chính xác.** MAE 0,755 có nghĩa sai lệch tuyệt đối trung bình khoảng 0,755 điểm trên thang mô phỏng 0–10.
2. **R² không nói rằng từng học sinh được dự báo đúng ở mức đó.** Đây là chỉ số tổng hợp so với biến thiên của nhãn trong tập đánh giá; cần xem sai số theo từng hồ sơ và nhóm điểm.
3. **Feature importance không phải quan hệ nhân quả.** Với CatBoost Toán, `toan_lop9` có importance 58,60%, `toan_lop8` 18,75%, thi thử Toán 7,72%. Điều này mô tả cách mô hình sử dụng đặc trưng trong bộ mô phỏng; không chứng minh tăng điểm lớp 9 gây ra mức tăng tương ứng ở điểm thi.
4. **Điểm thi thử phải có trước mốc dự báo.** Mã dùng điểm thi thử tương ứng làm đầu vào. Báo cáo nghiên cứu cần định nghĩa rõ ngày/mốc tư vấn và bảo đảm điểm đó thực sự đã có tại thời điểm ấy.
5. **Đầu ra chưa bị ràng buộc trong miền 0–10 ở tầng API.** Cần kiểm tra dự đoán ngoài miền trên test và quyết định quy tắc xử lý, đồng thời báo cáo quy tắc đó; không âm thầm cắt điểm để làm đẹp metric.
6. **Mô hình điểm không phải mô hình chọn môn.** Các artifact CatBoost/SVR này dự đoán điểm Toán, Ngữ văn, Ngoại ngữ. Chúng không đánh giá trực tiếp mức phù hợp của bốn môn tự chọn. Các mô hình phân loại môn quan tâm trong thư mục là CatBoostClassifier/SVC riêng, với nhãn yes/no và một bài toán khác.

## 7. Kết luận có thể đưa vào báo cáo nghiên cứu

Có thể viết rằng: *“Trong thử nghiệm trên bộ dữ liệu mô phỏng 50.000 hồ sơ, các cấu hình CatBoostRegressor và SVR đều tốt hơn mốc điểm lớp 9 ở phần lớn phép đo/môn. Trong thí nghiệm Toán với 1.500 dòng train dùng chung, CatBoost đạt MAE 0,810, RMSE 1,009, R² 0,849; SVR đạt 0,924, 1,158, 0,802. Kết quả ban đầu nghiêng về cấu hình CatBoost được thử nghiệm, nhưng chưa khẳng định ưu thế tổng quát của thuật toán do mới có một cấu hình, một môn trong so sánh train-size-matched và nhãn hoàn toàn mô phỏng.”*

Không nên viết rằng CatBoost đã dự báo chính xác cho học sinh thật, rằng SVR kém hơn trong mọi điều kiện, hoặc rằng hệ thống đã xác định được môn tự chọn phù hợp.

## 8. Việc cần hoàn thành trước khi chốt kết quả nghiên cứu

1. **Tạo bộ so sánh cân bằng cho cả ba môn:** lấy đúng cùng các nhóm nguồn train cho CatBoost và SVR; giữ nguyên 55 đặc trưng, validation, test, seed và baseline. Có thể bổ sung các mức train-size như 1.500, 7.500 và 30.000 dòng để thấy đường cong hiệu năng/chi phí. Báo riêng cấu hình production hiện tại nếu giữ hai mức train khác nhau.
2. **Chọn cấu hình chỉ trên validation:** mô tả phạm vi tìm kiếm và ngân sách tương đương cho hai mô hình; không dùng test để chỉnh tham số. Khóa test cho đến khi toàn bộ lựa chọn đã chốt.
3. **Chạy/ghi metric cho validation và test từng môn:** metadata CatBoost hiện chỉ ghi test metric; nên lưu validation metric, cấu hình cuối, vòng dừng và phiên bản thư viện cho mỗi mô hình.
4. **Làm rõ trạng thái hội tụ:** notebook SVR dùng `max_iter=5.000` và ghi trạng thái chưa hội tụ, còn script/artifact dùng `max_iter=20.000` và hội tụ. Chọn một quy trình được duyệt làm kết quả chính; chạy lại notebook hoặc cập nhật nội dung/đầu ra để notebook và artifact không mâu thuẫn.
5. **Bổ sung phân tích sai số:** phân phối phần dư, MAE theo dải điểm và theo khóa mô phỏng; kiểm tra dự đoán ngoài [0,10]; báo thời gian fit/predict và tài nguyên trên cùng môi trường.
6. **Ước lượng độ bất định:** dùng bootstrap theo `nhom_ho_so_goc` cho chênh lệch sai số ghép cặp trên cùng test; không coi các dòng biến thể cùng nguồn là độc lập. Nếu thay đổi công thức sinh, giữ một kịch bản độc lập để đánh giá độ nhạy.
7. **Tách mục tiêu gợi ý môn thành nghiên cứu riêng:** xác định “phù hợp” bằng tiêu chí/nhãn được định nghĩa trước và có nguồn độc lập (ví dụ lựa chọn thực tế, mức hứng thú/đánh giá sau học, hoặc đánh giá chuyên gia được thiết kế minh bạch). Không dùng nhãn do chính công thức gợi ý tạo ra làm bằng chứng duy nhất rằng gợi ý tốt.

## 9. Tệp đã rà soát

- [`train_models.py`](./train_models.py) và [`train_svr_models.py`](./train_svr_models.py): pipeline huấn luyện, cách chia/ lấy mẫu, tham số và metric.
- [`models/metadata.json`](./models/metadata.json) và [`models/svr_metadata.json`](./models/svr_metadata.json): kết quả artifact hiện lưu.
- [`analys/catboost.ipynb`](./analys/catboost.ipynb): CatBoost Toán với đủ 30.000 dòng train và kết quả validation/test.
- [`analys/svr.ipynb`](./analys/svr.ipynb): so sánh thăm dò Toán trên 1.500 dòng train và trạng thái hội tụ của lần chạy notebook.
- [`Mo_ta.md`](./Mo_ta.md), [`mo_ta_dl.md`](./mo_ta_dl.md): phạm vi nghiên cứu, nguồn dữ liệu, cách sinh mô phỏng và giới hạn diễn giải.

## 10. Tỷ lệ dự đoán nằm trong ngưỡng sai số trên tập test

> Phần trăm dưới đây được tính lại bằng cách chạy các artifact CatBoost và SVR hiện lưu trên 10.000 dòng test cho từng môn, rồi đếm tỷ lệ có `|điểm dự đoán − điểm mô phỏng thực tế|` không vượt quá ngưỡng. Mốc so sánh là điểm lớp 9. Các con số là tỷ lệ trong ngưỡng sai số, không phải accuracy phân loại và không có nghĩa là tỷ lệ dự đoán hoàn toàn chính xác.

| Môn | Mô hình | Train rows | Sai số ≤ ±0,5 điểm | Sai số ≤ ±1,0 điểm |
|---|---|---:|---:|---:|
| Toán | CatBoost | 30.000 | 41,00% | 70,60% |
| Toán | SVR | 1.500 | 33,65% | 61,03% |
| Toán | Mốc `toan_lop9` | — | 30,40% | 54,69% |
| Ngữ văn | CatBoost | 30.000 | 41,35% | 71,42% |
| Ngữ văn | SVR | 1.500 | 32,78% | 60,92% |
| Ngữ văn | Mốc `ngu_van_lop9` | — | 34,16% | 61,04% |
| Ngoại ngữ | CatBoost | 30.000 | 41,86% | 72,35% |
| Ngoại ngữ | SVR | 1.500 | 33,33% | 61,29% |
| Ngoại ngữ | Mốc `ngoai_ngu_lop9` | — | 31,37% | 55,95% |

**Diễn giải:** với artifact hiện lưu, khoảng 41,00–41,86% dự đoán CatBoost nằm trong ±0,5 điểm và 70,60–72,35% nằm trong ±1,0 điểm. Với SVR, các tỷ lệ tương ứng là 32,78–33,65% và 60,92–61,29%. Mốc điểm lớp 9 có 30,40–34,16% trong ±0,5 điểm và 54,69–61,04% trong ±1,0 điểm. Kết quả này nhất quán với MAE/RMSE rằng CatBoost lưu hiện tại sát nhãn mô phỏng hơn SVR và mốc trên tập test, nhưng phép so sánh CatBoost–SVR vẫn bị ảnh hưởng bởi chênh lệch số dòng train.

Hai ngưỡng ±0,5 và ±1,0 điểm là cách trình bày bổ sung, không phải chuẩn đánh giá phổ quát. Nếu dùng làm chỉ số chính trong báo cáo nghiên cứu, nhóm cần giải thích ý nghĩa thực tế của ngưỡng và chốt trước khi xem kết quả test. Các tỷ lệ trên cũng chỉ đo độ gần với nhãn mô phỏng, không phải xác suất đúng với điểm thi thật.

## 11. Các tham số có thể điều chỉnh

Bảng dưới phân biệt **cấu hình hiện được ghi trong mã/artifact** với **tham số có thể đưa vào tuning**. “Giá trị hiện tại” không đồng nghĩa đó là giá trị tối ưu. Nên chọn tham số bằng validation hoặc cross-validation theo nhóm nguồn trong phần train; không chọn theo kết quả test.

### 11.1. CatBoostRegressor

| Tham số | Giá trị hiện tại | Vai trò và hướng điều chỉnh |
|---|---:|---|
| `iterations` | 1.000 tối đa | Số vòng boosting/số cây tối đa. Có thể tăng, nhưng nên dùng early stopping trên validation; số cây thực tế hiện dừng ở 253–266 tùy môn khi huấn luyện đủ 30.000 dòng. |
| `depth` | 8 | Độ sâu cây, ảnh hưởng độ phức tạp. Thử lưới nhỏ như 4, 6, 8, 10; cây sâu hơn có thể học tương tác tốt hơn nhưng tăng nguy cơ khớp quá mức và chi phí. |
| `learning_rate` | 0,05 | Mức cập nhật mỗi vòng. Thử thấp hơn/cao hơn cùng với `iterations`; learning rate thấp thường cần nhiều vòng hơn. |
| `loss_function` | `RMSE` | Hàm mục tiêu tối ưu khi huấn luyện. Giữ RMSE nếu mục tiêu là phạt lỗi lớn; có thể thử `MAE` nếu muốn tối ưu trực tiếp sai số tuyệt đối, nhưng phải nêu lý do và đánh giá bằng cùng bộ chỉ số. |
| `eval_metric` | `RMSE` | Chỉ số dùng theo dõi validation/early stopping. Có thể đổi sang `MAE` nếu nghiên cứu chốt MAE là tiêu chí chọn mô hình; không nên đổi sau khi xem test. |
| `early_stopping_rounds` | 100 | Số vòng không cải thiện trước khi dừng. Thử một vài giá trị định trước (ví dụ 50, 100, 150) theo validation. |
| `l2_leaf_reg` | Mặc định CatBoost | Regularization L2 cho lá cây. Đưa vào tuning để kiểm soát độ phức tạp; cần ghi rõ giá trị thay vì dựa vào mặc định ngầm. |
| `random_strength` | Mặc định CatBoost | Mức ngẫu nhiên khi chọn điểm chia, có thể ảnh hưởng regularization và độ ổn định. Chỉ tuning sau khi cố định giao thức và seed. |
| `bagging_temperature` / `subsample` | Mặc định CatBoost | Điều khiển mức lấy mẫu ngẫu nhiên tùy `bootstrap_type`. Chọn tham số tương thích với bootstrap đã khai báo; tránh đưa cả hai vào một lưới nếu cấu hình bootstrap không cho phép. |
| `random_seed` | 42 | Seed ngẫu nhiên; cố định trong so sánh chính để tái lập. Sau khi chọn cấu hình, chạy thêm seed khác để khảo sát độ ổn định, không coi các seed trên cùng test là mẫu độc lập. |

Trong `train_models.py`, depth, learning rate, loss/eval metric, số vòng tối đa, seed và early stopping đã được đặt tường minh. Các tham số ghi “mặc định CatBoost” hiện không được khai báo trong mã; nếu đưa vào báo cáo chính thức, nên lưu giá trị thật hoặc khai báo tường minh trong cấu hình thực nghiệm.

### 11.2. SVR (RBF) và bước chuẩn hóa

| Tham số | Giá trị hiện tại trong artifact | Vai trò và hướng điều chỉnh |
|---|---:|---|
| `kernel` | `rbf` | Dạng hàm kernel. Có thể so sánh `linear` với `rbf`; `poly` chỉ nên thử nếu có giả thuyết về tương tác đa thức và cần công bố thêm bậc. Không chọn kernel theo test. |
| `C` | 10,0 | Mức phạt lỗi/độ regularization. Thử theo thang log (ví dụ 0,1; 1; 10; 100); C lớn khớp train chặt hơn nhưng dễ nhạy với nhiễu. |
| `epsilon` | 0,1 | Bề rộng vùng sai số không bị phạt trong hàm mục tiêu SVR. Thử vài giá trị theo độ phân giải và ý nghĩa của thang điểm, chẳng hạn 0,05; 0,1; 0,2; 0,5. |
| `gamma` | `scale` | Phạm vi ảnh hưởng của một điểm với kernel RBF. So sánh `scale`, `auto` và một dải giá trị log; tương tác với chuẩn hóa và `C`. |
| `tol` | 0,01 | Ngưỡng dừng thuật toán. Có thể giảm để yêu cầu nghiệm chính xác hơn, nhưng thời gian tăng; phải xác nhận `fit_status_ == 0`. |
| `max_iter` | 20.000 trong artifact; 5.000 ở lần chạy notebook thăm dò | Giới hạn số vòng lặp. Tăng khi chưa hội tụ; không báo metric như kết quả cuối nếu thuật toán chạm giới hạn. Mã huấn luyện hiện kiểm tra hội tụ và dừng nếu SVR chưa hội tụ. |
| `shrinking` | `True` (mặc định sklearn) | Bật/tắt heuristic shrinking có thể ảnh hưởng tốc độ hội tụ. Có thể thử như tuning phụ, không phải ưu tiên đầu tiên. |
| `cache_size` | 200 MB (mặc định sklearn) | Bộ nhớ cache kernel, chủ yếu ảnh hưởng thời gian/tài nguyên. Chỉ điều chỉnh khi đo trên cùng phần cứng; không xem đây là tham số chất lượng dự báo chính. |
| `StandardScaler.with_mean`, `with_std` | `True`, `True` (mặc định) | Chuẩn hóa trung bình và độ lệch chuẩn. SVR nhạy với thang đo nên giữ scaler trong Pipeline; scaler phải chỉ fit trên train ở mỗi fold. |

`train_svr_models.py` đặt rõ kernel, C, epsilon, gamma, tolerance và giới hạn vòng lặp. `shrinking`, `cache_size`, cùng cấu hình `StandardScaler` đang dựa vào mặc định của thư viện; nếu kết quả cần tái lập giữa môi trường, hãy ghi phiên bản scikit-learn và lưu cấu hình đầy đủ.

### 11.3. Giao thức tuning đề xuất

1. Chốt trước metric chọn mô hình (đề xuất MAE chính; RMSE và R² bổ sung) cùng ngưỡng sai số phần trăm nếu sẽ báo cáo.
2. Dùng cùng các nhóm nguồn train cho CatBoost và SVR ở từng mức dữ liệu; giữ validation chung. Nếu dùng cross-validation, chia fold theo `nhom_ho_so_goc`.
3. Dùng lưới nhỏ hoặc randomized search với ngân sách/chiến lược có thể so sánh; lưu mọi cấu hình đã thử và kết quả validation.
4. Chọn cấu hình bằng validation, huấn luyện lại theo quy trình đã định, rồi chỉ đánh giá test một lần. Báo cấu hình, seed, thời gian fit/predict, hội tụ SVR và số dòng/nhóm dùng để train.
5. Sau khi chọn cấu hình, có thể khảo sát nhiều seed hoặc bootstrap theo nhóm để đánh giá độ ổn định; không dùng việc thử nhiều cấu hình trên test để tìm ra kết quả tốt nhất.
