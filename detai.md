# Đề cương đề tài nghiên cứu đề xuất

## Mục lục

1. [Tên đề tài](#1-tên-đề-tài)
2. [Bối cảnh và vấn đề](#2-bối-cảnh-và-vấn-đề)
3. [Mục tiêu nghiên cứu](#3-mục-tiêu-nghiên-cứu)
4. [Câu hỏi và giả thuyết](#4-câu-hỏi-và-giả-thuyết)
5. [Phạm vi](#5-phạm-vi)
6. [Đối tượng và dữ liệu](#6-đối-tượng-và-dữ-liệu)
7. [Phương pháp và thiết kế thực nghiệm](#7-phương-pháp-và-thiết-kế-thực-nghiệm)
8. [Thước đo và tiêu chí thành công](#8-thước-đo-và-tiêu-chí-thành-công)
9. [Đóng góp dự kiến](#9-đóng-góp-dự-kiến)
10. [Lộ trình](#10-lộ-trình)
11. [Rủi ro và đạo đức](#11-rủi-ro-và-đạo-đức)
12. [Việc nhóm cần chốt](#12-việc-nhóm-cần-chốt)
13. [Câu hỏi còn mở](#13-câu-hỏi-còn-mở)

## 1. Tên đề tài

Các phương án sau đây là **[Giả định cần xác nhận]**, bởi repo hiện chưa có dữ liệu trường/nguyện vọng thực:

1. **Khuyến nghị – nếu bài báo tập trung mô hình:** “So sánh CatBoostRegressor và Support Vector Regression trong dự báo điểm thi mô phỏng của học sinh cuối cấp THCS”. Tên này khớp phần đã triển khai nhất, không khẳng định có tư vấn trường.
2. “Dự báo điểm thi và phân tích độ ổn định mô hình trên dữ liệu học sinh mô phỏng: so sánh CatBoost và SVR”. Dùng nếu có thực nghiệm nhiều seed/kịch bản sinh.
3. “Hệ thống hỗ trợ học sinh lớp 9 cân nhắc lựa chọn trường THPT theo điểm dự báo và tiêu chí cá nhân”. Chỉ dùng khi bổ sung dữ liệu trường đáng tin cậy, quy tắc tuyển sinh, học phí/khoảng cách và đánh giá người dùng; hiện chưa đủ bằng chứng trong repo.

Không đề xuất đưa “tư vấn môn học phù hợp” vào tiêu đề dựa trên các nhãn Yes/No hiện tại, vì code chỉ học nhãn quan tâm/lựa chọn quan sát được và không có tiêu chí độc lập về sự phù hợp.

## 2. Bối cảnh và vấn đề

**[Có bằng chứng trong repo]** Tài liệu dự án mô tả mong muốn hỗ trợ học sinh cuối cấp THCS cân nhắc lựa chọn học tập; quan tâm đến điểm số, sở thích, học phí và khoảng cách. Các nhánh thực thi hiện có tập trung vào dự báo ba điểm thi mô phỏng hoặc dự đoán chín nhãn môn; chưa có dữ liệu trường/nguyện vọng thật hay đánh giá của học sinh/chuyên gia (`CATBoost-SVR/Mo_ta.md`, `README.md`, `report.md`).

**[Suy luận của tôi]** Trước khi viết nghiên cứu về DSS chọn trường, cần làm rõ đâu là mô hình prediction và đâu là chính sách ranking. Độ chính xác dự báo điểm không tự chứng minh rằng một trường là lựa chọn tốt, cũng không suy ra xác suất trúng tuyển.

Vấn đề nghiên cứu phiên bản phù hợp với repo nhất là: liệu CatBoostRegressor và SVR có khác biệt về sai số và độ ổn định khi dự đoán nhãn điểm thi được sinh trong cùng điều kiện mô phỏng, cùng dữ liệu train và cùng giao thức đánh giá hay không?

## 3. Mục tiêu nghiên cứu

### Mục tiêu tổng quát

Đánh giá có thể tái lập khả năng dự báo điểm thi mô phỏng của CatBoostRegressor và SVR, trong đúng điều kiện và giới hạn của bộ dữ liệu hiện có, làm cơ sở thiết kế nghiên cứu trên dữ liệu thật sau này.

### Mục tiêu cụ thể

| Mã | Mục tiêu | Sản phẩm đo được |
|---|---|---|
| MT1 | Chốt thời điểm dự báo, target, đặc trưng hợp lệ và semantics của nhãn | Đặc tả bài toán, bảng thời điểm feature/label |
| MT2 | Kiểm tra nguồn dữ liệu, cấu trúc nhóm gốc, thiếu/trùng/miền giá trị | Từ điển dữ liệu, checksum, báo cáo kiểm tra |
| MT3 | So sánh CatBoostRegressor, SVR với baseline | MAE/RMSE/R² trên cùng grouped folds |
| MT4 | Đánh giá độ nhạy với seed/nhóm/mức nhiễu | Trung bình, SD/CI và learning curves |
| MT5 | Phân tích đóng góp nhóm feature và sai số | Ablation, residual/error analysis |
| MT6 | Đề xuất điều kiện cần để mở rộng sang DSS chọn trường | Đặc tả dữ liệu và protocol đánh giá riêng |

## 4. Câu hỏi và giả thuyết

| Câu hỏi | Cách kiểm tra |
|---|---|
| RQ1: CatBoostRegressor và SVR khác nhau ra sao khi dự đoán từng môn? | Cùng nhóm split, feature, target, budget; so MAE/RMSE/R² và CI theo nhóm |
| RQ2: Mô hình học máy có hơn baseline dễ hiểu không? | So với điểm cùng môn lớp gần nhất và Linear/Ridge |
| RQ3: Mức cải thiện phụ thuộc vào số nhóm train thế nào? | Learning curve trên cùng các cỡ mẫu nhóm gốc |
| RQ4: Kết quả đổi thế nào khi bỏ điểm thi thử hoặc thay đổi feature group? | Ablation được định trước, không chọn theo test |
| RQ5: Kết quả có giữ ổn định giữa seed/kịch bản sinh không? | Holdout nhóm và kịch bản sinh độc lập; repeated runs |

Giả thuyết cần kiểm tra, không giả định trước kết quả:

- H1: Ít nhất một mô hình giảm MAE so với baseline điểm gần nhất trên các holdout mô phỏng đã định nghĩa.
- H2: Xếp hạng tương đối giữa CatBoost và SVR có thể thay đổi theo kích thước train hoặc kịch bản sinh.
- H3: Tác động của điểm thi thử lên metric phụ thuộc việc biến này có trước mốc dự báo và mức tương quan do quy tắc sinh.

Không dùng giả thuyết để khẳng định tính mới. Cần tổng quan công trình liên quan trước khi nêu khoảng trống khoa học.

## 5. Phạm vi

### Trong phạm vi đề xuất

- Dự báo hồi quy ba điểm Toán, Ngữ văn, Ngoại ngữ được đánh dấu mô phỏng trong bộ 50k.
- So sánh CatBoostRegressor và SVR với baseline, tập trung vào thiết kế đánh giá công bằng.
- Kiểm tra grouped split do nhiều hồ sơ cùng nhóm nguồn; phân tích sai số và độ nhạy mô phỏng.
- Báo cáo rõ mọi kết quả là trên dữ liệu mô phỏng.

### Ngoài phạm vi với bằng chứng hiện tại

- Xác suất trúng tuyển trường THPT thật hoặc xếp hạng trường đã kiểm chứng.
- Tư vấn “ngành nghề phù hợp”, năng lực cố định, hay quyết định học tập thay học sinh.
- Khẳng định nhãn Yes/No là đăng ký môn/nhóm môn chính thức.
- Tuyên bố mô hình tốt nhất, có tính mới, hoặc có tác động tích cực đến học sinh trước thực nghiệm.
- Dùng K-Means như mô hình dự báo hoặc đánh giá hiệu quả tư vấn.

## 6. Đối tượng và dữ liệu

### Dữ liệu đang có

| Tệp / bộ | Bằng chứng và giới hạn |
|---|---|
| `CATBoost-SVR/dulieuhocsinh_50000_mo_phong.csv` | Theo `CATBoost-SVR/models/metadata.json`: 50.000 hàng, split 30.000/10.000/10.000, 6.000/2.000/2.000 nhóm gốc, seed 42. Có điểm lớp 6–9, điểm thi thử, feature hành vi và ba nhãn `*_mo_phong`. Bộ mô phỏng; không xem như học sinh độc lập 50k. |
| `CATBoost-SVR/dulieuhocsinh_10000.csv` | Tài liệu mô tả 10.000 hồ sơ, 60 cột gồm ID, 50 điểm và 9 nhãn Yes/No. Ý nghĩa cụ thể của Yes/No chưa xác nhận là đăng ký thực tế; nhánh classifier dùng cho task khác. |
| `XGBoost/data/hoc_sinh_moi.csv` | Báo cáo EDA/data ghi 10.000 hồ sơ, 50 cột, 40 feature điểm, 9 nhãn, không thiếu/trùng và mỗi hồ sơ có bốn nhãn Yes. Chất lượng kiểm tra theo báo cáo; nguồn sinh cần xác minh trước diễn giải ngoài repo. |
| `KNN-HQTT/.../data/raw/dulieuhocsinh_10000.csv` | Dùng trong pipeline Logistic Regression/KMeans; 8.000/2.000 split, 50 feature và 9 nhãn theo metadata. Không gộp kết quả với file cùng tên ở nhánh khác nếu chưa đối chiếu checksum. |

### Nguyên tắc dữ liệu

1. Không dùng `ma_hoc_sinh` làm feature.
2. Lưu checksum, tên phiên bản, nguồn sinh và từ điển cột.
3. Chia theo `nhom_ho_so_goc`; mọi biến thể của một hồ sơ chỉ được nằm trong một phân vùng/fold.
4. Xác nhận điểm thi thử và mọi feature có trước mốc dự báo. Cột sau mốc bị loại khỏi X.
5. Kiểm tra miền điểm 0–10, nhãn, thiếu/trùng; không tự điền/xóa khi chưa có chính sách được mô tả.
6. Nếu dùng nhiều bản dữ liệu có cùng tên, lưu đường dẫn, checksum và ID nhóm; không mặc định chúng giống nhau.

## 7. Phương pháp và thiết kế thực nghiệm

### Pipeline dự báo điểm đề xuất

1. **Chốt estimand:** định nghĩa rõ ai/điểm nào được dự đoán, kỳ thi giả lập nào, mốc sử dụng và thời gian có feature.
2. **Tách nhóm độc lập:** tạo grouped train/validation/test hoặc grouped CV; giữ một test theo nhóm/kịch bản sinh hoàn toàn ngoài tuning.
3. **Baseline:** điểm môn gần nhất (đang có trong metadata), Linear Regression/Ridge; tất cả dùng cùng folds.
4. **CatBoostRegressor:** khai báo feature numeric/categorical chính xác; tuning depth, learning rate, iterations/early stopping bằng validation/CV.
5. **SVR:** Pipeline `StandardScaler` + `SVR`; tuning C, epsilon, gamma trong cùng search budget; preprocessing fit bên trong từng fold.
6. **So sánh công bằng:** cùng nhóm train, số nhóm, feature, target, folds và metric. Nếu giới hạn thời gian, báo learning curve ở 300, 1.000, 3.000, 6.000 nhóm thay cho chỉ so một cỡ mẫu.
7. **Đánh giá:** MAE là metric chính dễ diễn giải theo điểm; RMSE nhạy với lỗi lớn; R² làm metric bổ trợ. Báo theo từng môn, fold/seed và khoảng bất định bootstrap theo nhóm.
8. **Ablation:** lịch sử điểm; điểm thi thử (nếu hợp lệ); giờ tự học/chuyên cần; đánh giá riêng tác động feature.
9. **Sensitivity:** nhiều seed/kịch bản sinh độc lập và mức nhiễu; không trộn kết quả tạo sinh vào cùng tập test.
10. **Giải thích:** feature importance/SHAP chỉ là mô tả hành vi mô hình trên phân phối mô phỏng, không diễn giải nhân quả.

### Các mô hình nhãn môn nếu nhóm chọn hướng đó

Định nghĩa labels rõ là dự đoán sở thích/lựa chọn đã ghi nhận. So sánh Logistic Regression, CatBoostClassifier, SVC và XGBoost trên split nhóm tương ứng, với baseline popularity/random; chọn trước metric chính (Macro F1 hoặc precision/recall@4) và báo per-label, Top-4 ranking cùng calibration. Không gọi output là “phù hợp” trừ khi có nhãn độc lập về phù hợp và protocol đánh giá tương ứng.

### Ba lớp hệ thống nếu mở rộng thành DSS

1. **Dự báo điểm:** chỉ đầu ra điểm/khoảng dự báo.
2. **Đánh giá tham chiếu nguyện vọng:** cần ngưỡng tuyển sinh tương thích về trường, năm, phương thức và địa bàn; chênh lệch điểm không tự động là xác suất trúng tuyển.
3. **Xếp hạng lựa chọn cá nhân:** bộ lọc ngân sách/địa bàn và trọng số sở thích/khoảng cách, có công thức giải thích được và người dùng điều chỉnh. Cần đánh giá với dữ liệu trường và người dùng riêng.

## 8. Thước đo và tiêu chí thành công

Tiêu chí hoàn thành nghiên cứu phiên bản mô phỏng:

- Đặc tả target, mốc thời gian, feature, nhóm nguồn và quy tắc sinh có thể truy vết.
- Không có nhóm gốc giao nhau giữa train/validation/test.
- Có baseline và CatBoost/SVR chạy trên cùng protocol; test không tham gia tuning.
- Có metric theo môn, độ bất định qua folds/seeds và phân tích lỗi.
- Có kết quả sensitivity/ablation được báo cả khi giả thuyết không được ủng hộ.
- Mọi bảng kết quả chỉ ghi số từ run log/artifact tái lập; nêu rõ dataset mô phỏng.
- Nếu không có dữ liệu thực/người dùng, kết luận chỉ về dự báo trên các kịch bản mô phỏng đã kiểm tra.

Không đặt một ngưỡng “độ chính xác” tùy ý. Cải thiện cần được đối chiếu baseline, khoảng bất định và tính hợp lệ thực nghiệm.

## 9. Đóng góp dự kiến

**[Giả định cần xác nhận]** Đóng góp có thể là: (a) protocol so sánh CatBoostRegressor/SVR có grouped split trên dữ liệu học sinh mô phỏng; (b) phân tích mức phụ thuộc kết quả vào quy tắc sinh, mốc dự báo và lượng dữ liệu train; (c) đặc tả rõ ranh giới giữa dự báo điểm và xếp hạng quyết định.

Không tuyên bố tính mới hoặc khoảng trống khoa học trước khi rà soát tài liệu. Không coi một giao diện/app hoặc một metric tốt trên dữ liệu mô phỏng là đóng góp tư vấn giáo dục đã được xác nhận.

## 10. Lộ trình

| Giai đoạn | Việc làm | Deliverable |
|---|---|---|
| 1. Chốt phạm vi | Chọn task primary, mốc dự báo, nghĩa nhãn | Đặc tả 1–2 trang và tên đề tài |
| 2. Kiểm toán dữ liệu | Truy nguồn, checksum, data dictionary, nhóm gốc, label generator | Báo cáo nguồn/quality và split plan |
| 3. Thiết kế protocol | Baseline, folds, metrics, search budget, seeds | Kế hoạch thực nghiệm trước khi chạy |
| 4. Huấn luyện | Chạy baseline, CatBoost, SVR cùng groups/folds | Config, logs, predictions, models |
| 5. Kiểm tra độ bền | Repeated grouped CV, ablation, sensitivity | Bảng chính, CI, lỗi, giới hạn |
| 6. Quyết định hướng DSS | Thu dữ liệu trường và đánh giá người dùng nếu có thể | Phác thảo pipeline/đạo đức; nếu thiếu thì giới hạn paper |
| 7. Viết bài | Tổng quan, phương pháp, kết quả, thảo luận | Bản thảo và artifact tái lập |

Thời lượng chưa ấn định vì chưa biết hạn nộp, nhân lực và khả năng lấy dữ liệu thật.

## 11. Rủi ro và đạo đức

- **Mô phỏng:** mô hình có thể học đúng công thức sinh nhưng kém trên học sinh thật. Phải nêu rõ ranh giới này trong abstract/conclusion.
- **Leakage theo thời gian/nhóm:** điểm thi thử và biến thể cùng hồ sơ có thể làm metric lạc quan nếu không chốt mốc và split group.
- **Nhãn gây hiểu nhầm:** Yes/No sở thích không phải năng lực hay phù hợp; không dùng để giới hạn cơ hội học tập.
- **Dữ liệu trẻ vị thành niên:** nếu thu thập dữ liệu thực sau này, cần đồng thuận phù hợp, giảm định danh, giới hạn truy cập và chỉ thu thập biến cần thiết; cần tuân thủ quy trình đạo đức/đơn vị của nhóm.
- **Ngưỡng tuyển sinh thay đổi:** dữ liệu trường phải có năm, nguồn, phương thức tuyển sinh; thông tin cũ có thể gây hại nếu hiển thị như hiện hành.
- **Diễn giải quá mức:** điểm dự báo không phải khả năng đỗ, điểm probability chưa calibration không phải xác suất thật, feature importance không phải quan hệ nhân quả.
- **Tái lập:** model artifacts có thể phụ thuộc version; cần lưu môi trường, checksum, code revision, split IDs và lệnh chạy.

## 12. Việc nhóm cần chốt

1. Chọn bài báo dự báo điểm mô phỏng hay DSS chọn trường. Khuyến nghị hiện tại là tập trung dự báo điểm cho nghiên cứu đầu tiên.
2. Chốt thời điểm dự báo và xác nhận tính hợp lệ của `diem_thi_thu_*`.
3. Xác nhận nguồn/thuật toán sinh target `*_mo_phong`, bộ dữ liệu 10k và nhãn Yes/No.
4. Quyết định CatBoost/SVR có được chạy lại trên cùng số nhóm train và tuning budget không.
5. Nếu giữ DSS: chọn tỉnh/thành, loại trường, kỳ tuyển sinh, dữ liệu học phí/khoảng cách và nguồn cập nhật.
6. Xác định nhóm môn định hướng có phải đầu ra nghiên cứu hay chỉ phạm vi app.
7. Xác định có thể mời học sinh/chuyên gia đánh giá hay paper chỉ dừng ở thực nghiệm kỹ thuật.
8. Chốt hạn nộp, nơi dự kiến gửi và quy định hình thức bài báo.

## 13. Câu hỏi còn mở

- “Học sinh lớp 9” là xác nhận chính thức cho đối tượng hay chỉ giả định theo bộ dữ liệu?
- Ba môn thi mô phỏng đại diện kỳ thi nào và vì sao không có dữ liệu trường?
- Data 50k có seed sinh/config và cách tạo năm bản ghi mỗi nhóm được lưu ở đâu?
- Có dữ liệu thực hoặc khả năng kiểm chứng ngoài mô phỏng không?
- Đơn vị sẽ đánh giá DSS là học sinh, phụ huynh, giáo viên hay chỉ số kỹ thuật?
- Có cần giữ classifier môn tự chọn trong cùng bài, hay chuyển thành nhánh phụ/loại khỏi phạm vi?
