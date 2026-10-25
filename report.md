# Báo cáo rà soát repo và đánh giá mô hình

## Mục lục

1. [Tóm tắt điều hành](#1-tóm-tắt-điều-hành)
2. [Phạm vi và giới hạn](#2-phạm-vi-và-giới-hạn)
3. [Inventory repo và tác vụ](#3-inventory-repo-và-tác-vụ)
4. [Phân tích từng mô hình](#4-phân-tích-từng-mô-hình)
5. [So sánh tổng hợp](#5-so-sánh-tổng-hợp)
6. [Ma trận chấm điểm](#6-ma-trận-chấm-điểm)
7. [Kiểm chứng chéo và rủi ro](#7-kiểm-chứng-chéo-và-rủi-ro)
8. [Đề xuất cải thiện](#8-đề-xuất-cải-thiện)
9. [Kết hợp dự báo với tiêu chí cá nhân](#9-kết-hợp-dự-báo-với-tiêu-chí-cá-nhân)
10. [Pipeline đề nghị cho bài báo](#10-pipeline-đề-nghị-cho-bài-báo)
11. [Câu hỏi còn mở](#11-câu-hỏi-còn-mở)
12. [Phụ lục: phạm vi đọc](#12-phụ-lục-phạm-vi-đọc)

## 1. Tóm tắt điều hành

- [Có bằng chứng trong repo] Repo có ba hướng tác vụ khác nhau: dự báo 3 điểm thi mô phỏng bằng CatBoostRegressor/SVR; dự đoán 9 nhãn quan tâm môn bằng CatBoostClassifier/SVC, Logistic Regression và XGBoost rồi lấy Top 4; và phân cụm điểm bằng K-Means. Không nhánh nào hiện thực đầy đủ gợi ý trường THPT theo học phí, khoảng cách và quy chế tuyển sinh.
- [Có bằng chứng trong repo] Nhánh điểm thi ở `CATBoost-SVR` dùng bộ mô phỏng 50.000 dòng, chia sẵn 30k/10k/10k theo nhóm hồ sơ gốc, seed 42. CatBoost và SVR đều có baseline điểm lớp 9 và metric test trong metadata.
- [Có bằng chứng trong repo] CatBoostRegressor thắng SVR theo MAE/RMSE trên cả ba môn test; nhưng đây là kết quả trên nhãn mô phỏng. SVR chỉ học 1.500 dòng từ 300 nhóm train để giảm chi phí, trong khi CatBoost học toàn bộ 30.000 dòng train. So sánh hiện tại không cùng ngân sách dữ liệu.
- [Có bằng chứng trong repo] Nhánh dự đoán nhãn 9 môn có báo cáo test cho CatBoost/SVC, Logistic Regression và XGBoost; các nhãn được mô tả là lựa chọn/quan tâm ghi nhận, không được xác nhận là đăng ký học thật. XGBoost Macro F1 test 0,5345, Exact Match 0,0293 khi buộc chọn đúng 4 môn; baseline 4 nhãn phổ biến có Macro F1 0,2751 và Exact Match 0,0107 (`XGBoost/reports/evaluation_report.md`).
- [Suy luận của tôi] CatBoostRegressor nên là mô hình chính để tái kiểm chứng bài toán điểm thi; SVR nên là đối chứng bắt buộc nhưng cần được chạy lại với cùng train groups và ngân sách tuning. Với nhãn quan tâm, XGBoost là đối chứng multi-label mạnh nhất có báo cáo đang đọc được, nhưng task này khác mục tiêu tư vấn trường.
- Rủi ro lớn nhất: sai khác giữa đề tài trong tài liệu (chọn trường THPT) và nhãn/code hiện có (điểm thi mô phỏng hoặc sở thích môn); thêm vào đó, các kết quả mô phỏng có thể phản ánh trực tiếp quy tắc sinh dữ liệu.

## 2. Phạm vi và giới hạn

Rà soát chỉ đọc file, không chạy huấn luyện, không tải dữ liệu ngoài, không sửa code/dữ liệu/báo cáo gốc. Tệp này tổng hợp bằng chứng repo hiện có, không phải kiểm toán độc lập trên dữ liệu hoặc tái lập thí nghiệm. Số liệu dưới đây được chép từ metadata/báo cáo đã lưu và phải được xem là **kết quả được repo báo cáo**, chưa được chạy lại trong lượt rà soát.

Quy ước nhãn:

- **[Có bằng chứng trong repo]**: được ghi trong code, dữ liệu mô tả, artifact metadata hoặc báo cáo hiện có.
- **[Suy luận của tôi]**: đánh giá phương pháp từ các bằng chứng đó.
- **[Giả định cần xác nhận]**: lựa chọn thiết kế nhóm chưa chốt hoặc repo không chứng minh.

Điểm quan trọng về phạm vi: `README.md` mô tả hệ thống dự đoán bốn môn tự chọn trong chín nhãn. `CATBoost-SVR/Mo_ta.md` đề xuất nghiên cứu chọn trường THPT, nhưng tài liệu đó cũng ghi đây là định hướng chưa có dữ liệu trường/nguyện vọng thực tế. Hai việc không thể gộp thành cùng một mục tiêu đánh giá. Đề tài cần chọn rõ: (a) dự báo điểm thi mô phỏng; (b) dự đoán nhãn quan tâm môn; hoặc (c) hỗ trợ chọn trường, cần dữ liệu và tiêu chí trường riêng.

## 3. Inventory repo và tác vụ

| Nhánh / vị trí | Mô hình, nội dung | Task thực tế | Dữ liệu và đầu ra | Đánh giá / ghi chú |
|---|---|---|---|---|
| `CATBoost-SVR/train_models.py`, `train_svr_models.py`, `models/metadata.json`, `models/svr_metadata.json` | 3 CatBoostRegressor và 3 pipeline StandardScaler+RBF-SVR | Regression: dự báo `diem_thi_*_mo_phong` cho Toán, Ngữ văn, Ngoại ngữ | `dulieuhocsinh_50000_mo_phong.csv`; metadata cùng model `.cbm`/`.joblib` | MAE, RMSE, R²; baseline là điểm lớp 9 |
| `CATBoost-SVR/train_elective_models.py`, `models/elective_metadata.json` | 9 CatBoostClassifier và 9 SVC | 9 bài toán binary độc lập về nhãn Yes/No; giao diện xếp xác suất lấy Top 4 | `dulieuhocsinh_10000.csv`; artifacts và metric theo nhãn | Accuracy, Precision, Recall, F1, ROC-AUC; không phải multi-label estimator chung |
| `CATBoost-SVR/analys/*.ipynb`, `report.md`, `review.md`, `pratice_cat.md` | Notebook và tài liệu phân tích/thực hành | Chủ yếu regression điểm Toán; tài liệu cũng nêu các hướng nghiên cứu | Notebook có cell/code/output đã lưu; báo cáo có thể không khớp code hiện hành | Đọc lại kết quả notebook cần lưu ý phiên bản và tình trạng execution |
| `XGBoost/{eda_report,preprocessing,train,evaluate,analyze}.py`, `run_pipeline.py` | MultiOutputClassifier(XGBClassifier) | 9 nhãn binary độc lập, rồi chọn xác suất 4 nhãn cao nhất | `hoc_sinh_moi.csv` 10.000 dòng, 40 điểm + 9 nhãn; báo cáo, dự đoán test, model pickle | Chia 70/15/15, seed 42; chọn 6 cấu hình trên validation; test Macro/Micro F1, Exact Match; baseline 4 nhãn phổ biến |
| `KNN-HQTT/Logistic_Regression_TuVanToHop/src/step_*.py` | MultiOutputClassifier(LogisticRegression); Pipeline imputer/scaler | 9 nhãn Yes/No, mỗi nhãn phân loại riêng; Top 4 minh họa | CSV 10.000; 8.000 train/2.000 test, artifacts và báo cáo | Accuracy/Precision/Recall/F1/ROC-AUC từng nhãn; seed 42; không có bằng chứng CV hay baseline |
| `KNN-HQTT/.../step_08_train_kmeans.py`, `step_09_evaluate_kmeans.py` | K-Means với imputer và scaler | Phân cụm không giám sát các vector điểm | Cùng bộ 10.000, 8k/2k; metadata và silhouette | k chọn trên mẫu train; silhouette test không chứng minh cụm có ý nghĩa giáo dục |

### Mô hình nào gắn với đề tài hiện tại?

| Mô hình | Task đúng theo triển khai | Quan hệ với chọn trường THPT |
|---|---|---|
| CatBoostRegressor, SVR | Dự báo điểm thi mô phỏng liên tục của 3 môn | Có thể là thành phần dự báo nếu đích và mốc dự báo được xác nhận; không dự báo đỗ trường |
| CatBoostClassifier, SVC, Logistic Regression, XGBoost | Ước lượng 9 nhãn Yes/No liên quan sở thích/lựa chọn môn | Không thay thế mức phù hợp, năng lực, lựa chọn nhóm môn chính thức hay nguyện vọng trường |
| K-Means | Gom nhóm hồ sơ điểm theo khoảng cách trong không gian đã chuẩn hóa | Không dự báo điểm, môn lựa chọn hoặc trường; chỉ dùng khám phá nếu cụm được diễn giải/kiểm chứng |

Không thấy triển khai `CatBoostClassifier` cho kết quả tuyển sinh trường, `SVC` cho khả năng trúng tuyển, ranking trường, hay tính đa tiêu chí học phí/khoảng cách. Không thấy nhãn tuyển sinh trường trong dữ liệu 10k theo `CATBoost-SVR/mo_ta_dl.md` và các hướng dẫn repo.

## 4. Phân tích từng mô hình

### 4.1. CatBoostRegressor — dự báo điểm thi mô phỏng

- **Mục tiêu và dữ liệu.** [Có bằng chứng] Mỗi mô hình dự báo một trong ba cột `diem_thi_toan_mo_phong`, `diem_thi_ngu_van_mo_phong`, `diem_thi_ngoai_ngu_mo_phong`. Metadata ghi dữ liệu mô phỏng, scale 0–10, train/validation/test = 30.000/10.000/10.000, số nhóm tương ứng 6.000/2.000/2.000, seed 42 (`CATBoost-SVR/models/metadata.json`).
- **Đầu vào.** Các điểm lớp 6–9, `gio_tu_hoc_tuan`, `ti_le_chuyen_can`, và điểm thi thử cùng môn. Bỏ ID/nhóm khỏi đặc trưng. Điểm thi thử chỉ hợp lệ nếu mốc tư vấn xảy ra sau khi có kết quả thi thử; nhóm phải chốt mốc này.
- **Thiết lập.** `iterations=1000`, `depth=8`, `learning_rate=0.05`, loss/eval RMSE, early stopping 100, seed 42; validation dùng để chọn vòng dừng (`CATBoost-SVR/train_models.py`). Không khai báo categorical feature; theo metadata các features là số nên điều này không tạo lỗi rõ ràng.
- **Metric được lưu trên test mô phỏng:**

| Môn | MAE | RMSE | R² | Baseline MAE | Baseline RMSE | Baseline R² | Cây |
|---|---:|---:|---:|---:|---:|---:|---:|
| Toán | 0,7552 | 0,9510 | 0,8663 | 1,0650 | 1,3518 | 0,7298 | 256 |
| Ngữ văn | 0,7450 | 0,9361 | 0,8771 | 0,9238 | 1,1652 | 0,8096 | 253 |
| Ngoại ngữ | 0,7374 | 0,9311 | 0,8751 | 1,0364 | 1,3146 | 0,7510 | 266 |

  Nguồn: `CATBoost-SVR/models/metadata.json`, đối chiếu cấu hình tại `CATBoost-SVR/train_models.py`. Baseline là điểm môn tương ứng lớp 9. Kết quả chỉ mô tả test mô phỏng.
- **Ưu điểm.** Có chia tập theo nhóm; baseline dễ hiểu; lưu cấu hình, feature, best iteration và metric. CatBoost thường xử lý tốt quan hệ phi tuyến và không yêu cầu scale đầu vào.
- **Rủi ro/giới hạn.** Cột điểm thi thử có thể là thông tin sau mốc dự báo. Nhãn đầu ra tổng hợp từ cơ chế mô phỏng có thể phụ thuộc mạnh vào điểm lớp gần nhất. Tài liệu thực hành notebook và script cho biết cùng quy trình 50k, nhưng notebook có kết quả từng thời điểm; cần đảm bảo báo cáo dùng đúng artifact/model hiện hành. Không có cross-validation, dải bất định, ablation hoặc đánh giá ngoài mô phỏng.
- **Kết luận tạm.** [Suy luận] Ứng viên tốt để làm nhánh regression chính khi nhóm xác nhận mốc dự báo, nhãn và quy tắc sinh. Chưa đủ bằng chứng dùng làm tư vấn thực tế.

### 4.2. SVR — đối chứng regression

- **Mục tiêu và nhãn.** Cùng ba nhãn điểm mô phỏng CatBoost, cùng feature list theo metadata CatBoost.
- **Thiết lập.** Pipeline `StandardScaler` → `SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale', tol=0.01, max_iter=20000)`. Seed 42 được dùng lấy ngẫu nhiên 300 nhóm từ train; 1.500 dòng (5 hồ sơ/nhóm) dùng để fit, so với toàn bộ 30.000 dòng train của CatBoost. Validation 10.000, test 10.000. Metadata báo hội tụ, không warnings (`CATBoost-SVR/train_svr_models.py`, `models/svr_metadata.json`).
- **Test và baseline:**

| Môn | MAE | RMSE | R² | Baseline MAE | Baseline RMSE | Baseline R² |
|---|---:|---:|---:|---:|---:|---:|
| Toán | 0,9243 | 1,1578 | 0,8018 | 1,0650 | 1,3518 | 0,7298 |
| Ngữ văn | 0,9305 | 1,1619 | 0,8107 | 0,9238 | 1,1652 | 0,8096 |
| Ngoại ngữ | 0,9193 | 1,1515 | 0,8090 | 1,0364 | 1,3146 | 0,7510 |

  Nguồn: `CATBoost-SVR/models/svr_metadata.json`. SVR tốt hơn baseline rõ với Toán/Ngoại ngữ theo ba metric; với Ngữ văn MAE kém baseline 0,0067 và RMSE/R² chỉ nhỉnh rất ít. CatBoost tốt hơn SVR trên cả 3 môn, nhưng so sánh bị nhiễu bởi chênh lệch 20 lần số dòng train.
- **Ưu điểm.** Có scaling trong pipeline, metric validation/test, seed cho lấy nhóm train, kiểm tra hội tụ, lưu support vector count. Là đối chứng phi tuyến hợp lý cho các feature số.
- **Rủi ro/giới hạn.** Không có tuning/CV thể hiện; chỉ một cấu hình cố định và tập con train 1.500 dòng. RBF-SVR khó giải thích từng dự đoán; dự báo ngoài miền có thể kém. Không có thời gian fit/inference đo được. Chia theo nhóm áp dụng cho train/validation/test được kiểm tra, nhưng tính chất phụ thuộc nguồn được xác nhận chủ yếu qua tài liệu dữ liệu.
- **Kết luận tạm.** Dùng có điều kiện làm đối chứng, sau khi chạy lại trên cùng phân vùng, toàn bộ hoặc các cỡ train được ghép cặp và ngân sách tuning như CatBoost.

### 4.3. CatBoostClassifier và SVC — nhãn quan tâm 9 môn

- **Task.** 9 phân loại nhị phân độc lập trên `dulieuhocsinh_10000.csv`, không phải classifier tuyển sinh. Script bỏ `ma_hoc_sinh`, dùng 50 cột điểm làm đầu vào, labels yes/no chuyển 1/0; split ngẫu nhiên train/validation/test 6.000/2.000/2.000 theo seed 42 (`CATBoost-SVR/train_elective_models.py`, `models/elective_metadata.json`).
- **Tiền xử lý.** CatBoost nhận điểm dạng số, không có categorical features. SVC dùng `StandardScaler` + SVC RBF và `probability=True`. Mỗi nhãn được fit riêng; dự đoán Top 4 bằng xác suất từng nhãn. Không thấy `class_weight`; dữ liệu 10k được mô tả gần cân bằng. Chia random theo dòng phù hợp nếu mỗi ID duy nhất và không có họ hồ sơ biến thể; báo cáo/metadata ghi ID không trùng và split độc lập.
- **Metric lưu.** Trung bình test F1 theo nhãn: CatBoost khoảng 0,713; SVC khoảng 0,709 (trung bình tính từ 9 số F1 trong `elective_metadata.json`; mỗi nhãn có metric riêng). ROC-AUC từng nhãn xấp xỉ 0,768–0,809 ở CatBoost và 0,762–0,796 ở SVC. `top_four_mean_interest_match` lần lượt 0,5989/0,5984, còn exact match của 9 nhãn theo ngưỡng 0,5 là 0,7176/0,7111. Hai đại lượng sau có ý nghĩa khác nhau: top-four mean là tỷ lệ nhãn Yes trung bình trong 4 nhãn cao nhất; exact match so toàn vector 9 nhãn, không chứng minh chọn đúng tổ hợp chính thức.
- **Ưu điểm.** Có metric theo từng nhãn, validation/test; SVC có scale; CatBoost/SVC test tương đối gần nhau, nên không có cơ sở tuyên bố một họ áp đảo.
- **Rủi ro/giới hạn.** Tài liệu nguồn không xác nhận các Yes/No là đăng ký thực tế. Không có ràng buộc “mỗi học sinh chọn đúng 4” trong training labels theo metadata; giao diện chọn Top 4 do chính sách hậu xử lý. Không có baseline phù hợp trong metadata, threshold tuning, calibration hay CV. Việc cùng lúc báo vector exact match và Top-4 có thể gây hiểu lầm nếu không định nghĩa trước.
- **Kết luận tạm.** Không dùng để kết luận môn “phù hợp” hoặc trường “khả thi”. Nếu mục tiêu là dự đoán lựa chọn quan sát được, nên đánh giá riêng Top-K/ranking với nhãn đã xác minh và baseline.

### 4.4. XGBoost — multi-label 9 môn, lấy Top 4

- **Task/dữ liệu.** `MultiOutputClassifier(XGBClassifier)` fit một binary classifier trên từng nhãn; `selection.py` lấy đúng bốn xác suất cao nhất. Bộ báo cáo chỉ ra 10.000 hồ sơ, 40 feature điểm, 9 labels, mỗi dòng gốc chọn bốn nhãn; train/validation/test 7.000/1.500/1.500, seed 42 (`XGBoost/reports/training_report.json`, `data_report.md`).
- **Thiết lập.** Sáu cấu hình XGBoost được so sánh trên validation theo Macro F1, tie-break Exact Match rồi Micro F1; chọn config depth 2, learning rate 0,05, 240 estimators, subsample/colsample 0,8, min_child_weight 3, alpha 0,1, lambda 2. Fit cuối trên train+validation; test chỉ đánh giá cuối (`XGBoost/train.py`). Baseline là bốn nhãn phổ biến nhất trong train.
- **Kết quả test được lưu:** Macro F1 0,5345; Micro F1 0,5352; Exact Match 0,0293 so baseline lần lượt 0,2751; 0,4483; 0,0107 (`XGBoost/reports/evaluation_report.md`). Tỷ lệ chọn đúng bốn môn trong test là 100%, nghĩa là metric Exact Match có trần cơ học do dự đoán luôn đúng bốn nhãn; exact match thấp cho thấy bộ dự đoán chỉ khớp cả bốn nhãn ở khoảng 2,93% mẫu. Kết quả per-label F1 nằm khoảng 0,484–0,585.
- **Ưu điểm.** Pipeline lưu đầy đủ audit, split, sáu trial validation, test predictions, baseline; tiền xử lý không tự xóa/sửa hàng lỗi; ID bị loại khỏi X; nhãn không dùng để suy ra feature. Có tests trong repo nhưng lượt này không chạy test.
- **Rủi ro/giới hạn.** Split theo dòng chứ không theo nhóm nguồn; đối với CSV hiện hành có một hàng/ID nên không thấy nhóm phụ thuộc trong báo cáo, nhưng nếu dùng các bản mở rộng/biến thể thì cần group split. Không có CV, calibration hay nhiều seed. Tên `hoc_sinh_moi.csv` và nội dung file có khả năng là dữ liệu tạo mới; báo cáo ghi rõ không coi là dữ liệu thật. Feature importance không phải giải thích nhân quả.
- **Kết luận tạm.** Ứng viên baseline/mô hình so sánh cho task xếp hạng nhãn quan sát được, không phải cho lựa chọn trường. Lợi ích so baseline trên test chỉ hợp lệ trong bộ này và cần kiểm tra nguồn sinh/độc lập.

### 4.5. Logistic Regression — multi-output nhãn quan tâm

- **Task và cấu hình.** 9 classifiers logistic nhị phân trong `MultiOutputClassifier`, Pipeline median imputer + StandardScaler + LogisticRegression. Chia 80/20 random_state 42; 50 điểm làm feature, ID bỏ; `class_weight` tắt theo ghi chú gần cân bằng (`KNN-HQTT/.../README.md`, `src/step_03_prepare_dataset.py`, `src/step_04_train_logistic_regression.py`).
- **Kết quả test đã lưu.** 2.000 mẫu. Accuracy theo nhãn khoảng 0,700–0,725; ROC-AUC khoảng 0,772–0,807; F1 lớp Yes khoảng 0,702–0,727 (`outputs/step_05/bao_cao_danh_gia.txt`). Không thấy baseline hoặc cross-validation trong artifact/report.
- **Ưu điểm.** Mô hình đơn giản, có thể đọc hệ số sau chuẩn hóa, pipeline fit imputer/scaler trong train, script tái lập theo seed.
- **Rủi ro/giới hạn.** Không có tuning/threshold tuning hoặc kiểm định calibration. Report metrics per-label nhưng không có đánh giá Top-4 độc lập đầy đủ trong report. Dữ liệu/nhãn mô phỏng hoặc chưa xác minh; không dự đoán điểm thi/trường.
- **Kết luận tạm.** Nên giữ làm baseline tuyến tính cho các thử nghiệm 9 nhãn, sau khi thống nhất đúng dataset version và metric multi-label.

### 4.6. K-Means — phân cụm khám phá

- **Task.** Phân cụm 50 feature điểm sau imputation/scaling; không dùng labels yes/no. Chọn k=2 theo silhouette trên mẫu train tối đa 2.000 dòng; fit train 8.000; test 2.000. Train silhouette 0,2403; test 0,2434 (`outputs/kmeans/bao_cao_huan_luyen.txt`, `bao_cao_danh_gia.txt`).
- **Ưu điểm.** K-means dùng train-only fit; test không chọn k; silhouette được diễn giải đúng là độ gắn kết/tách biệt trong không gian đặc trưng.
- **Rủi ro/giới hạn.** Silhouette ~0,24 không chứng minh nhóm tự nhiên, hữu ích hay công bằng trong bối cảnh giáo dục. Không có nhãn cụm chuyên gia, stability/seed analysis hoặc kiểm định tác động. Không thể dự đoán môn lựa chọn.
- **Kết luận tạm.** Không đưa làm mô hình chính; chỉ giữ như EDA và chỉ trình bày nếu có câu hỏi nghiên cứu riêng, phân tích độ ổn định và diễn giải độc lập.

## 5. So sánh tổng hợp

| Hướng | Vai trò phù hợp | Bằng chứng hiện có | Điểm yếu quyết định | Xếp hạng tạm |
|---|---|---|---|---|
| CatBoostRegressor | Regression điểm thi mô phỏng | Metric test 3 môn, baseline, grouped split và artifact metadata | Dữ liệu/nhãn mô phỏng, mốc thi thử chưa chốt, chưa CV | **Nên dùng có điều kiện** làm mô hình chính regression |
| SVR | Regression đối chứng | Scale, metric val/test, kiểm tra hội tụ | Train 1.500 dòng so với 30.000 CatBoost; không tuning | **Dùng có điều kiện** sau chuẩn hóa giao thức |
| XGBoost | Dự đoán 9 nhãn multi-label/Top 4 | Test, validation tuning, baseline, audit, predictions | Khác mục tiêu chọn trường; source labels cần xác minh; split theo dòng | **Dùng có điều kiện** nếu chốt task nhãn sở thích |
| CatBoostClassifier | 9 binary labels | Metric theo nhãn và hai dạng tổng hợp Top-4/vector | Không có baseline/CV; semantics label chưa xác nhận | **Tạm hoãn** đến khi nhãn/mục tiêu xác định |
| SVC | 9 binary labels | Scaled pipeline, metric per-label | Không có CV/baseline, calibration không rõ | **Tạm hoãn** tương tự CatBoostClassifier |
| Logistic Regression | Baseline tuyến tính 9 nhãn | 80/20, preprocessing pipeline, metric từng nhãn | Thiếu baseline so sánh và multi-label summary | **Nên thêm** làm baseline nếu task là lựa chọn nhãn |
| K-Means | Khám phá cấu trúc dữ liệu | Train/test silhouette, chọn k trên train | Cụm không có nhãn ý nghĩa và không dự báo | **Không dùng làm mô hình dự báo chính** |

CatBoostRegressor vs SVR là so sánh gần đề tài CatBoost/SVM nhất, nhưng chỉ cùng label/split/feature, chưa cùng lượng dữ liệu huấn luyện. CatBoostClassifier vs SVC có cùng dạng nhãn và gần nhau về F1; XGBoost/Logistic Regression là nhánh khác task, do đó không xếp thứ hạng chung giữa regression và classification.

## 6. Ma trận chấm điểm

Thang 1–5; điểm cao ở mục rủi ro leakage/overfit nghĩa là rủi ro thấp hơn. Điểm chất lượng bằng chứng phản ánh khả năng truy nguyên hiện tại, không phải chất lượng khoa học tổng quát. Tổng có trọng số = tổng (điểm × trọng số), thang 1–5.

| Mô hình | Phù hợp task 25% | Bằng chứng 20% | Tái lập 15% | Rủi ro thấp 15% | Giải thích 10% | Chi phí/mở rộng 10% | Chuyển dữ liệu thật 5% | Tổng / 5 | Phân hạng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CatBoostRegressor | 4 | 4 | 4 | 3 | 3 | 4 | 3 | **3,70** | Dùng có điều kiện |
| SVR | 4 | 3 | 3 | 3 | 2 | 2 | 3 | **3,05** | Dùng có điều kiện |
| CatBoostClassifier | 2 | 3 | 3 | 3 | 3 | 3 | 2 | **2,70** | Tạm hoãn |
| SVC | 2 | 3 | 3 | 3 | 2 | 2 | 2 | **2,55** | Tạm hoãn |
| XGBoost | 2 | 4 | 4 | 3 | 2 | 3 | 2 | **2,95** | Dùng có điều kiện cho nhãn sở thích |
| Logistic Regression | 2 | 3 | 3 | 3 | 4 | 5 | 3 | **3,00** | Baseline có điều kiện |
| K-Means | 1 | 2 | 3 | 3 | 2 | 4 | 2 | **2,25** | Không dùng làm mô hình chính |

**Lý do các điểm chính:** Regression CatBoost/SVR khớp phần dự báo điểm trong kế hoạch, nhưng chỉ gián tiếp phục vụ chọn trường; classifier/XGBoost/Logistic khớp nhãn môn trong repo chứ không khớp trường, nên điểm phù hợp thấp. Báo cáo/mã/metadata đầy đủ hơn ở XGBoost và regression; CatBoost/SVR có grouped split nhưng thiếu CV. Logistic Regression dễ giải thích và rẻ. Khả năng chuyển dữ liệu thật ở mọi mô hình còn thấp vì chưa có dữ liệu thật/kiểm định người dùng. Các điểm là [Suy luận của tôi], cần cập nhật nếu nhóm chốt task khác.

## 7. Kiểm chứng chéo và rủi ro

### 7.1. Nhóm bản ghi mô phỏng và data leakage

- [Có bằng chứng] `CATBoost-SVR/train_models.py` và `train_svr_models.py` kiểm tra giao nhau `nhom_ho_so_goc` giữa train/validation/test; metadata cho 6.000/2.000/2.000 nhóm. Đây là biện pháp phù hợp với 50k biến thể phụ thuộc.
- [Có bằng chứng] Mô tả dữ liệu `CATBoost-SVR/mo_ta_dl.md` ghi bộ 50k mở rộng từ bộ 10k; nhóm hồ sơ gốc có nhiều biến thể nên phải group split. Tệp dữ liệu được xem theo schema/mô tả, không kiểm tra từng giá trị dòng trong lượt rà soát.
- [Suy luận] Group split giảm rò rỉ trực tiếp giữa các bản ghi cùng nguồn, nhưng chưa bảo đảm độc lập nếu nhiều nhóm vẫn dùng cùng quy tắc sinh/seed/đặc trưng nền. Cần đánh giá theo seed hoặc kịch bản sinh độc lập.
- [Có bằng chứng] Trong regression, điểm thi thử cùng môn được đưa vào features; nhãn là điểm thi mô phỏng. Không thể quyết định leakage nếu thiếu định nghĩa thời điểm dự báo và công thức sinh target. Nếu dự báo trước kỳ thi thử, đây là leakage theo thời gian; nếu sau thi thử, có thể là feature hợp lệ nhưng cần ghi rõ.
- [Có bằng chứng] XGBoost preprocessing báo `estimated_feature_columns: []`, không tạo feature từ nhãn/thống kê toàn tập; ID bị loại. Train/validation/test chia theo dòng 70/15/15. Với dữ liệu có biến thể cùng hồ sơ, cần group split.
- [Có bằng chứng] CatBoost/SVC elective chia 60/20/20 theo dòng; code kiểm tra ID trùng nhưng không group-aware. Trên file gốc 10k mỗi ID được ghi duy nhất; không nên mang cách chia này sang dữ liệu mở rộng.
- [Có bằng chứng] Logistic Regression split 80/20 theo dòng và bỏ ID; tiền xử lý nằm trong Pipeline và chỉ fit train. Đây là điểm tốt chống leakage tiền xử lý.

### 7.2. Vòng lặp mô phỏng và ý nghĩa kết quả

- Regression target có hậu tố `_mo_phong`; tài liệu ghi dữ liệu và trường/khu vực giả định. Không thấy bằng chứng nhãn điểm thi được quan sát ngoài đời.
- 50k được mô tả là mở rộng từ 10k theo nhóm; 50k không tương đương 50k học sinh độc lập. Mẫu hiệu dụng và độ bất định cần tính ở cấp nhóm, không coi mỗi biến thể là quan sát độc lập.
- Các nhãn Yes/No 10k là interest/choice theo mô tả code; không xác nhận là đăng ký môn thực tế. Classifier có thể học quy tắc tạo nhãn hoặc tương quan nhân tạo trong dữ liệu.
- Không có thí nghiệm sensitivity theo noise/seed hoặc holdout kịch bản sinh độc lập trong báo cáo mà tôi đọc được. Không thể kết luận độ bền mô hình khi đổi cơ chế sinh.

### 7.3. Chia tập, tuning, metric và bất cân bằng

- Regression: một seed, split sẵn theo nhóm, không thấy CV hoặc nhiều seed; MAE/RMSE/R² phù hợp cho regression. Baseline điểm lớp 9 có ý nghĩa. Cần báo khoảng tin cậy theo nhóm/seed.
- XGBoost: 6 cấu hình, chọn validation Macro F1 rồi Exact Match/Micro F1; test không dùng tuning. Báo cáo có macro/micro/per-label F1 và baseline. Tuy nhiên mỗi dòng test đúng bốn nhãn làm Exact Match bị ảnh hưởng mạnh bởi policy top-four; cần thêm Jaccard/sample-F1/precision@4/recall@4 và rank metrics đã định nghĩa trước.
- CatBoostClassifier/SVC: Accuracy/F1/ROC-AUC per-label là metric hợp lý cho binary classification; thiếu baseline, CV và hiệu chỉnh xác suất. Số nhãn có positive rate gần 0,5 theo metadata, chưa thấy mất cân bằng nặng; không dùng accuracy đơn lẻ.
- Logistic Regression: per-label reports có precision/recall/F1/ROC-AUC; baseline thiếu. `class_weight` tắt được giải thích bằng phân bố gần cân bằng, nhưng cần so majority baseline và kiểm tra calibration.
- K-Means: silhouette phù hợp để mô tả hình học, không để tuyên bố chất lượng dự đoán; cần ổn định cụm theo seed và profile ngoài tập.

### 7.4. Tái lập, chi phí, categorical

- Có requirements riêng cho các nhánh, seed trong script/metadata; thiếu thông tin thống nhất về phiên bản thư viện, OS/phần cứng, thời gian chạy và checksum đối với bộ 50k.
- Tệp mô hình đã lưu không tự đảm bảo tái lập: cần lưu split IDs/groups, checksum dữ liệu, commit/code version, thư viện, cấu hình và lệnh chạy. Các model pickle/joblib phụ thuộc phiên bản thư viện.
- Không thấy thời gian fit/inference đo đạc. SVR dùng 1.500 dòng là dấu hiệu cần kiểm soát chi phí; không thể so chi phí bằng suy đoán.
- Dữ liệu feature hiện là số. CatBoost xử lý numeric trực tiếp; SVC/SVR được scale; Logistic Regression có scale/imputation. Nếu sau này thêm tên trường/khu vực/loại hình là categorical, phải khai báo categorical CatBoost và encode trong Pipeline cho SVM/linear model; không ordinal-encode tuỳ tiện.

### 7.5. Mâu thuẫn và dấu hiệu cần điều tra

1. Tài liệu `Mo_ta.md` đề xuất chọn THPT và nhóm môn, trong khi README và code chính đang làm dự đoán 4 môn trong 9 nhãn. Đây là khác biệt phạm vi, không phải mâu thuẫn về metric.
2. Trong metadata regression, feature `diem_thi_thu_*` nằm trong X; mốc dự báo chưa chốt. Cần kiểm tra thời điểm có sẵn trước khi coi là hợp lệ.
3. CatBoostClassifier/SVC dùng 50 features theo script; tài liệu `mo_ta_dl.md` mô tả file 10k 60 cột gồm 9 nhãn, 50 điểm và ID — khớp cấu trúc. Tuy nhiên nhãn semantic vẫn chưa xác minh.
4. Repo giữ một số model/report artifacts nhưng không có log chạy/commit hash gắn kết; artifact tồn tại không chứng minh nội dung được tạo từ đúng phiên bản code hiện tại.
5. Báo cáo Logistic Regression ghi đường dẫn lưu model `D:\BAIBAO\...` trong console log cũ; README nói script hiện dựa đường dẫn tương đối. Đây có thể là log cũ; không suy ra app hiện tại phụ thuộc đường dẫn đó.

## 8. Đề xuất cải thiện

| Ưu tiên | Vấn đề | Giải pháp | Lợi ích dự kiến | Chi phí |
|---|---|---|---|---|
| **Cao** | Task nghiên cứu chưa thống nhất với nhánh code/data | Chọn một task primary; lập bảng feature/label/mốc thời gian; tách hoàn toàn dự báo điểm, lựa chọn nhãn và xếp hạng trường | Tránh kết luận sai nhiệm vụ, metric và tiêu đề bài báo | Thấp, cần quyết định nhóm |
| **Cao** | Chưa rõ target mô phỏng sinh từ đâu và điểm thi thử có sẵn lúc nào | Viết đặc tả quy tắc sinh, DAG/luồng thời gian, checksum; phân loại từng biến trước/sau mốc dự báo; loại feature sau mốc | Đánh giá đúng leakage và giới hạn kết luận | Trung bình |
| **Cao** | CatBoost/SVR không công bằng về lượng train | Chạy hai mô hình trên cùng danh sách group train; báo learning curves ở cùng n nhóm và cùng search budget; thêm baseline Linear/Ridge và baseline điểm gần nhất | So sánh có ý nghĩa về thuật toán | Trung bình/cao |
| **Cao** | Một split/seed, không có độ bất định | Grouped repeated CV hoặc repeated group holdout theo nhiều seed; bootstrap theo nhóm; báo trung bình, SD/CI | Đo độ ổn định và tránh phụ thuộc một split | Cao với CatBoost, vừa với SVR |
| **Cao** | Nhãn interest bị hiểu thành sự phù hợp/đăng ký | Xác minh nguồn labels; đổi cách diễn đạt giao diện và paper; định nghĩa mục tiêu là hành vi quan sát nếu đó là nguồn nhãn | Tránh diễn giải gây hại hoặc quá mức | Thấp–trung bình |
| **Trung bình** | Chưa kiểm tra độ nhạy cơ chế mô phỏng | Tạo holdout theo seed/quy tắc sinh độc lập; thay mức nhiễu và quan hệ feature-target; không gộp các kịch bản vào một split | Phân biệt học quy tắc sinh với khái quát hoá | Trung bình |
| **Trung bình** | Metric Top-4 chưa phản ánh đầy đủ xếp hạng | Báo precision@4, recall@4, Jaccard/sample-F1, NDCG/MAP nếu thứ hạng có ý nghĩa; định trước metric chính; so popularity/random baselines | Đọc được lỗi xếp hạng và match từng môn | Thấp |
| **Trung bình** | Chưa có calibration/giải thích đáng tin | Calibration curve/Brier score trên validation; SHAP/feature importance chỉ để mô tả và kiểm tra ổn định; không diễn giải nhân quả | Hạn chế dùng xác suất như mức tin cậy tuyệt đối | Trung bình |
| **Trung bình** | Thiếu log tái lập và timing | Lưu checksum, commit, versions, split group IDs, params, hardware, elapsed fit/predict, predictions và command | Tái tạo/report audit được | Thấp |
| **Thấp** | K-Means chưa chứng minh cụm có ích | So stability theo seed/ARI giữa lần chạy, profiling cụm, đánh giá chuyên gia; loại khỏi bài chính nếu không có câu hỏi cụm | Ngăn overclaim cụm | Trung bình |

## 9. Kết hợp dự báo với tiêu chí cá nhân

Điểm dự báo chỉ là một đầu vào, không phải xác suất đỗ hay mức phù hợp. Để gợi ý trường, cần bảng trường có địa bàn, chỉ tiêu/quy chế theo năm, điểm chuẩn theo phương thức, học phí và nguồn/niên hạn dữ liệu; cần khoảng cách hoặc thời gian di chuyển có nguồn; cần ngân sách do người dùng tự nhập. Repo hiện không có dữ liệu trường/nguyện vọng xác thực nên không thể tính ranking trường có căn cứ.

Đề xuất tách thành các bước có thể kiểm toán:

1. **Bộ lọc cứng:** loại lựa chọn vượt ngân sách tối đa, không đúng địa bàn/điều kiện người dùng, hoặc không có dữ liệu còn hiệu lực.
2. **Ước lượng khả thi:** chỉ tính khi có điểm chuẩn/lịch sử tuyển sinh đủ tương thích. Nếu dùng chênh lệch điểm dự báo–ngưỡng tham chiếu, gọi là “khoảng cách tham chiếu”, không gọi xác suất đỗ nếu chưa có mô hình và hiệu chuẩn.
3. **Xếp hạng sở thích/chi phí/khoảng cách:** công khai công thức chuẩn hóa và trọng số; cho phép người dùng thay trọng số. Trình bày từng thành phần và nguồn cập nhật.
4. **Bất định và lựa chọn người dùng:** hiển thị khoảng dự báo hoặc kịch bản; đưa ra danh sách cân nhắc, không tự loại quyền lựa chọn hoặc gán nhãn năng lực.
5. **Đánh giá hệ thống:** kiểm tra tính đúng của dữ liệu trường, độ ổn định thứ hạng và đánh giá với học sinh/chuyên gia theo protocol; nếu chưa có, chỉ báo cáo kiểm thử kỹ thuật.

## 10. Pipeline đề nghị cho bài báo

Nếu nhóm chọn nghiên cứu dự báo điểm:

1. Chốt mốc dự báo và target, đặc biệt xác nhận điểm thi thử có trước mốc.
2. Tách nhóm gốc giữa train/validation/test; nếu tạo thêm data, giữ nguồn/seed sinh thành nhóm độc lập.
3. Tạo baseline: điểm môn gần nhất và Linear/Ridge; sau đó CatBoostRegressor, SVR pipeline scale.
4. Chạy cùng folds, cùng danh sách train groups, cùng ngân sách tìm kiếm, cùng metric MAE/RMSE/R². Tuning chỉ trên train/CV hoặc validation; test một lần sau khi chốt.
5. Lặp theo seeds/folds; báo CI hoặc SD ở cấp group, residual plots, lỗi theo dải năng lực/nhóm dữ liệu.
6. Phân tích feature ablation (điểm thi thử, lịch sử điểm, chuyên cần/tự học) và kiểm tra kịch bản nhiễu.
7. Gọi rõ kết quả là hiệu năng trên nhãn mô phỏng; không đưa ra kết luận tư vấn tuyển sinh thực tế.

Nếu nhóm chọn gợi ý trường, cần một nghiên cứu riêng hoặc pha tiếp theo với dữ liệu trường thực, mục tiêu đánh giá đa tiêu chí đã định nghĩa và đánh giá người dùng. Các classifier môn tự chọn không thay thế pipeline đó.

## 11. Câu hỏi còn mở

1. Bài báo chốt dự báo điểm, xu hướng chọn môn, hay lựa chọn trường THPT? Mục tiêu chính chỉ nên có một.
2. Mốc dự báo là cuối lớp 8, giữa lớp 9, sau thi thử, hay trước kỳ thi tuyển sinh?
3. Ai/thuật toán tạo nhãn điểm thi mô phỏng và nhãn Yes/No? Có mã sinh/config/version độc lập không?
4. Bộ 10k ban đầu có nguồn gốc mô phỏng hay chưa xác minh? Các Yes/No là sở thích, lựa chọn dự kiến hay ghi danh thật?
5. Nhóm có thể tiếp cận dữ liệu tuyển sinh/trường thực, học phí/khoảng cách đã xác thực, học sinh hoặc chuyên gia để đánh giá không?
6. Muốn dùng “nhóm môn định hướng” theo cấu trúc THPT nào, và có danh sách tổ hợp chính thức theo địa bàn/năm không?
7. Có thể lưu ID nhóm gốc/fold và chạy lại hai mô hình regression trên cùng lượng dữ liệu không?
8. Cần paper tập trung so sánh thuật toán hay phát triển hệ thống end-to-end? Ngân sách thời gian/hạn nộp là gì?

## 12. Phụ lục: phạm vi đọc

### Đã đọc hoặc kiểm tra nội dung

- `README.md` gốc; `CATBoost-SVR/AGENTS.md`, `Mo_ta.md`, `mo_ta_dl.md`, `report.md`, `review.md`, `pratice_cat.md`.
- Script CatBoost/SVR/classification và app tại `CATBoost-SVR/`; metadata JSON trong `CATBoost-SVR/models/`.
- `XGBoost/README.md`; pipeline chính (`run_pipeline.py`, `eda_report.py`, `preprocessing.py`, `train.py`, `evaluate.py`, `analyze.py`, `selection.py`) ở mức code liên quan; báo cáo EDA/data/training/evaluation/analysis.
- `KNN-HQTT/Logistic_Regression_TuVanToHop/README.md`; source step scripts và metadata/report đã lưu ở mức có liên quan tới split, pipeline, metric và K-Means.
- Liệt kê toàn bộ file repo và xác nhận có CSV, notebooks, model artifacts, test scripts, ảnh/biểu đồ và dữ liệu sinh.

### Chưa đọc đầy đủ / chưa giải mã

- Không xem từng dòng của các CSV lớn; chỉ dùng tài liệu/schema, report và metadata. Không tính lại thống kê từ raw data.
- Không chạy hoặc trích toàn bộ từng cell của tất cả notebook; `CATBoost-SVR/analys/catboost.ipynb`, `svr.ipynb`, `XGBoost/eda/eda.ipynb` cần rà soát cell-by-cell nếu dùng chúng làm nguồn kết quả chính.
- Không mở/đánh giá nội dung nhị phân của các model `.cbm`, `.joblib`, `.pkl`; sự tồn tại của artifact không xác minh model tương ứng với code hiện tại.
- Chưa đọc mọi test case từng dòng, các file output CSV lớn, ảnh biểu đồ và toàn bộ file nguồn phụ trong các nhánh; không chạy tests/huấn luyện theo phạm vi chỉ đọc.
- Không thấy tài liệu `HUONG_DAN_DU_LIEU_50000.md` trong cây file repo hiện tại dù một tài liệu tham khảo nhắc đến tên này; cần xác nhận file có bị thiếu hay tên đã đổi.
