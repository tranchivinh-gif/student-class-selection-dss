# CatBoost cho 18 tổ hợp môn lớp 10: báo cáo bàn giao

## 1. Bài toán, dữ liệu và kết luận hiện tại

Mỗi hồ sơ có đúng 4 trong 9 môn tự chọn. Trong bộ dữ liệu có 18 tổ hợp khác nhau; mục tiêu MultiClass là ước lượng xác suất cho 18 tổ hợp ấy rồi xếp hạng. Bộ dữ liệu **mô phỏng hoặc có nguồn nhãn chưa xác minh**, gồm 10.000 học sinh: 8.500 hồ sơ phát triển (dev) và 1.500 hồ sơ test đã được đánh giá một lần. F4 gồm 70 đặc trưng số: điểm cả năm lớp 6–9 của tám môn nền, thống kê và xu hướng điểm, chỉ số tương quan nhóm môn và thứ hạng điểm. Mã học sinh chỉ để nối/chia tập, không đưa vào mô hình. A1 là CatBoost MultiClass F4, 331 cây, không đặt trọng số lớp; A4 trộn xác suất A1 và Logistic Regression theo tỷ lệ 50/50. Nguồn: `../../experiments/catboost/combo_selection/metadata.json`, `../../experiments/catboost/exact_improvement/run_config.json`.

**Hai mục tiêu có hai kết luận.** A1 có tín hiệu xếp hạng Top-3 trong bản thử nghiệm: trên test, tổ hợp thật nằm trong ba phương án đầu ở 882/1.500 hồ sơ (58,80%), so 858/1.500 (57,20%) của baseline tần suất tổ hợp. Nhưng dự đoán đúng *nguyên bộ* chỉ 386/1.500 (25,73%), thấp hơn 392/1.500 (26,13%) của baseline chọn tổ hợp phổ biến nhất; CI95 của hiệu Exact chứa 0. Quyết định chính thức vẫn là **CHƯA ĐẠT**. A4 chỉ là ứng viên thăm dò trên dev, chưa được chấm trên tập độc lập chưa dùng. Nguồn: `../../experiments/catboost/combo_selection/{final_metrics.csv,test_ci.csv,recall_by_combo.csv,decision.md}`, `../../experiments/catboost/final_review/paired_summary.json`.

| Cổng theo quyết định cũ | Kết quả | Bằng chứng trên test hoặc dev đã lưu |
|---|---|---|
| G1 Exact | FAIL | Hiệu −0,0040; CI95 [−0,0173; 0,0100] |
| G2 Top-3 | PASS | Hiệu +0,0160; CI95 [0,0040; 0,0280] |
| G3 LogLoss18 | PASS | Hiệu −0,0237; CI95 [−0,0359; −0,0120] |
| G4 Ổn định nội bộ | PASS | Exact hơn baseline ở 15/15 fold; SD 0,0033 |
| G5 Không sụp lớp | FAIL | Chỉ 3/18 tổ hợp được dự báo; recall macro 0,0595 < 0,0833 |
| G6 Precision@4 | PASS | Hiệu +0,0273; CI95 [0,0172; 0,0372] so baseline tổ hợp phổ biến nhất |

Nguồn: `../../experiments/catboost/combo_selection/decision.md`. Các cổng không được thay đổi sau khi xem test.

## 2. Thông số điều chỉnh được

Các phép quét thô trong `sensitivity.csv` dùng **MultiLogloss chín nhãn trên mẫu 2.000 dòng**, nên chỉ cho gợi ý về độ nhạy, không chứng minh cùng tác động ở MultiClass A1. Đối với A1, cấu hình chốt và thử trọng số lớp trên dev lấy từ `combo_selection/metadata.json`, `tune_choice.csv` và `weight_tuning/summary.csv`.

| Tham số | Ý nghĩa | A1 | Mức đã thử và điều quan sát được | Khi cân nhắc chỉnh |
|---|---|---:|---|---|
| `iterations` | Số cây tối đa hoặc số cây cố định | **331 cây cố định** | Quét MultiLogloss trần 200/500/1.000, dừng sớm ở 47/103/109 cây; A1 dùng 331 cây chốt từ CV, không phải trần 500 | Chỉ xét lại trên dev khi đổi dữ liệu; giữ riêng trần và số cây thực tế |
| `depth` | Độ sâu, mức tương tác | 2 | MultiLogloss thử 4/6/8/10: P@4 train tăng 0,7517→0,9694, validation không đơn điệu 0,6919/0,6925/0,6881/0,6850; thời gian 3,0→84,2 giây | Khi chênh train–validation lớn, cân nhắc cây nông hơn |
| `learning_rate` | Bước cập nhật mỗi cây | 0,0586 | MultiLogloss thử 0,01/0,03/0,1/0,3; mức 0,3 dừng sau 6 cây và P@4 validation 0,6738 | Chỉnh cùng số cây, theo LogLoss dev |
| `l2_leaf_reg` | Phạt độ lớn giá trị lá | 15,5268 | MultiLogloss thử 1/3/10/30; P@4 train giảm khi phạt tăng, validation dao động 0,6800–0,6913 | Khi có dấu hiệu quá khớp, kiểm tra trên fold dev |
| `border_count` | Số ngưỡng rời rạc hóa điểm số | 254 | MultiLogloss thử 32/128/254; thời gian 1,9→5,6 giây, P@4 validation gần nhau | Khi cần giảm chi phí tính toán |
| `random_strength` | Nhiễu khi chọn phép chia | 4,4869 | MultiLogloss thử 0,5/1/3; không thấy cải thiện validation nhất quán | Chỉ chỉnh theo CV, không theo một fold |
| `rsm` | Tỷ lệ đặc trưng xét mỗi phép chia | 0,9808 | Có trong cấu hình A1; chưa có phép quét tách riêng trong artifact | Khi cần kiểm tra độ phụ thuộc đặc trưng |
| `bagging_temperature` | Mức ngẫu nhiên lấy mẫu | 0,8485 | Có trong cấu hình A1; chưa có phép quét tách riêng trong artifact | Khi kiểm tra quá khớp trên dev |
| `auto_class_weights` | Tăng trọng số lớp hiếm | `None` | MultiClass thử None/SqrtBalanced/Balanced: Exact dev 0,2740/0,2691/0,0216; Balanced dự báo nhiều tổ hợp hơn nhưng sai nhiều | Không bật chỉ để tăng số tổ hợp được chọn |
| `random_seed` và fold seed | Điều khiển phần ngẫu nhiên | 42 cho cấu hình test cũ; dev 42/101/202 | Đã đánh giá 5 fold × 3 seed trên cùng bộ mô phỏng | Đổi seed để đo dao động nội bộ, không xem như dữ liệu độc lập |
| `thread_count` | Số luồng chạy | 4 | Cố định; chưa có phép quét | Chỉnh theo máy khi chạy lại, ghi cùng thời gian |

Nguồn: `../../experiments/catboost/sensitivity.csv`, `../../experiments/catboost/combo_selection/{metadata.json,tune_choice.csv}`, `../../experiments/catboost/weight_tuning/summary.csv`, `../../experiments/catboost/exact_improvement/run_config.json`. Tuning MultiClass ghi 15 trial hợp lệ, 10 bị cắt, 0 lỗi; kết quả cải thiện LogLoss nhỏ nên giữ cấu hình đã chốt (`tune_choice.csv`).

## 3. So sánh cùng giao thức và cách đọc chỉ số

Trên dev, mỗi mô hình được đánh giá ngoài mẫu theo 5 fold × 3 seed; baseline tổ hợp được tính từ *train của từng fold*. Bảng là trung bình 15 fold; SD chỉ đo dao động giữa các lượt chia cùng bộ mô phỏng.

| Cách chọn trên dev | Exact ± SD | Top-3 | LogLoss18 | Precision@4 | F1 macro18 | Recall macro18 | Fold thắng baseline Exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tổ hợp phổ biến nhất / xếp theo tần suất train | 0,2619 | 0,5721 | 2,2362 | 0,6194 | — | 0,0556 | — |
| A1 CatBoost F4 | 0,2740 ± 0,0033 | 0,5791 | 2,2039 | 0,6588 | 0,0424 | 0,0649 | 15/15 |
| A2 Logistic Regression | 0,2761 ± 0,0087 | 0,5774 | 2,2022 | 0,6612 | 0,0470 | 0,0672 | 14/15 |
| A4 trộn 50/50 | 0,2776 ± 0,0053 | 0,5807 | 2,1999 | 0,6611 | 0,0456 | 0,0667 | 15/15 |

Nguồn: `../../experiments/catboost/exact_improvement/summary.csv`, `../../experiments/catboost/weight_tuning/summary.csv`. A2 chọn `C` trong {0,01; 0,1; 1} bằng CV ba fold ở train của mỗi fold ngoài; A4 trộn xác suất sau khi căn đúng thứ tự 18 lớp (`exact_improvement/run.py`).

| Cách chọn trên **test cũ** | Exact | Top-3 | LogLoss18 | Precision@4 |
|---|---:|---:|---:|---:|
| Tổ hợp phổ biến nhất / xếp theo tần suất train | 0,2613 | 0,5720 | 2,2378 | 0,6193 |
| A1 CatBoost F4 | 0,2573 | 0,5880 | 2,2141 | 0,6467 |
| Bốn môn phổ biến riêng lẻ | 0,0087 | không áp dụng | không cùng thang đo | 0,6617 |

Nguồn: `../../experiments/catboost/combo_selection/{final_metrics.csv,test_ci.csv}`, `../../experiments/catboost/metrics.csv`. A2/A4 **không có kết quả test**. Baseline tổ hợp phổ biến nhất chọn một bộ bốn môn cố định để tính Exact; cùng tần suất train tạo bảng xác suất để xếp Top-3 và tính LogLoss18. Baseline bốn môn phổ biến ghép bốn nhãn riêng lẻ; LogLoss của thí nghiệm chín nhãn không so trực tiếp với LogLoss18.

- **Exact-match** đo đúng *toàn bộ tổ hợp* bốn môn. Ví dụ A1 đúng 386/1.500 test, baseline 392/1.500, hiệu −0,40 điểm phần trăm; CI95 [−1,73; +1,00] điểm phần trăm.
- **Top-3** đo tổ hợp thật có nằm trong ba tổ hợp xác suất cao nhất. A1 đúng 882/1.500, baseline 858/1.500, hiệu +1,60 điểm phần trăm; CI95 [+0,40; +2,80]. Đây vẫn là chỉ số **tổ hợp**, không phải ba môn.
- **Precision@4** đo số môn đúng trong bốn môn được chọn, chia 4 rồi lấy trung bình. A1 có 0,6467 so 0,6193 của baseline tổ hợp; nhưng thấp hơn 0,6617 của baseline bốn môn phổ biến. Đây là chỉ số **từng môn trong bộ bốn**.
- **LogLoss18** phạt xác suất thấp gán cho tổ hợp thật; nhỏ hơn là tốt hơn. A1 2,2141 so 2,2378 trên test, hiệu −0,0237, CI95 [−0,0359; −0,0120]. Đây là chỉ số xác suất **nguyên tổ hợp**.
- **F1 macro18** lấy F1 trung bình đều trên 18 tổ hợp; A1 dev 0,0424. **Recall macro18** lấy trung bình tỷ lệ đúng của từng tổ hợp; A1 test 0,0595, tức khoảng 5,95% theo trung bình lớp, dù Exact tổng 25,73%. Cả hai đo **18 lớp tổ hợp** và bộc lộ nhóm hiếm bị bỏ sót.

## 4. Độ tin cậy, dữ liệu và độ ổn định

CI95 **bootstrap theo cặp** lấy lại học sinh, giữ hai dự đoán trên cùng học sinh đi cùng nhau. Với A4−A1, trung bình hiệu Exact qua ba seed của mỗi học sinh là +0,3569 điểm phần trăm; CI95 trực tiếp [+0,0391; +0,6589] điểm phần trăm trên 8.500 học sinh dev. A4 đúng/A1 sai 502 lượt học sinh–seed, chiều ngược lại 411 lượt. Hiệu dương ở seed 42/101/202 lần lượt +0,4588/+0,2941/+0,3176 điểm phần trăm. **A4 được chọn sau khi xem dev**, nên CI này chỉ là thăm dò và không xác nhận cải thiện trên dữ liệu độc lập. Nguồn: `../../experiments/catboost/final_review/{paired_summary.json,by_seed.csv}`.

Hiệu chuẩn xác suất được nhóm theo độ tin cậy cao nhất của dự đoán OOF trung bình ba seed. Với A1, nhóm thấp nhất có xác suất cao nhất trung bình 0,2035 và Exact 0,2047; nhóm cao nhất là 0,3776 và 0,3788, mỗi nhóm 1.700 học sinh. Đây là mức khớp mô tả theo nhóm trên dev, **không phải** mức độ phù hợp cá nhân với tổ hợp. Nguồn: `../../experiments/catboost/exact_improvement/calibration.csv`.

Dữ liệu lệch mạnh: test chỉ có 3 ca Âm nhạc và 25 ca Mĩ thuật; A1 không dự báo hai tổ hợp tương ứng, và trên test chỉ dự báo 3/18 tổ hợp. Trên dev, A1 dồn trung bình 75,51% dự đoán vào tổ hợp phổ biến nhất; Balanced mở rộng số tổ hợp được dự báo nhưng Exact rơi còn 0,0216. Các hiện tượng này liên quan **mất cân bằng lớp**. Chúng không định lượng độ nhạy với sai số điểm, điểm thiếu hoặc nhãn bị ghi sai. Với nhiễu đầu vào/nhãn hiện **chưa có thí nghiệm định lượng**; có thể kiểm tra trên dev bằng cách thêm nhiễu có kiểm soát và đo lại Exact, Top-3 trên cùng fold, không dùng test cũ để chọn mức nhiễu. Dữ liệu mô phỏng có thể phản ánh quy tắc sinh dữ liệu. Nguồn: `../../experiments/catboost/combo_selection/recall_by_combo.csv`, `../../experiments/catboost/weight_tuning/summary.csv`, `../../experiments/catboost/exact_improvement/summary.csv`.

SD Exact của A1/A2/A4 trên 15 fold là 0,0033/0,0087/0,0053; số fold hơn baseline là 15/14/15. Các seed của A1 và A4 đều có hiệu dương trên dev, nhưng đây là nhiều cách chia **cùng bộ mô phỏng**. Chưa biết mô hình có ổn định trên dữ liệu độc lập hay không. Nguồn: `../../experiments/catboost/exact_improvement/summary.csv`, `../../experiments/catboost/final_review/by_seed.csv`.

## 5. Kết luận và tái lập

Giữ **A1** làm mô hình tham chiếu; dùng Top-3 để xếp hạng phương án trong bản thử nghiệm, ghi rõ đây là gợi ý có xếp hạng. Exact vẫn **CHƯA ĐẠT** theo cổng đã lưu. A4 là ứng viên cho lần kiểm tra độc lập, chưa phải mô hình thay thế. Để kiểm tra vòng tiếp theo cần danh sách tổ hợp trường mở theo năm và sở thích/nguyện vọng ghi trước lựa chọn, có khóa nối học sinh–trường–năm; không dùng nhãn đăng ký cuối làm đầu vào. Chưa có nguồn đủ điều kiện được xác minh trong repo. Nguồn: `../../experiments/catboost/final_review/{report.md,data_readiness.csv}`.

Notebook đi kèm đọc artifact hiện có, không huấn luyện lại hoặc chấm lại test. Chạy từ gốc repo:

```powershell
$env:JUPYTER_ALLOW_INSECURE_WRITES='1'  # Cần trong môi trường Windows hiện tại để tạo kernel
python -m nbconvert --to notebook --execute --inplace deliverables/catboost_to_hop/catboost_multiclass_18_to_hop.ipynb
```

Nguồn tái lập chi tiết: `../../experiments/catboost/combo_selection/`, `../../experiments/catboost/weight_tuning/`, `../../experiments/catboost/exact_improvement/`, `../../experiments/catboost/final_review/`.

Notebook đã lưu output để xem ngay. Tại thời điểm bàn giao, các artifact trong `experiments/catboost/` đang **chưa được Git theo dõi**; commit chỉ gồm hai sản phẩm này sẽ không đủ để chạy lại notebook trên một bản clone mới nếu chưa cung cấp các artifact nguồn.
