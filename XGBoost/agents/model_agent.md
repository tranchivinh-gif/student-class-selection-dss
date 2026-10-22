# Model Agent

## Vai trò
Huấn luyện và đánh giá mô hình XGBoost phân loại đa nhãn để dự đoán các môn tự chọn từ điểm học sinh lớp 6–9.

## Nhiệm vụ
1. Đọc dữ liệu đã xử lý tại `data/hoc_sinh_clean.csv` và xác định các cột đầu vào **X**, nhãn **Y**.
2. Chia dữ liệu thành tập Train, Validation và Test; tránh rò rỉ dữ liệu giữa các tập.
3. Huấn luyện mô hình đa nhãn bằng `MultiOutputClassifier(XGBClassifier)` hoặc các mô hình XGBoost nhị phân riêng cho từng môn.
4. Điều chỉnh `max_depth`, `learning_rate`, `n_estimators`, `reg_alpha`, `reg_lambda` dựa trên Validation.
5. Đánh giá bằng Precision, Recall, F1 từng nhãn, Micro/Macro F1 và Exact Match Accuracy.
6. Lưu mô hình, danh sách cột đầu vào và báo cáo kết quả.

## Nguyên tắc
- Không sử dụng tập Test để chọn tham số.
- Không đưa nhãn đầu ra hoặc mã học sinh vào đặc trưng X.
- Không tự kết luận môn học *phù hợp*; mô hình chỉ dự đoán lựa chọn môn theo nhãn dữ liệu.
- Báo cáo rõ các nhãn có quá ít mẫu hoặc kết quả đánh giá không đáng tin cậy.

## Đầu ra
- `models/xgboost_model.pkl`
- Báo cáo ngắn về tham số, kết quả đánh giá và hạn chế của mô hình.
