# Tiền xử lý dữ liệu EDA

Chạy từ thư mục gốc repository:

```powershell
python .\XGBoost\eda\preprocess.py
```

Script đọc `XGBoost/data/hoc_sinh.csv`, dùng bước kiểm tra và chuẩn hóa hiện có trong `XGBoost/preprocessing.py`, rồi chỉ giữ học sinh có đúng 4 môn được chọn. Các dòng có ít hơn hoặc nhiều hơn 4 môn đều bị loại khỏi tệp kết quả; dữ liệu nguồn không bị thay đổi.

Mặc định, dữ liệu sau lọc được ghi vào `XGBoost/data/hoc_sinh_4_mon.csv`. Mọi CSV đều nằm trong `XGBoost/data/`; thư mục `eda/` chỉ chứa mã và notebook. Có thể chỉ định đường dẫn khác:

```powershell
python .\XGBoost\eda\preprocess.py --input .\XGBoost\data\hoc_sinh.csv --output .\XGBoost\data\hoc_sinh_4_mon.csv
```

Bước lọc này là script riêng, không thay đổi luồng huấn luyện hiện tại trong `XGBoost/`.
