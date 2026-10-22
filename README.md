# Hệ thống dự đoán môn tự chọn của học sinh

Dự án dùng điểm lớp 6–9 để dự đoán **4 môn tự chọn** trong 9 môn. Mô hình học các lựa chọn đã quan sát trong dữ liệu; kết quả không phải tư vấn môn học phù hợp với từng học sinh.

Mã thực thi nằm trong [`XGBoost/`](XGBoost/). Các lệnh dưới đây chạy từ **thư mục gốc repository**.

## Cài đặt

Yêu cầu Python 3.11 trở lên.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\XGBoost\requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, gọi `.\.venv\Scripts\python.exe` thay cho `python`.

## Dữ liệu đầu vào

Đường dẫn mặc định là [`XGBoost/data/hoc_sinh.csv`](XGBoost/data/hoc_sinh.csv). CSV cần có `ma_hoc_sinh` duy nhất, các cột điểm theo mẫu `*_lop6` đến `*_lop9` (số từ 0 đến 10), và các cột nhãn môn tự chọn (`Yes/No` hoặc `1/0`). Mọi cột ngoài mã và điểm đều được coi là nhãn: không thêm họ tên, lớp hoặc ghi chú vào CSV huấn luyện. Không để ô trống, dòng trùng hoặc mã trùng. Chương trình dừng và báo lỗi để kiểm tra thủ công; nó không tự xóa dòng hay điền điểm.

## Chạy toàn bộ luồng

```powershell
python .\XGBoost\run_pipeline.py
```

Đổi CSV nguồn mà không sửa mã:

```powershell
python .\XGBoost\run_pipeline.py --input .\du_lieu_moi.csv
```

Khi chạy từ thư mục gốc repository, dữ liệu trong `XGBoost/data/` có thể chỉ định bằng `--input .\XGBoost\data\hoc_sinh.csv` hoặc dạng ngắn `--input .\data\hoc_sinh.csv`. Dạng ngắn được `run_pipeline.py` tự tìm trong `XGBoost/data/` nếu không có thư mục `data/` ở vị trí đang chạy lệnh.

Luồng chạy **EDA → xử lý dữ liệu → huấn luyện → đánh giá → trình bày**. Có thể chạy từng bước:

| Bước | Lệnh | Tệp đầu ra dưới `XGBoost/` |
|---|---|---|
| 1. EDA | `python .\XGBoost\eda_report.py` | `reports/eda_report.md` |
| 2. Xử lý | `python .\XGBoost\preprocessing.py` | `data/hoc_sinh_clean.csv`, `reports/data_report.md` |
| 3. Huấn luyện | `python .\XGBoost\train.py` | `models/xgboost_model.pkl`, `reports/training_report.json`, `reports/test_predictions.csv` |
| 4. Đánh giá | `python .\XGBoost\evaluate.py` | `reports/evaluation.json`, `reports/evaluation_report.md` |
| 5. Trình bày | `python .\XGBoost\analyze.py` | `reports/analysis_report.md`, `reports/feature_importance.csv` và hai biểu đồ PNG |

Nếu chạy từng bước với CSV mới, truyền cùng `--input` cho EDA và xử lý. `train.py` mặc định đọc CSV sạch của bước 2; `--input` ở đây chỉ dùng cho **CSV đã làm sạch**. Khi nguồn thay đổi, chạy lại từ bước 1 để các đầu ra đồng bộ. Các tệp đầu ra cùng tên sẽ được ghi đè, nên sao lưu trước nếu cần giữ lần chạy cũ.

### Mỗi bước làm gì?

1. **EDA** chỉ đọc nguồn, thống kê số học sinh, ô trống, dòng trùng và số môn đã chọn. Notebook [`XGBoost/eda/eda.ipynb`](XGBoost/eda/eda.ipynb) phục vụ khám phá tương tác và không cần để chạy pipeline.
2. **Xử lý** kiểm tra điểm, nhãn, dòng trùng và mã trùng; đổi nhãn sang `0/1`. CSV nguồn không bị sửa.
3. **Huấn luyện** chia Train/Validation/Test 70/15/15 với seed 42. Sáu cấu hình XGBoost được chọn theo Macro F1 trên Validation, phá hòa bằng Exact Match rồi Micro F1. Mô hình cuối được huấn luyện trên Train + Validation và ghi dự đoán Test.
4. **Đánh giá** dùng Test đã lưu, so với baseline 4 môn phổ biến nhất trong Train. Test không dùng để chọn tham số.
5. **Trình bày** tạo bảng lỗi theo môn, bảng importance và biểu đồ từ mô hình cùng Test.

## Dự đoán và giao diện

CSV dự đoán cần đủ các cột điểm ghi trong mô hình; không cần nhãn. Có thể có `ma_hoc_sinh` để giữ mã trong kết quả.

```powershell
python .\XGBoost\predict.py --input .\du_lieu_can_du_doan.csv --output .\du_doan.csv
python .\XGBoost\app.py
```

Giao diện ở <http://127.0.0.1:5000>. Thay điểm mẫu bằng điểm thực tế. Mỗi dòng dự đoán có xác suất từng môn và đúng 4 cột `predicted::<môn>` bằng `1`.

## Kiểm tra và mở rộng

```powershell
cd .\XGBoost
python -m unittest discover -s tests -v
```

- Quy tắc kiểm tra dữ liệu: `preprocessing.py`.
- Cấu hình mô hình và chia tập: `train.py`.
- Số môn và quy tắc chọn: `selection.py` (cập nhật cả giao diện và tài liệu khi đổi số môn).
- Chỉ số và báo cáo: `evaluate.py`; biểu đồ: `analyze.py`.
- `estimate_missing_features.py` và `eda/preprocess.py` là công cụ thử nghiệm riêng, không thuộc luồng mặc định. Công cụ sau lọc những học sinh chọn đúng 4 môn vào `XGBoost/data/hoc_sinh_4_mon.csv`.

**Giới hạn:** Khi học sinh thực tế chọn khác 4 môn, dự đoán cố định 4 môn không thể khớp hoàn toàn. Importance không chứng minh quan hệ nhân quả; xác suất không đo độ phù hợp học tập. Cần kiểm tra trên dữ liệu mới trước khi ứng dụng thực tế.
