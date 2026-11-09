# `src_real` — Hướng dẫn sử dụng và vận hành

`src_real` là pipeline chính của dự án, dùng để huấn luyện, đánh giá, triển khai
và sử dụng mô hình tư vấn tổ hợp môn từ dữ liệu học sinh thật. Tài liệu này bao
gồm toàn bộ hướng dẫn cập nhật dữ liệu, huấn luyện lại và lấy model làm ứng dụng
mà không cần sửa mã Python.

## 1. Quy ước quan trọng

Có hai cấp artifact:

1. **Ba model của từng lần chạy**: `baseline.joblib`, `xgboost.joblib`,
   `catboost.joblib`; `champion.joblib` là bản sao thuật toán được chọn.
2. **Ba model ứng dụng ổn định** nằm tại `artifacts/src_real/`. Web cho phép
   chuyển đổi giữa chúng; `champion.joblib` là model mặc định.

Huấn luyện thông thường chỉ tạo model trong thư mục `runs`; nó không làm thay
đổi model web đang dùng. Chỉ lệnh `src_real.deploy` hoặc cờ `--promote` mới thay
model ứng dụng.

## 2. Chuẩn bị môi trường

Chạy các lệnh từ thư mục gốc của dự án:

```powershell
python -m pip install -r requirements.txt
python -m src_real.train --help
python -m src_real.deploy --help
python -m src_real.predict --help
```

## 3. Chuẩn bị dữ liệu mới

File phải là CSV UTF-8 và có:

- `ma_hoc_sinh`: không trống, không trùng;
- 96 cột điểm lớp 6–9 được định nghĩa trong `src_real/config.py`;
- 9 cột nhãn môn học;
- điểm là số từ 0 đến 10 và không thiếu;
- nhãn chỉ nhận 0 hoặc 1;
- mỗi dòng thuộc một trong bốn tổ hợp hiện được hỗ trợ.

| Mã | Các môn |
|---|---|
| `000111100` | Địa lí + GDKT&PL + Tin học + Công nghệ |
| `101001100` | Vật lí + Sinh học + Tin học + Công nghệ |
| `110001100` | Vật lí + Hóa học + Tin học + Công nghệ |
| `111001000` | Vật lí + Hóa học + Sinh học + Tin học |

Nếu trường có tổ hợp thứ năm, đây là thay đổi bài toán chứ không chỉ là dữ liệu
mới. Khi đó phải cập nhật `ALLOWED_COMBINATIONS` trong `config.py` và kiểm tra
lại giao diện, metric và số mẫu của từng lớp.

## 4. Huấn luyện một phiên bản mới

```powershell
python -m src_real.train `
  --data "data/data-clean/du_lieu_that_2027.csv" `
  --run-name "du_lieu_that_2027"
```

Không dùng lại tên của một lần chạy cũ. Nếu không truyền `--run-name`, chương
trình tự tạo tên theo thời gian, ví dụ `run_20271108_153000`.

Các tham số tùy chọn:

```text
--folds 5          số fold CV
--repeats 5        số lần lặp CV
--test-size 0.2    tỷ lệ test độc lập
--feature-set all  dùng đủ 96 cột (mặc định)
--feature-set annual  chỉ dùng 32 cột điểm cả năm
--output PATH      tự chỉ định toàn bộ thư mục kết quả
--promote          huấn luyện xong và áp dụng ngay
```

Chạy nhanh để kiểm tra kỹ thuật (không dùng để báo cáo chính thức):

```powershell
python -m src_real.train `
  --data "data/data-clean/du_lieu_that_2027.csv" `
  --run-name "kiem_tra_nhanh_2027" `
  --folds 3 --repeats 1
```

Huấn luyện chỉ bằng điểm cả năm:

```powershell
python -m src_real.train `
  --data "data/data-clean/du_lieu_that_2027.csv" `
  --run-name "diem_ca_nam_2027" `
  --feature-set annual `
  --promote
```

Artifact tự lưu danh sách đặc trưng đã dùng. Web và `src_real.predict` sẽ tự
chọn đúng 32 cột, không cần sửa mã sau huấn luyện.

## 5. Đọc kết quả

Mở `artifacts/src_real/runs/<tên-lần-chạy>/`:

| File | Ý nghĩa |
|---|---|
| `baseline.joblib`, `xgboost.joblib`, `catboost.joblib` | Ba model được huấn luyện |
| `champion.joblib` | Model đứng đầu theo repeated CV |
| `model_comparison.csv` | Xếp hạng và độ lệch chuẩn của ba model |
| `selection_report.json` | Quy tắc và lý do chọn champion |
| `MODEL_LOCATION.txt` | Đường dẫn model và trạng thái triển khai |
| `metadata.json` | Nguồn dữ liệu, thời gian, cấu hình và metric |
| `cv_summary.csv` | Trung bình, độ lệch chuẩn, min/max của CV |
| `repeated_cv_metrics.csv` | Metric từng fold |
| `repeated_oof_predictions.csv` | Dự đoán validation qua các fold |
| `independent_test_metrics.csv` | Kết quả cuối trên test độc lập |
| `independent_test_predictions.csv` | Đúng/sai của từng mẫu test |
| `development.csv` | Dữ liệu dùng phát triển và huấn luyện artifact |
| `locked_independent_test.csv` | Test khóa của riêng lần chạy |

Nên xem ít nhất `independent_test_metrics.csv`, `cv_summary.csv` và phân bố lớp
trước khi triển khai. Với tập test nhỏ, không chọn model chỉ vì chênh lệch một
vài điểm phần trăm.

## 6. Đưa model đã duyệt vào ứng dụng

```powershell
python -m src_real.deploy `
  --run "artifacts/src_real/runs/du_lieu_that_2027"
```

Lệnh này kiểm tra artifact, sao lưu model cũ vào `artifacts/src_real/backups/`,
chép cả ba model vào `artifacts/src_real/`, đặt model được chọn thành
`champion.joblib`, ghi `deployment_manifest.json` và giúp web cho phép chọn
Baseline, XGBoost hoặc CatBoost.

Có thể huấn luyện và triển khai ngay bằng `--promote`, nhưng cách an toàn hơn là
huấn luyện, xem metric, rồi mới chạy `src_real.deploy`.

## 7. Chạy ứng dụng

```powershell
python -m streamlit run src_real/web.py
```

Mở `http://localhost:8501`. Chọn thuật toán trong sidebar. XGBoost và CatBoost
có phần giải thích SHAP; baseline giải thích bằng tần suất tổ hợp.

## 8. Dự đoán bằng dòng lệnh

Dùng model ứng dụng:

```powershell
python -m src_real.predict `
  --input "data_du_doan/hoc_sinh_moi.csv" `
  --output "ket_qua/du_doan_2027.csv" `
  --top-k 3
```

Dùng model của một run chưa deploy:

```powershell
python -m src_real.predict `
  --input "data_du_doan/hoc_sinh_moi.csv" `
  --model "artifacts/src_real/runs/du_lieu_that_2027/champion.joblib" `
  --output "ket_qua/thu_model_2027.csv"
```

## 9. Lấy model sang ứng dụng khác

Ứng dụng Python cần các file sau:

```text
artifacts/src_real/champion.joblib
src_real/model.py
src_real/data.py
src_real/config.py
```

Môi trường đích cần Python và các phiên bản tương thích của CatBoost, pandas,
NumPy, scikit-learn, joblib. Joblib không phải định dạng độc lập ngôn ngữ.

```python
from src_real.data import load_features
from src_real.model import RealCombinationModel

model = RealCombinationModel.load("artifacts/src_real/champion.joblib")
students = load_features("hoc_sinh_moi.csv")
result = model.recommend(students, top_k=3)
```

## 10. Phản hồi và học bổ sung

```powershell
python -m src_real.feedback prepare --input "hoc_sinh_moi.csv"
python -m src_real.feedback learn `
  --feedback "artifacts/src_real/feedback_pending.csv"
```

`learn` tạo candidate và không tự thay model. Chỉ promote sau khi đã xem báo
cáo. Tuyệt đối không đưa học sinh thuộc test khóa vào phản hồi huấn luyện.

## 11. Lỗi thường gặp

- **Thiếu cột:** đối chiếu với `FEATURES` trong `config.py`, không sửa mã theo
  từng file dữ liệu.
- **Run đã tồn tại:** đặt `--run-name` mới để bảo toàn kết quả cũ.
- **Tổ hợp không được hỗ trợ:** xác minh dữ liệu; không âm thầm đổi nhãn.
- **Web vẫn dùng model cũ:** chạy `src_real.deploy`, khởi động lại Streamlit hoặc
  xóa cache resource.
- **Máy khác không mở được model:** cài dependencies tương thích và mang theo
  package `src_real`, không chỉ chép riêng joblib.
