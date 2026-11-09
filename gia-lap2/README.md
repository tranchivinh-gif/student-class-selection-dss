# Hệ thống tư vấn tổ hợp môn

Nhánh chính của dự án là [`src_real`](src_real/README.md), dùng dữ liệu học sinh
thật để huấn luyện, đánh giá, dự đoán và chạy giao diện Streamlit. README trong
thư mục này cũng là hướng dẫn đầy đủ để thay dữ liệu, huấn luyện phiên bản mới,
đọc kết quả và lấy model làm ứng dụng.

## Bắt đầu nhanh

```powershell
python -m pip install -r requirements.txt

python -m src_real.train `
  --data "data/data-clean/du_lieu_that.csv" `
  --run-name "du_lieu_that_2026"

python -m src_real.deploy `
  --run "artifacts/src_real/runs/du_lieu_that_2026"

python -m streamlit run src_real/web.py
```

Model mà ứng dụng sử dụng nằm tại `artifacts/src_real/champion.joblib`. Mỗi lần
huấn luyện được lưu riêng trong `artifacts/src_real/runs/` để không ghi đè kết
quả cũ.

Các thư mục `src_v2`, `src_v3`, `src`, `experiments` được giữ làm lịch sử và
nghiên cứu; không phải luồng triển khai chính.
