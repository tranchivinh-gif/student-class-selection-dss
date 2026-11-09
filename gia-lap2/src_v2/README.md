# Hệ thống gợi ý môn học end-to-end

`src_v2` là pipeline độc lập nhận dữ liệu mới, huấn luyện, chọn mô hình và phục vụ web.

## Dữ liệu đầu vào

CSV UTF-8 gồm `ma_hoc_sinh`, 96 cột điểm lớp 6–9 và 9 nhãn môn học. Mỗi dòng phải có đúng bốn nhãn bằng 1.

## Huấn luyện hoàn chỉnh

```powershell
python -m src_v2.train --data data/data-splits/development.csv
```

Pipeline tự động:

1. Kiểm định schema, điểm 0–10, ID và nhãn.
2. Khóa 10% dữ liệu làm holdout.
3. So sánh baseline, XGBoost và CatBoost bằng 5-fold CV.
4. Chọn champion theo NDCG@4.
5. Đánh giá một lần trên holdout.
6. Huấn luyện artifact triển khai trên toàn bộ dữ liệu.

Artifact và báo cáo nằm trong `artifacts/src_v2/`.

## Chạy web

```powershell
python -m streamlit run src_v2/web.py
```

Mặc định giao diện chạy tại `http://localhost:8501`.

Khi nhận bộ dữ liệu mới, dùng một thư mục output mới nếu muốn bảo toàn phiên bản cũ:

```powershell
python -m src_v2.train --data du_lieu_moi.csv --output artifacts/phien_ban_2026
```
