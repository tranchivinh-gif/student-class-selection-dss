# Dự đoán trực tiếp tổ hợp bốn môn

`src_v3` không dự đoán từng môn độc lập. Mỗi tổ hợp bốn môn là một lớp multiclass hoàn chỉnh.

```powershell
python -m src_v3.train --data data/data-splits/development.csv
```

Pipeline so sánh:

- Baseline: tổ hợp phổ biến nhất.
- XGBoost multiclass.
- CatBoost multiclass.

Mô hình được chọn theo `Exact_Match_Accuracy`, tức tỷ lệ dự đoán đúng toàn bộ 4/4 môn.

Dự đoán dữ liệu mới:

```powershell
python -m src_v3.predict --input du_lieu_diem.csv --alternatives 3
```

Kết quả trả về tổ hợp tốt nhất và hai tổ hợp thay thế, mỗi tổ hợp luôn có đúng bốn môn.
