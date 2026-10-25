# Thực nghiệm kiểm chứng

## Môi trường và giới hạn

Thí nghiệm chỉ dùng dữ liệu mô phỏng có trong repo, không tải dữ liệu và không ghi đè code, dữ liệu, báo cáo hoặc model gốc. Môi trường thực thi được ghi ở `results/env.json`. Script chỉ dùng các thư viện đã nằm trong requirements của các nhánh repo; không cài thêm thư viện.

`config/regression.yaml` là cú pháp JSON hợp lệ trong YAML 1.2 và được đọc bằng `json` của Python chuẩn, nên không cần PyYAML. Mỗi fit CatBoost/SVR có process riêng và time cap 15 giây. Fit SVR lớn có thể timeout hoặc không hội tụ; xem cột `status` và `warnings`.

Thiết kế runtime được giới hạn để hoàn thành kiểm chứng: CatBoost và SVR dùng một bộ tham số cố định định trước (tuning trials = 1), thay vì random search nhiều cấu hình; Ridge chọn alpha trên validation. Vì vậy số liệu trả lời so sánh theo cùng nhóm/số mẫu trong cấu hình hiện tại, chưa phải tìm kiếm siêu tham số với ngân sách tối ưu như nhau.

## Thứ tự chạy

Chạy từ thư mục gốc repo trong PowerShell:

```powershell
python .\experiments\scripts\00_capture_env.py
python .\experiments\scripts\01_audit_data.py
python .\experiments\scripts\02_fair_regression.py
python .\experiments\scripts\03_elective_compare.py
python .\experiments\scripts\04_make_figures.py
```

1. `01_audit_data.py` đọc schema và toàn bộ CSV 50k/ba bản 10k; ước tính dưới vài phút trên máy hiện tại. Kết quả: `audit_50k.json`, `audit_10k_identity.json`, `audit_10k_files.csv`, `audit_group_variation.csv`, `audit_target_linear_r2.csv`, `audit_target_correlations.csv`, `audit_nearest_distances.csv`.
2. `02_fair_regression.py` lặp 5 seed, 4 cỡ train groups, 2 feature sets, 3 target, CatBoost/SVR cùng baseline/Ridge. Thời gian có thể nhiều phút đến hàng chục phút tùy timeout SVR. Kết quả được checkpoint sau mỗi seed tại `regression_metrics.csv`, `regression_timing.csv`, và paired CI tại `regression_pairwise_ci.csv` khi kết thúc.
3. `03_elective_compare.py` chỉ chạy trên bản 10k `CATBoost-SVR/dulieuhocsinh_10000.csv`; checksum trùng với bản trong nhánh Logistic Regression, nhưng không trùng với bản XGBoost. Ước tính vài phút đến hàng chục phút; huấn luyện 9 nhãn × 4 mô hình × 5 seed. Kết quả: `elective_metrics_by_seed.csv`, `elective_metrics_summary.csv`, `elective_per_label_roc_auc.csv`, `elective_protocol.json`.
4. `04_make_figures.py` tạo biểu đồ learning curve và residual theo lớp 9 trong `figures/` từ CSV kết quả.

## Kết quả và diễn giải

- `RESULTS.md` tổng hợp kết quả, trạng thái fit, giới hạn thiết kế và đối chiếu với các nhận định trước đó.
- `../bao_cao_gvhd.md` là bản tóm tắt dành cho giáo viên hướng dẫn.
- Mọi kết luận chỉ áp dụng cho dữ liệu/nhãn mô phỏng đã kiểm tra. Không suy ra khả năng tư vấn tuyển sinh thực tế.
