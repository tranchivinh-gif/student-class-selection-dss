# Trạng thái công việc thực nghiệm

**Cập nhật:** 26/10/2026 (Asia/Saigon). **Trạng thái:** đã dừng theo yêu cầu; kết quả regression hiện chỉ có hai seed hoàn chỉnh. Đây là bản ghi tiến độ, không phải báo cáo kết quả cuối cùng.

## Đã hoàn thành

- Đọc lại `report.md`, `detai.md`, `AGENTS.md`; kiểm tra requirements trước khi chạy.
- Ghi môi trường ở [`experiments/results/env.json`](experiments/results/env.json): Python 3.10.11, Windows 10, 12 logical CPU; các phiên bản thư viện đã cài được liệt kê trong file. Git commit lúc bắt đầu: `ee5dd635d3b346f98164a29795e6bd4d96b1f3cd`.
- Kiểm toán bộ mô phỏng 50k. Nguồn: [`audit_50k.json`](experiments/results/audit_50k.json), [`audit_group_variation.csv`](experiments/results/audit_group_variation.csv), [`audit_target_linear_r2.csv`](experiments/results/audit_target_linear_r2.csv), [`audit_target_correlations.csv`](experiments/results/audit_target_correlations.csv), [`audit_nearest_distances.csv`](experiments/results/audit_nearest_distances.csv).
- Đối chiếu ba tệp 10k. Hai tệp `CATBoost-SVR/dulieuhocsinh_10000.csv` và `KNN-HQTT/.../data/raw/dulieuhocsinh_10000.csv` có cùng SHA-256; `XGBoost/data/hoc_sinh_moi.csv` khác checksum và schema. Nguồn: [`audit_10k_identity.json`](experiments/results/audit_10k_identity.json).
- Tạo script audit, regression, elective và vẽ hình; cấu hình regression ở [`experiments/config/regression.yaml`](experiments/config/regression.yaml). Chỉ audit đã chạy hoàn tất. Các script không sửa dữ liệu hay model gốc.

## Kết quả kiểm toán dữ liệu

- Bộ 50k có 50.000 dòng, 131 cột, 10.000 nhóm; mỗi nhóm có 5 dòng. Không có ô thiếu, dòng trùng hoặc mã học sinh trùng; các cột điểm được rà đều nằm trong khoảng 0–10. SHA-256: `f8dff20fda3296cee13c9ee01be7035a575c5e3f5b10b1674bf69ac7f97878f8`. Nguồn: [`audit_50k.json`](experiments/results/audit_50k.json).
- Hồi quy tuyến tính chỉ dùng điểm lớp 9 đạt R² test lần lượt: Toán 0,804396; Ngữ văn 0,864761; Ngoại ngữ 0,821699. Thêm điểm thi thử, R² là 0,825134; 0,866173; 0,839220. Dùng đủ feature tuyến tính, R² là 0,866234; 0,875930; 0,874656. Nguồn: [`audit_target_linear_r2.csv`](experiments/results/audit_target_linear_r2.csv). Đây là dấu hiệu target mô phỏng có quan hệ tuyến tính mạnh với một số feature, không phải bằng chứng hiệu năng trên học sinh thật.
- Trên mẫu 300 nhóm test đã chuẩn hóa, khoảng cách láng giềng gần nhất từ dòng test sang nhóm train khác có trung vị 5,389907; khoảng cách cặp trong cùng nhóm có trung vị 4,228815. Nguồn: [`audit_nearest_distances.csv`](experiments/results/audit_nearest_distances.csv). Đây là so sánh mô tả trên mẫu, không phải kiểm định độc lập hoàn chỉnh.

## Regression: tiến độ và trạng thái

- Kế hoạch cấu hình ghi 5 seed (`42, 101, 202, 303, 404`), 4 cỡ train nhóm (`300, 1.000, 3.000, 6.000`), hai bộ feature và ba target. Nguồn: [`regression.yaml`](experiments/config/regression.yaml).
- Kết quả đã checkpoint cho seed 42 và 101; mỗi seed có 96 dòng trong [`regression_metrics.csv`](experiments/results/regression_metrics.csv). Seed 202 đang chạy nhưng chưa checkpoint khi bị dừng; seed 303 và 404 chưa chạy xong.
- Mỗi seed đã checkpoint có 24 lần CatBoost thành công, 24 baseline lớp 9, 24 Ridge, và SVR gồm 6 thành công, 6 không hội tụ, 12 timeout (time cap 15 giây). Nguồn: [`regression_metrics.csv`](experiments/results/regression_metrics.csv), [`regression_timing.csv`](experiments/results/regression_timing.csv). Metric SVR không hội tụ được giữ với trạng thái riêng; timeout không có metric.
- Fit SVR bị gắn `non_converged` có cảnh báo chạm `max_iter=20000`; cấu hình lớn có trạng thái `timeout`. Kết quả fit dở trong thư mục tạm `experiments/results/fit__zmy_j7k/` không được đưa vào CSV tổng hợp và không được coi là một run hoàn chỉnh.
- Thiết kế đã giản lược so với yêu cầu ban đầu: mỗi thuật toán CatBoost/SVR dùng một cấu hình cố định (một trial), không thực hiện random search; Ridge chọn alpha từ validation. Do đó chưa thể kết luận về mô hình sau tuning công bằng.
- Chưa tạo `regression_pairwise_ci.csv`: cần đủ các seed và dữ liệu dự đoán để bootstrap ghép cặp. Chưa tạo biểu đồ learning curve/residual. Script vẽ hình chưa được chạy.

## Chưa thực hiện

- Chưa chạy so sánh nhãn môn tự chọn. Script đã tạo, nhưng chỉ nên chạy trên file 10k có checksum trùng giữa nhánh CatBoost-SVR và Logistic Regression; bản XGBoost khác checksum/schema.
- Chưa chạy tests, không huấn luyện model elective, không lưu model mới.
- Chưa tạo `experiments/RESULTS.md` hoặc `bao_cao_gvhd.md`. Vì vậy chưa có kết luận tổng hợp cuối hay đề xuất cập nhật chính thức cho `report.md`/`detai.md`.
- Chưa hoàn thành yêu cầu 5 seed, bootstrap 1.000 lần, câu trả lời định lượng tổng hợp, timing/learning curve đầy đủ và residual analysis đủ mô hình.

## Trạng thái Git

Trước khi chạy, `git status` ghi nhận `CATBoost-SVR/analys/catboost.ipynb` đã bị sửa từ trước; file đó không được chỉnh bởi lượt thực nghiệm và không được đưa vào commit. `AGENTS.md`, `detai.md`, `report.md` là các file mới từ lượt trước. Tại thời điểm cập nhật, các file thực nghiệm mới và `status.md` chưa được commit. Không có commit/push/merge nào được thực hiện trong quá trình thực nghiệm.

## Bước tiếp theo để hoàn tất

1. Thêm chế độ resume để chạy tiếp từ seed 202, hoặc chạy lại đủ 5 seed từ đầu.
2. Quyết định có tăng time cap/điều chỉnh cấu hình SVR hay giữ các timeout như một giới hạn tính toán được báo cáo.
3. Hoàn tất regression, paired bootstrap CI, residual summaries và figures.
4. Chạy elective comparison theo cùng split/seed trên bản 10k đã xác định; sau đó tạo `RESULTS.md` và `bao_cao_gvhd.md`.
5. Rà lại Git để loại notebook đã sửa trước đó khỏi mọi commit.
