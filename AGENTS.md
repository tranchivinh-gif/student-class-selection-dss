# Hướng dẫn làm việc với repo

## Dự án

Repo gồm các nguyên mẫu dự báo điểm thi mô phỏng, dự đoán nhãn môn tự chọn và phân cụm điểm học sinh. Tài liệu nghiên cứu đề xuất thêm hướng hỗ trợ chọn trường THPT; hiện chưa có dữ liệu trường/nguyện vọng thực tế đủ để kiểm chứng hướng đó.

## Cấu trúc chính

- `CATBoost-SVR/`: CatBoostRegressor, SVR, CatBoostClassifier, SVC, dữ liệu, notebook, model artifacts và tài liệu nghiên cứu.
- `XGBoost/`: pipeline EDA, tiền xử lý, huấn luyện/đánh giá XGBoost multi-label, giao diện và tests.
- `KNN-HQTT/Logistic_Regression_TuVanToHop/`: Logistic Regression đa nhãn, K-Means, scripts theo bước, output và giao diện.
- `README.md`: hướng dẫn chạy mặc định nhánh XGBoost.

## Cài đặt và chạy

Chỉ dùng các lệnh cài đặt/chạy được ghi trong README và requirements của từng nhánh. README gốc yêu cầu Python 3.11+ và cài `XGBoost/requirements.txt`; chạy pipeline XGBoost bằng `python .\XGBoost\run_pipeline.py` từ thư mục gốc. Các nhánh khác có README riêng. Không giả định lệnh của nhánh này áp dụng cho nhánh khác.

## Quy tắc dữ liệu và nghiên cứu

- Dữ liệu trong repo là mô phỏng hoặc nguồn nhãn chưa xác minh; không trình bày kết quả như năng lực dự báo trên học sinh thật.
- Không gọi nhãn lựa chọn/quan tâm Yes/No là môn phù hợp hoặc đăng ký chính thức nếu chưa có xác nhận nguồn.
- Không trộn dự báo điểm, khả năng trúng tuyển và độ phù hợp cá nhân thành một đại lượng.
- Giữ seed, checksum, feature/label, split và nhóm hồ sơ; không đưa ID vào đặc trưng.
- Với bộ 50.000 hồ sơ, giữ `nhom_ho_so_goc` trong cùng một partition/fold.
- Kiểm tra mốc thời gian từng feature để ngăn data leakage; điểm thi thử chỉ dùng khi có trước thời điểm dự báo.
- Mọi số liệu trong báo cáo phải dẫn file/metadata/report nguồn. Phân biệt bằng chứng, suy luận và giả định.

## Quy ước khi sửa đổi

- Đọc hướng dẫn cục bộ trong thư mục con trước khi sửa nội dung ở đó; hiện có `CATBoost-SVR/AGENTS.md`.
- Không sửa/xóa dữ liệu gốc hay model artifacts nếu không được yêu cầu rõ.
- Giữ nguyên tên cột/API hiện hành trừ khi thay đổi được yêu cầu và cập nhật đầy đủ nơi dùng.
- Không lưu bí mật/API key trong repo. Ưu tiên script/pipeline tái lập, ghi cấu hình và phiên bản thư viện.
- Chỉ chạy tests/huấn luyện khi tác vụ yêu cầu kiểm tra hoặc người dùng yêu cầu; báo chính xác những gì đã chạy.
- Với báo cáo nghiên cứu, không khẳng định tính mới, tính nhân quả, tư vấn đáng tin cậy hoặc hiệu quả tuyển sinh nếu chưa có bằng chứng tương ứng.

## Tài liệu định hướng

- `report.md`: rà soát inventory, task, kết quả lưu sẵn, rủi ro và chấm điểm mô hình.
- `detai.md`: đề cương nghiên cứu và các quyết định nhóm cần chốt.
- `CATBoost-SVR/Mo_ta.md`, `CATBoost-SVR/mo_ta_dl.md`: mô tả đề tài/dữ liệu; đối chiếu với code, metadata và reports trước khi dùng làm bằng chứng triển khai.
