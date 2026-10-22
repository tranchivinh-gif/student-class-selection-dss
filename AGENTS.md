# Hướng dẫn làm việc với dự án nghiên cứu khoa học

## Bối cảnh dự án

- Đề tài nghiên cứu hệ thống hỗ trợ học sinh lớp 9 cân nhắc chọn trường THPT, dự kiến kết hợp dự báo điểm thi với tiêu chí cá nhân.
- Hướng so sánh mô hình được đề xuất là `CatBoostRegressor` và `SVR`; không mặc định mô hình nào tốt hơn trước khi thực nghiệm.
- Tách bạch ba việc: dự báo điểm, đánh giá nguyện vọng và gợi ý trường. Không xem điểm dự báo, khả năng trúng tuyển và mức phù hợp cá nhân là cùng một đại lượng.
- Xem `Mo_ta.md` và `mo_ta_dl.md` là tài liệu định hướng/diễn giải dữ liệu; kiểm tra mã và dữ liệu hiện có trước khi kết luận về trạng thái triển khai.

## Dữ liệu và giới hạn diễn giải

- Dữ liệu gốc `dulieuhocsinh_10000.csv` gồm 10.000 hồ sơ giả định, 60 cột; `ma_hoc_sinh` là mã định danh, không đưa trực tiếp làm đặc trưng.
- Tệp gốc có điểm các môn theo lớp và 9 cột lựa chọn/lĩnh vực quan tâm dạng `yes/no`; tệp này không có nhãn điểm thi cuối kỳ, điểm thi thử hay dữ liệu trường/nguyện vọng. Không tự tạo hoặc khẳng định đã có các nhãn đó nếu chưa có quy trình được mô tả và kiểm chứng.
- Bộ mở rộng `dulieuhocsinh_50000_mo_phong.csv` vẫn là dữ liệu mô phỏng. Năm hồ sơ có chung `nhom_ho_so_goc` có quan hệ phụ thuộc; khi chia train/validation/test hoặc tạo fold, phải giữ toàn bộ nhóm trong cùng một phần để tránh rò rỉ dữ liệu.
- Nêu rõ các giả định mô phỏng, cách tạo nhãn và cấu hình thực nghiệm. Không diễn giải kết quả mô phỏng thành hiệu quả tuyển sinh thực tế, quan hệ nhân quả, năng lực cố định hay chất lượng tư vấn đã được xác nhận.
- Nếu dùng dữ liệu 10.000 dòng, đọc theo UTF-8 BOM khi cần; chuẩn hóa khoảng trắng đầu/cuối tên cột (có cột ban đầu có khoảng trắng), chuẩn hóa `yes/no` nhất quán trước phân tích. Kiểm tra miền điểm 0–10 và giá trị không hợp lệ.
- Đặc trưng chỉ được dùng nếu sẵn có tại mốc dự báo đã xác định; không dùng điểm thi cuối cùng, điểm chuẩn hiện tại hoặc kết quả tuyển sinh tương lai làm đầu vào cho dự báo trước kỳ thi.

## Giao tiếp

- Trả lời bằng tiếng Việt, ngắn gọn và rõ ràng.
- Hỏi lại khi yêu cầu chưa rõ; không tự suy đoán thông tin chưa được xác nhận.

## Trước khi sửa

- Đọc code và tài liệu liên quan, bao gồm notebook trong `analys/` khi công việc liên quan.
- Tuân thủ cấu trúc, cách đặt tên và quy ước hiện có; chỉ sửa trong phạm vi yêu cầu.

## Khi viết code

- Tái sử dụng thành phần có sẵn.
- Không tự đổi tên bảng, cột hoặc API; giữ nguyên tên cột nguồn, chỉ tạo tên chuẩn hóa trong bước biến đổi có chủ đích.
- Viết từ khóa SQL bằng chữ thường.
- Không ghi mật khẩu hoặc API key vào code.
- Ghi lại hạt giống ngẫu nhiên, cách chia dữ liệu, đặc trưng, nhãn, chỉ số đánh giá và cấu hình để thực nghiệm có thể tái lập.

## Kiểm tra và báo cáo

- Chạy kiểm tra phù hợp với phần đã sửa.
- Nêu rõ thay đổi và kết quả kiểm tra; nếu chưa kiểm tra được, nói rõ lý do.
