# Báo cáo chuẩn bị dữ liệu

- Tệp nguồn: `hoc_sinh_moi.csv`
- SHA-256 tệp nguồn: `cb843034d435f7abdf8cdcc5cade4c78fca3eef27be390737756427e068ac21c`
- Số dòng: 10,000
- Số cột nguồn: 50
- Số đặc trưng điểm lớp 6-9: 40
- Số nhãn môn tự chọn: 9
- Dòng trùng lặp: 0
- Mã học sinh trùng: 0

## Kiểm tra chất lượng

- Giá trị thiếu: không có.
- Điểm ngoài khoảng [0, 10]: không có.
- Nhãn không thuộc Yes/No hoặc 0/1: không có.

## Thay đổi và giới hạn đầu vào

- Chỉ loại bỏ khoảng trắng đầu/cuối tên cột và nhãn; nhãn `Yes/No` được mã hóa thành `1/0`; nhãn `0/1` được giữ nguyên.
- Không xóa dòng hoặc sửa điểm/nhãn gốc.
- `ma_hoc_sinh` được giữ trong CSV đã làm sạch để truy vết nhưng sẽ không được đưa vào đặc trưng mô hình.
- Không tạo đặc trưng từ nhãn hoặc từ thống kê toàn tập.

## Phân bố nhãn

| Môn tự chọn | No (0) | Yes (1) |
|---|---:|---:|
| vat_ly | 5,528 | 4,472 |
| hoa_hoc | 5,620 | 4,380 |
| sinh_hoc | 5,578 | 4,422 |
| cong_nghe | 5,491 | 4,509 |
| tin_hoc_tu_chon | 5,570 | 4,430 |
| dia_ly | 5,550 | 4,450 |
| giao_duc_kinh_te_va_phap_luat | 5,543 | 4,457 |
| am_nhac_tu_chon | 5,631 | 4,369 |
| mi_thuat_tu_chon | 5,489 | 4,511 |
