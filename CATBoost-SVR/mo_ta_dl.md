# Mô tả dữ liệu `dulieuhocsinh_10000.csv`

## 1. Thông tin tổng quan

| Thuộc tính | Giá trị |
|---|---|
| Tên tệp | `dulieuhocsinh_10000.csv` |
| Số dòng dữ liệu | 10.000 |
| Số cột | 60 |
| Dòng tiêu đề | 1 |
| Mã định danh | `ma_hoc_sinh` |
| Định dạng | CSV, các trường phân cách bằng dấu phẩy |
| Kiểu mã hóa dự kiến | UTF-8 có BOM |
| Dữ liệu thiếu trong các cột đã kiểm tra | Không có |
| Dữ liệu trùng mã học sinh | Không có |

Bộ dữ liệu gồm thông tin điểm học tập của học sinh theo môn và khối lớp,
cùng với chín trường biểu thị các môn hoặc lĩnh vực được lựa chọn/quan tâm.
Các cột điểm trong tệp hiện có giá trị nguyên từ 0 đến 10.

> **Lưu ý:** Tệp này chỉ chứa mã học sinh, điểm học tập và các trường
> lựa chọn môn học. Tệp không chứa điểm thi mô phỏng, điểm thi thử, thông
> tin trường, học phí, khoảng cách, nguyện vọng hoặc kết quả tuyển sinh.

## 2. Phân nhóm các trường dữ liệu

| Nhóm | Vị trí cột | Số lượng | Kiểu dữ liệu chính |
|---|---:|---:|---|
| Mã định danh | 1 | 1 | Chuỗi |
| Điểm học tập | 2–51 | 50 | Số nguyên, từ 0 đến 10 |
| Lựa chọn/lĩnh vực quan tâm | 52–60 | 9 | Chuỗi nhị phân `yes/no` |

## 3. Mô tả chi tiết các trường

### 3.1. Trường định danh

| STT | Tên trường | Kiểu dữ liệu | Miền/giá trị | Mô tả |
|---:|---|---|---|---|
| 1 | `ma_hoc_sinh` | Chuỗi | Mã UUID | Mã định danh duy nhất của học sinh. Không dùng làm đặc trưng học máy trực tiếp. |

### 3.2. Điểm Toán

| STT | Tên trường | Kiểu dữ liệu | Miền giá trị | Mô tả |
|---:|---|---|---|---|
| 2 | `toan_lop6` | Số nguyên | 0–10 | Điểm môn Toán ở lớp 6. |
| 3 | `toan_lop7` | Số nguyên | 0–10 | Điểm môn Toán ở lớp 7. |
| 4 | `toan_lop8` | Số nguyên | 0–10 | Điểm môn Toán ở lớp 8. |
| 5 | `toan_lop9` | Số nguyên | 0–10 | Điểm môn Toán ở lớp 9. |

### 3.3. Điểm Ngữ văn

| STT | Tên trường | Kiểu dữ liệu | Miền giá trị | Mô tả |
|---:|---|---|---|---|
| 6 | `ngu_van_lop6` | Số nguyên | 0–10 | Điểm môn Ngữ văn ở lớp 6. |
| 7 | `ngu_van_lop7` | Số nguyên | 0–10 | Điểm môn Ngữ văn ở lớp 7. |

### 3.4. Điểm Ngoại ngữ

| STT | Tên trường | Kiểu dữ liệu | Miền giá trị | Mô tả |
|---:|---|---|---|---|
| 8 | `ngoai_ngu_lop6` | Số nguyên | 0–10 | Điểm môn Ngoại ngữ ở lớp 6. |
| 9 | `ngoai_ngu_lop7` | Số nguyên | 0–10 | Điểm môn Ngoại ngữ ở lớp 7. |
| 10 | `ngoai_ngu_lop8` | Số nguyên | 0–10 | Điểm môn Ngoại ngữ ở lớp 8. |
| 11 | `ngoai_ngu_lop9` | Số nguyên | 0–10 | Điểm môn Ngoại ngữ ở lớp 9. |

### 3.5. Điểm Khoa học tự nhiên

| STT | Tên trường | Kiểu dữ liệu | Miền giá trị | Mô tả |
|---:|---|---|---|---|
| 12 | `khoa_hoc_tu_nhien_lop6` | Số nguyên | 0–10 | Điểm môn Khoa học tự nhiên ở lớp 6. |
| 13 | `khoa_hoc_tu_nhien_lop7` | Số nguyên | 0–10 | Điểm môn Khoa học tự nhiên ở lớp 7. |
| 14 | `khoa_hoc_tu_nhien_lop8` | Số nguyên | 0–10 | Điểm môn Khoa học tự nhiên ở lớp 8. |
| 15 | `khoa_hoc_tu_nhien_lop9` | Số nguyên | 0–10 | Điểm môn Khoa học tự nhiên ở lớp 9. |

### 3.6. Điểm Lịch sử và Địa lý

| STT | Tên trường | Kiểu dữ liệu | Miền giá trị | Mô tả |
|---:|---|---|---|---|
| 16 | `lich_su_dia_ly_lop6` | Số nguyên | 0–10 | Điểm môn Lịch sử và Địa lý ở lớp 6. |
| 17 | `lich_su_dia_ly_lop7` | Số nguyên | 0–10 | Điểm môn Lịch sử và Địa lý ở lớp 7. |
| 18 | `lich_su_dia_ly_lop8` | Số nguyên | 0–10 | Điểm môn Lịch sử và Địa lý ở lớp 8. |
| 19 | `lich_su_dia_ly_lop9` | Số nguyên | 0–10 | Điểm môn Lịch sử và Địa lý ở lớp 9. |

### 3.7. Điểm Giáo dục công dân

| STT | Tên trường | Kiểu dữ liệu | Miền giá trị | Mô tả |
|---:|---|---|---|---|
| 20 | `giao_duc_cong_dan_lop6` | Số nguyên | 0–10 | Điểm môn Giáo dục công dân ở lớp 6. |
| 21 | `giao_duc_cong_dan_lop7` | Số nguyên | 0–10 | Điểm môn Giáo dục công dân ở lớp 7. |
| 22 | `giao_duc_cong_dan_lop8` | Số nguyên | 0–10 | Điểm môn Giáo dục công dân ở lớp 8. |
| 23 | `giao_duc_cong_dan_lop9` | Số nguyên | 0–10 | Điểm môn Giáo dục công dân ở lớp 9. |

### 3.8. Điểm Tin học

| STT | Tên trường | Kiểu dữ liệu | Miền giá trị | Mô tả |
|---:|---|---|---|---|
| 24 | `tin_hoc_lop6` | Số nguyên | 0–10 | Điểm môn Tin học ở lớp 6. |
| 25 | `tin_hoc_lop7` | Số nguyên | 0–10 | Điểm môn Tin học ở lớp 7. |
| 26 | `tin_hoc_lop8` | Số nguyên | 0–10 | Điểm môn Tin học ở lớp 8. |
| 27 | `tin_hoc_lop9` | Số nguyên | 0–10 | Điểm môn Tin học ở lớp 9. |

### 3.9. Điểm Công nghệ

| STT | Tên trường | Kiểu dữ liệu | Miền giá trị | Mô tả |
|---:|---|---|---|---|
| 28 | `cong_nghe_lop6` | Số nguyên | 0–10 | Điểm môn Công nghệ ở lớp 6. |
| 29 | `cong_nghe_lop7` | Số nguyên | 0–10 | Điểm môn Công nghệ ở lớp 7. |
| 30 | `cong_nghe_lop8` | Số nguyên | 0–10 | Điểm môn Công nghệ ở lớp 8. |
| 31 | `cong_nghe_lop9` | Số nguyên | 0–10 | Điểm môn Công nghệ ở lớp 9. |

### 3.10. Điểm Giáo dục thể chất

| STT | Tên trường | Kiểu dữ liệu | Miền giá trị | Mô tả |
|---:|---|---|---|---|
| 32 | `giao_duc_the_chat_lop6` | Số nguyên | 0–10 | Điểm môn Giáo dục thể chất ở lớp 6. |
| 33 | `giao_duc_the_chat_lop7` | Số nguyên | 0–10 | Điểm môn Giáo dục thể chất ở lớp 7. |
| 34 | `giao_duc_the_chat_lop8` | Số nguyên | 0–10 | Điểm môn Giáo dục thể chất ở lớp 8. |
| 35 | `giao_duc_the_chat_lop9` | Số nguyên | 0–10 | Điểm môn Giáo dục thể chất ở lớp 9. |

### 3.11. Điểm Âm nhạc

| STT | Tên trường | Kiểu dữ liệu | Miền giá trị | Mô tả |
|---:|---|---|---|---|
| 36 | `am_nhac_lop6` | Số nguyên | 0–10 | Điểm môn Âm nhạc ở lớp 6. |
| 37 | `am_nhac_lop7` | Số nguyên | 0–10 | Điểm môn Âm nhạc ở lớp 7. |
| 38 | `am_nhac_lop8` | Số nguyên | 0–10 | Điểm môn Âm nhạc ở lớp 8. |
| 39 | `am_nhac_lop9` | Số nguyên | 0–10 | Điểm môn Âm nhạc ở lớp 9. |

### 3.12. Điểm Mỹ thuật

| STT | Tên trường | Kiểu dữ liệu | Miền giá trị | Mô tả |
|---:|---|---|---|---|
| 40 | `mi_thuat_lop6` | Số nguyên | 0–10 | Điểm môn Mỹ thuật ở lớp 6. |
| 41 | `mi_thuat_lop7` | Số nguyên | 0–10 | Điểm môn Mỹ thuật ở lớp 7. |
| 42 | `mi_thuat_lop8` | Số nguyên | 0–10 | Điểm môn Mỹ thuật ở lớp 8. |
| 43 | `mi_thuat_lop9` | Số nguyên | 0–10 | Điểm môn Mỹ thuật ở lớp 9. |

### 3.13. Điểm Nội dung giáo dục địa phương

| STT | Tên trường | Kiểu dữ liệu | Miền giá trị | Mô tả |
|---:|---|---|---|---|
| 44 | `noi_dung_giao_duc_dia_phuong_lop6` | Số nguyên | 0–10 | Điểm Nội dung giáo dục địa phương ở lớp 6. |
| 45 | `noi_dung_giao_duc_dia_phuong_lop7` | Số nguyên | 0–10 | Điểm Nội dung giáo dục địa phương ở lớp 7. |
| 46 | `noi_dung_giao_duc_dia_phuong_lop8` | Số nguyên | 0–10 | Điểm Nội dung giáo dục địa phương ở lớp 8. |
| 47 | `noi_dung_giao_duc_dia_phuong_lop9` | Số nguyên | 0–10 | Điểm Nội dung giáo dục địa phương ở lớp 9. |

### 3.14. Điểm Hoạt động trải nghiệm, hướng nghiệp

| STT | Tên trường | Kiểu dữ liệu | Miền giá trị | Mô tả |
|---:|---|---|---|---|
| 48 | `hoat_dong_trai_nghiem_huong_nghiep_lop6` | Số nguyên | 0–10 | Điểm Hoạt động trải nghiệm, hướng nghiệp ở lớp 6. |
| 49 | `hoat_dong_trai_nghiem_huong_nghiep_lop7` | Số nguyên | 0–10 | Điểm Hoạt động trải nghiệm, hướng nghiệp ở lớp 7. |
| 50 | `hoat_dong_trai_nghiem_huong_nghiep_lop8` | Số nguyên | 0–10 | Điểm Hoạt động trải nghiệm, hướng nghiệp ở lớp 8. |
| 51 | `hoat_dong_trai_nghiem_huong_nghiep_lop9` | Số nguyên | 0–10 | Điểm Hoạt động trải nghiệm, hướng nghiệp ở lớp 9. |

### 3.15. Các trường lựa chọn/lĩnh vực quan tâm

Các trường từ 52 đến 60 dùng để biểu diễn việc học sinh có lựa chọn hoặc
quan tâm đến một môn/lĩnh vực. Trong dữ liệu gốc, giá trị xuất hiện ở cả
dạng viết thường và viết hoa. Khi phân tích nên chuẩn hóa:

```text
yes, Yes -> 1
no, No   -> 0
```

| STT | Tên trường trong CSV | Kiểu dữ liệu | Giá trị quan sát | Ý nghĩa |
|---:|---|---|---|---|
| 52 | `Vật lý` | Chuỗi nhị phân | `yes`, `no`, `Yes`, `No` | Lựa chọn/quan tâm đến Vật lý. |
| 53 | `Hóa học` | Chuỗi nhị phân | `yes`, `no`, `Yes`, `No` | Lựa chọn/quan tâm đến Hóa học. |
| 54 | `Sinh học` | Chuỗi nhị phân | `yes`, `no`, `Yes`, `No` | Lựa chọn/quan tâm đến Sinh học. |
| 55 | `Công nghệ` | Chuỗi nhị phân | `yes`, `no`, `Yes`, `No` | Lựa chọn/quan tâm đến Công nghệ. |
| 56 | `Tin học` | Chuỗi nhị phân | `yes`, `no`, `Yes`, `No` | Lựa chọn/quan tâm đến Tin học. |
| 57 | `Địa lý` | Chuỗi nhị phân | `yes`, `no`, `Yes`, `No` | Lựa chọn/quan tâm đến Địa lý. |
| 58 | `Giáo dục kinh tế và pháp luật` | Chuỗi nhị phân | `yes`, `no`, `Yes`, `No` | Lựa chọn/quan tâm đến Giáo dục kinh tế và pháp luật. |
| 59 | ` Âm nhạc` | Chuỗi nhị phân | `yes`, `no`, `Yes`, `No` | Lựa chọn/quan tâm đến Âm nhạc. Tên cột có một khoảng trắng ở đầu. |
| 60 | `Mỹ thuật` | Chuỗi nhị phân | `yes`, `no`, `Yes`, `No` | Lựa chọn/quan tâm đến Mỹ thuật. |

## 4. Kiểu dữ liệu đề xuất khi nạp bằng pandas

```python
import pandas as pd

df = pd.read_csv(
    "dulieuhocsinh_10000.csv",
    encoding="utf-8-sig",
)

# Xóa khoảng trắng thừa ở tên cột, đặc biệt là cột " Âm nhạc".
df.columns = df.columns.str.strip()

# Chuẩn hóa các trường yes/no về 0/1.
binary_columns = [
    "Vật lý",
    "Hóa học",
    "Sinh học",
    "Công nghệ",
    "Tin học",
    "Địa lý",
    "Giáo dục kinh tế và pháp luật",
    "Âm nhạc",
    "Mỹ thuật",
]

for column in binary_columns:
    df[column] = (
        df[column]
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
```

Sau bước chuẩn hóa, có thể kiểm tra các giá trị không hợp lệ bằng:

```python
for column in binary_columns:
    invalid = df.loc[~df[column].isin([0, 1]), column]
    print(column, len(invalid))
```

## 5. Khuyến nghị sử dụng cho phân tích và mô hình

1. Không đưa `ma_hoc_sinh` trực tiếp vào mô hình vì đây là mã định danh,
   không mang ý nghĩa học tập.
2. Kiểm tra miền giá trị các cột điểm trước mỗi lần chạy thực nghiệm.
3. Chuẩn hóa tên cột bằng cách loại bỏ khoảng trắng đầu/cuối.
4. Chuẩn hóa các trường `yes/no` về dạng số trước khi huấn luyện mô hình.
5. Nếu dự báo điểm một môn, cần xác định rõ các cột điểm được phép sử dụng
   tại mốc dự báo.
6. Không xem các trường lựa chọn/lĩnh vực quan tâm là điểm năng lực nếu
   chưa có tài liệu xác nhận ý nghĩa chính xác của chúng.
7. File này chưa có cột nhãn điểm thi cuối kỳ. Muốn huấn luyện mô hình hồi
   quy cần có thêm cột đích tương ứng hoặc một quy trình tạo nhãn được mô
   tả và kiểm chứng riêng.

## 6. Giới hạn diễn giải

Bộ dữ liệu hiện tại chỉ cho phép mô tả và phân tích các đặc điểm học tập
được ghi trong tệp. Không nên suy ra từ dữ liệu này:

- Khả năng trúng tuyển vào một trường cụ thể.
- Chất lượng hoặc năng lực cố định của một học sinh.
- Quan hệ nhân quả giữa điểm số và sở thích.
- Hiệu quả của hệ thống tư vấn khi chưa có dữ liệu trường, nguyện vọng,
  nhãn tuyển sinh và đánh giá độc lập.

