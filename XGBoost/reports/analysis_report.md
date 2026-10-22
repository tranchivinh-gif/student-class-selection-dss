# Báo cáo phân tích mô hình

- Mẫu phân tích: 1,500 kết quả trên tập Test.
- Phân tích chỉ đọc mô hình đã huấn luyện và dự đoán Test; không huấn luyện lại hoặc thay đổi dữ liệu.

## Tỷ lệ dự đoán và lỗi theo môn

| Môn | Actual Yes | Predicted Yes | False Positive | False Negative | Tỷ lệ sai |
|---|---:|---:|---:|---:|---:|
| vat_ly | 700 | 637 | 296 | 359 | 43.7% |
| hoa_hoc | 651 | 645 | 314 | 320 | 42.3% |
| sinh_hoc | 647 | 642 | 330 | 335 | 44.3% |
| cong_nghe | 683 | 709 | 320 | 294 | 40.9% |
| tin_hoc_tu_chon | 634 | 637 | 299 | 296 | 39.7% |
| dia_ly | 683 | 681 | 309 | 311 | 41.3% |
| giao_duc_kinh_te_va_phap_luat | 659 | 712 | 311 | 258 | 37.9% |
| am_nhac_tu_chon | 670 | 624 | 279 | 325 | 40.3% |
| mi_thuat_tu_chon | 673 | 713 | 331 | 291 | 41.5% |

False Positive là dự đoán Yes khi nhãn thực tế No; False Negative là dự đoán No khi nhãn thực tế Yes.

## Feature Importance

Bảng xếp hạng dưới đây là trung bình importance của các mô hình nhị phân theo từng nhãn. Đây là mức đóng góp dựa trên chia nhánh trong mô hình, không chứng minh quan hệ nhân quả.

| Đặc trưng điểm | Mean importance |
|---|---:|
| khoa_hoc_tu_nhien_lop8 | 0.045788 |
| khoa_hoc_tu_nhien_lop9 | 0.041450 |
| nghe_thuat_lop8 | 0.037471 |
| khoa_hoc_tu_nhien_lop7 | 0.033957 |
| giao_duc_cong_dan_lop8 | 0.032538 |
| nghe_thuat_lop9 | 0.032443 |
| lich_su_dia_ly_lop9 | 0.031771 |
| nghe_thuat_lop7 | 0.029885 |
| lich_su_dia_ly_lop8 | 0.029748 |
| khoa_hoc_tu_nhien_lop6 | 0.029043 |
| cong_nghe_lop8 | 0.027145 |
| giao_duc_cong_dan_lop9 | 0.026992 |
| tin_hoc_lop9 | 0.026948 |
| tin_hoc_lop8 | 0.026910 |
| tin_hoc_lop7 | 0.026308 |

Biểu đồ: `feature_importance.png` và `prediction_distribution.png`.

Kết quả và importance chỉ phản ánh dữ liệu thu thập được. Không sử dụng điểm importance hoặc xác suất như tư vấn môn học phù hợp cho từng học sinh.
