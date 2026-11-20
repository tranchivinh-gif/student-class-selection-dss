# Phân tích dữ liệu thăm dò — pilot seed 20261108

> Chỉ là pilot thăm dò dữ liệu tổng hợp. Các kết quả này không phải quan sát về học sinh thật,
> không phải bằng chứng hiệu năng mô hình; thiết kế C1–C6 đã được xác nhận.

## Phân bố nhãn C1–C6

| Nhãn | Số mẫu | Tỷ lệ (%) |
|---|---:|---:|
| C1 | 189 | 18.90 |
| C2 | 181 | 18.10 |
| C3 | 161 | 16.10 |
| C4 | 175 | 17.50 |
| C5 | 154 | 15.40 |
| C6 | 140 | 14.00 |

## Xác suất Softmax trung bình

| Nhãn | Xác suất Softmax trung bình |
|---|---:|
| C1 | 0.1771 |
| C2 | 0.1643 |
| C3 | 0.1615 |
| C4 | 0.1658 |
| C5 | 0.1701 |
| C6 | 0.1612 |

Giá trị là trung bình xác suất Softmax trước bước lấy mẫu nhãn.

## Bằng chứng kiểm tra ma trận tương quan R_Z

- Đối xứng: **True**.
- Đường chéo bằng 1: **True**.
- Tất cả trị riêng dương: **True** (0.685292, 0.697375, 0.809123, 1.179005, 1.629206).
- Cholesky decomposition: **success**.

Ma trận R_Z chính xác được sử dụng:

| Năng lực | quantitative | language | scientific_inquiry | social | digital_technology |
|---|---:|---:|---:|---:|---:|
| quantitative | 1.0000 | 0.1000 | 0.3000 | 0.0500 | 0.2500 |
| language | 0.1000 | 1.0000 | 0.1000 | 0.3000 | 0.1000 |
| scientific_inquiry | 0.3000 | 0.1000 | 1.0000 | 0.0500 | 0.2000 |
| social | 0.0500 | 0.3000 | 0.0500 | 1.0000 | 0.1000 |
| digital_technology | 0.2500 | 0.1000 | 0.2000 | 0.1000 | 1.0000 |

## Phân phối các chiều S và D

| Nhóm | Chiều | Mean | SD | Min | Q25 | Median | Q75 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D_career_interest | artistic | 0.5065 | 0.2906 | 0.0002 | 0.2452 | 0.5173 | 0.7683 | 0.9993 |
| D_career_interest | conventional | 0.4902 | 0.2887 | 0.0027 | 0.2359 | 0.4821 | 0.7432 | 0.9986 |
| D_career_interest | enterprising | 0.4983 | 0.2924 | 0.0014 | 0.2429 | 0.5054 | 0.7531 | 0.9998 |
| D_career_interest | investigative | 0.4790 | 0.2871 | 0.0041 | 0.2306 | 0.4600 | 0.7343 | 0.9990 |
| D_career_interest | realistic | 0.4986 | 0.2867 | 0.0004 | 0.2528 | 0.4967 | 0.7454 | 1.0000 |
| D_career_interest | social | 0.4991 | 0.2841 | 0.0002 | 0.2572 | 0.5162 | 0.7364 | 0.9990 |
| S_subject_interest | biology | 0.5019 | 0.2946 | 0.0005 | 0.2463 | 0.5134 | 0.7549 | 0.9985 |
| S_subject_interest | chemistry | 0.4774 | 0.2839 | 0.0030 | 0.2264 | 0.4550 | 0.7335 | 0.9986 |
| S_subject_interest | economics_and_law | 0.4947 | 0.2863 | 0.0030 | 0.2389 | 0.4868 | 0.7451 | 0.9988 |
| S_subject_interest | geography | 0.5037 | 0.2902 | 0.0037 | 0.2515 | 0.4943 | 0.7559 | 0.9993 |
| S_subject_interest | informatics | 0.4879 | 0.2886 | 0.0023 | 0.2493 | 0.4719 | 0.7344 | 0.9985 |
| S_subject_interest | music | 0.4907 | 0.2957 | 0.0014 | 0.2344 | 0.4853 | 0.7521 | 0.9992 |
| S_subject_interest | physics | 0.4882 | 0.2857 | 0.0005 | 0.2441 | 0.4793 | 0.7365 | 0.9995 |
| S_subject_interest | technology | 0.4851 | 0.2918 | 0.0014 | 0.2302 | 0.4771 | 0.7398 | 0.9978 |
| S_subject_interest | visual_arts | 0.4976 | 0.2947 | 0.0002 | 0.2389 | 0.4904 | 0.7585 | 0.9986 |

S là các chiều sở thích môn học; D là các chiều định hướng nghề nghiệp RIASEC. Các giá trị được biến đổi vào [0;1] bằng CDF chuẩn chuẩn hóa.

- Chênh lệch C1–C6: 49 mẫu (4.90 điểm phần trăm).
- Tỷ số số mẫu C1/C6: 1.35.
- Độ lệch chuẩn tỷ lệ lớp: 1.67 điểm phần trăm.
- Đánh giá mất cân bằng: **thấp** (quy ước: tỷ số lớp lớn nhất/nhỏ nhất ≤ 1,5 là thấp; ≤ 2,0 là vừa).
- Entropy phân phối nhãn: **1.7867 nat**.
- Entropy chuẩn hóa: **0.9972** (H / ln(6)); 1,0 là phân phối đều hoàn toàn.
- Công thức: H = −Σ pᵢ ln(pᵢ), với pᵢ là tỷ lệ của lớp i.

## Kiểm tra dữ liệu

- Số bản ghi: 1,000
- Đặc trưng điểm cả năm: 32
- Khoảng điểm quan sát: 2.9–10.0
- Ô điểm bị thiếu: 0
- Ô điểm bằng 0 hoặc 10: 0.33%
- Nhãn không có bản ghi: 0
- Tỷ lệ lớp lớn nhất/nhỏ nhất: 1.35

## Tệp

- `score_summary.csv`: thống kê mô tả và giá trị biên theo đặc trưng.
- `label_distribution.csv`: số lượng và tỷ lệ phần trăm theo lớp.
- `mean_softmax_probabilities.csv`: xác suất Softmax trung bình theo lớp.
- `correlation_matrix_R_Z.csv`: ma trận R_Z chính xác được sử dụng.
- `interest_career_dimension_summary.csv`: bảng mô tả các chiều S và D.
- `score_distribution_by_subject.csv`, `score_distribution_by_grade.csv`: mean và độ lệch chuẩn theo môn và khối.
- `subject_correlations.csv`: tương quan giữa điểm trung bình các môn.
- `within_subject_year_correlations.csv`: tương quan cùng môn qua các khối 6–9.
- `score_correlations.csv`: tương quan từng cặp của các đặc trưng điểm.
- `score_means_by_class.csv`: trung bình đặc trưng theo nhãn được sinh.
- `score_distributions.png`, `score_correlations.png`, `label_distribution.png`: biểu đồ.

## Kiểm tra phân phối điểm học tập

- Clipping tại 0: 0.00%; clipping tại 10: 0.33%.
- Tỷ lệ giá trị trùng sau làm tròn 1 chữ số: 99.78% (1 − số giá trị duy nhất / tổng số ô điểm; 71/32000).
- Tương quan liên môn được tính trên điểm trung bình bốn khối của từng môn; tương quan cùng môn được tính cho từng cặp khối trong 6–7, 6–8, 6–9, 7–8, 7–9, 8–9.
- μ_s = 7,0 cho cả tám môn là giả định đơn giản hóa của benchmark, không phải khẳng định điểm trung bình các môn ngoài thực tế bằng nhau.

## Giới hạn diễn giải

Dùng các đầu ra này để kiểm tra bộ sinh hoạt động đúng đặc tả. Mọi thay đổi giả định của bộ sinh phải được ghi lại trước khi sinh các bộ dữ liệu lớn hơn. Pilot 1.000 bản ghi quá nhỏ để khẳng định tính hợp lệ thực tế hoặc sự cân bằng lớp cuối cùng.
