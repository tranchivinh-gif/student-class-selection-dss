# Báo cáo đánh giá XGBoost đa nhãn

- Số mẫu test: 1,500
- Mỗi học sinh được đề xuất đúng 4 môn có xác suất cao nhất.
- Tỷ lệ học sinh thực tế chọn đúng 4 môn: 100.0%. Đây là trần lý thuyết của Exact Match Accuracy khi luôn đề xuất 4 môn.
- Tham số chọn trên Validation: `{'max_depth': 2, 'learning_rate': 0.05, 'n_estimators': 240, 'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 2.0}`
- Test không được dùng để chọn tham số.

## Chỉ số tổng thể trên Test

| Chỉ số | XGBoost | Baseline 4 môn phổ biến nhất trong Train |
|---|---:|---:|
| Micro Precision | 0.5352 | 0.4483 |
| Micro Recall | 0.5352 | 0.4483 |
| Micro F1 | 0.5352 | 0.4483 |
| Macro Precision | 0.5347 | 0.1993 |
| Macro Recall | 0.5351 | 0.4444 |
| Macro F1 | 0.5345 | 0.2751 |
| Exact Match Accuracy | 0.0293 | 0.0107 |

## Kết quả theo nhãn

| Môn | Precision | Recall | F1 | Số mẫu Yes trong Test |
|---|---:|---:|---:|---:|
| vat_ly | 0.5353 | 0.4871 | 0.5101 | 700 |
| hoa_hoc | 0.5132 | 0.5084 | 0.5108 | 651 |
| sinh_hoc | 0.4860 | 0.4822 | 0.4841 | 647 |
| cong_nghe | 0.5487 | 0.5695 | 0.5589 | 683 |
| tin_hoc_tu_chon | 0.5306 | 0.5331 | 0.5319 | 634 |
| dia_ly | 0.5463 | 0.5447 | 0.5455 | 683 |
| giao_duc_kinh_te_va_phap_luat | 0.5632 | 0.6085 | 0.5850 | 659 |
| am_nhac_tu_chon | 0.5529 | 0.5149 | 0.5332 | 670 |
| mi_thuat_tu_chon | 0.5358 | 0.5676 | 0.5512 | 673 |

## Chọn tham số và giới hạn

Cấu hình được chọn theo Macro F1 trên Validation, dùng Exact Match và Micro F1 để phá hòa; không lựa chọn dựa trên Test.

| Cấu hình | Validation Macro F1 | Validation Micro F1 | Validation Exact Match |
|---|---:|---:|---:|
| `{'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 140, 'reg_alpha': 0.0, 'reg_lambda': 1.0}` | 0.5274 | 0.5283 | 0.0267 |
| `{'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 'reg_alpha': 0.0, 'reg_lambda': 2.0}` | 0.5230 | 0.5240 | 0.0307 |
| `{'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 180, 'reg_alpha': 0.1, 'reg_lambda': 1.0}` | 0.5136 | 0.5145 | 0.0233 |
| `{'max_depth': 2, 'learning_rate': 0.05, 'n_estimators': 240, 'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 2.0}` | 0.5282 | 0.5293 | 0.0333 |
| `{'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 240, 'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 2.0}` | 0.5160 | 0.5170 | 0.0240 |
| `{'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 320, 'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 2.0}` | 0.5180 | 0.5187 | 0.0240 |

Mỗi nhãn có ít nhất 30 mẫu ở cả hai lớp trong Test; không phát hiện nhãn hiếm theo ngưỡng 30 mẫu.

Mô hình dự đoán lựa chọn quan sát được trong dữ liệu, không đánh giá mức độ phù hợp hoặc năng lực học tập của học sinh. Hiệu năng cần được đối chiếu với baseline và dữ liệu mới trước khi sử dụng thực tế.
