# Logistic Regression tư vấn tổ hợp môn

Dự án dùng dữ liệu CSV đã được kiểm tra để huấn luyện mô hình Logistic
Regression nhị phân riêng cho từng nhãn yes/no, chia tập **80% huấn luyện /
20% kiểm tra**, đánh giá mô hình và thử dự đoán qua giao diện web.

> Mô hình chỉ ước lượng xu hướng các nhãn yes/no trong dữ liệu. Không xem kết quả
> là mức độ phù hợp môn học, năng lực, điểm thi hay xác suất đỗ. Các tổ hợp môn
> chính thức chưa được cấu hình vì cần danh sách chính xác từ người dùng.

## Cấu trúc

```text
Logistic_Regression_TuVanToHop/
├── data/
│   ├── raw/dulieuhocsinh_10000.csv
│   └── processed/du_lieu_da_lam_sach.csv
├── models/
├── outputs/step_01/ ... outputs/step_06/
├── src/
│   ├── common.py
│   ├── subject_combinations.py
│   ├── step_01_inspect_data.py
│   ├── step_02_clean_data.py
│   ├── step_03_prepare_dataset.py
│   ├── step_04_train_logistic_regression.py
│   ├── step_05_evaluate_model.py
│   ├── step_06_predict_top4_subjects.py
│   ├── step_07_predict_new_student.py
│   ├── step_08_train_kmeans.py
│   ├── step_09_evaluate_kmeans.py
│   ├── app.py
│   └── app_kmeans.py
├── README.md
├── Chay_giao_dien.bat
├── Chay_giao_dien_KMeans.bat
└── requirements.txt
```

## Cài đặt và chạy tuần tự

Mở thư mục `Logistic_Regression_TuVanToHop` trong VS Code. Mở Terminal tại
thư mục dự án và cài thư viện:

```powershell
python -m pip install -r requirements.txt
```

Chạy từng bước theo thứ tự:

```powershell
python src\step_01_inspect_data.py
python src\step_02_clean_data.py
python src\step_03_prepare_dataset.py
python src\step_04_train_logistic_regression.py
python src\step_05_evaluate_model.py
python src\step_06_predict_top4_subjects.py
```

Đọc báo cáo đầu ra sau mỗi bước. Bước 6 chọn một học sinh từ tập kiểm tra để
minh họa; chương trình không đưa mã học sinh vào đặc trưng.

### Ý nghĩa các bước

1. **Kiểm tra dữ liệu:** thống kê tên cột, kiểu dữ liệu, thiếu/trùng, điểm và nhãn.
2. **Làm sạch:** chuẩn hóa yes/no và khoảng trắng tên cột; không tự xóa dòng,
   không đoán nhãn hoặc cắt điểm bất thường.
3. **Chuẩn bị:** `X` là các cột điểm được nhận diện theo hậu tố `_lop6`–`_lop9`;
   `y` là các cột yes/no hợp lệ được mã hóa no=0, yes=1. Mã học sinh bị loại.
   Chia ngẫu nhiên bằng `random_state=42`, 80% train và 20% test.
4. **Huấn luyện:** một Logistic Regression nhị phân cho mỗi nhãn, gói chung
   trong `MultiOutputClassifier` và Pipeline. Median Imputer cùng StandardScaler
   chỉ được fit trên tập train. `class_weight` không bật vì nhãn trong CSV gần
   cân bằng; điều này không thay thế việc đánh giá.
5. **Đánh giá:** Accuracy, Precision, Recall, F1, ROC-AUC nếu đủ hai lớp và
   Confusion Matrix cho từng nhãn, chỉ trên test.
6. **Dự đoán minh họa:** sắp xếp điểm xác suất của một dòng X_test rồi lưu Top 4.
7. **Dự đoán học sinh mới:** nhập tuần tự các điểm trong Terminal.
8. **Giao diện:** nhập điểm trên trang web bằng Streamlit sau khi huấn luyện.

Không học thống kê tiền xử lý từ tập test: nếu cần điền điểm thiếu thì
`SimpleImputer` nằm trong Pipeline và chỉ gọi `fit` với tập huấn luyện. Tập test
chỉ dùng ở bước đánh giá.

## Giao diện thử nghiệm

Sau khi chạy bước 4, chạy giao diện:

```powershell
python -m streamlit run src\app.py
```

Streamlit sẽ mở giao diện trong trình duyệt. Điền đủ các cột điểm được hiển thị
trong từng nhóm môn, dùng điểm 0–10, sau đó nhấn **Dự đoán xu hướng lựa chọn**.
Kết quả gồm bảng xác suất ước lượng, nhãn dự đoán, Top 4 và nút tải CSV.

Để thử nhanh mà không nhập điểm, chọn một trong **Mẫu kiểm tra 01–10** ở danh
sách **Chọn cách thử**. Mỗi mẫu tự điền điểm từ tập test vào biểu mẫu; chọn
**Tự nhập điểm** để nhập hồ sơ khác. Mẫu có sẵn là các dòng test dùng minh họa,
không phải học sinh mới.

Nếu có lỗi `streamlit` chưa được cài, chạy lại lệnh cài đặt requirements. Dừng
giao diện bằng `Ctrl+C` trong Terminal.

## Nhánh K-Means để thử phân cụm

K-Means là mô hình **không giám sát**: chỉ nhận các cột điểm, không dùng nhãn
yes/no để huấn luyện và không thể trả lời học sinh có xu hướng chọn môn nào.
Nhánh này dùng lại tập chia ở bước 3 (80% train / 20% test).

1. Chọn số cụm `k` trong khoảng 2–8 bằng silhouette score tính trên một mẫu của
   tập train; sau đó fit imputer, scaler và K-Means trên toàn bộ train.
2. Đánh giá silhouette trên test và lưu mã cụm gán cho từng dòng test.
3. Mở giao diện K-Means bằng lệnh:

```powershell
python src\step_08_train_kmeans.py
python src\step_09_evaluate_kmeans.py
python -m streamlit run src\app_kmeans.py --server.port 8766
```

Hoặc nhấp đúp `Chay_giao_dien_KMeans.bat` sau khi đã chạy xong bước 8. Chọn một
trong 10 mẫu test hoặc tự nhập điểm. Giao diện hiển thị cụm gần nhất, khoảng
cách tới các tâm và hồ sơ điểm trung tâm. Mã cụm chỉ là nhãn tùy ý, không biểu
thị tốt/xấu, phù hợp/không phù hợp hay xác suất.

Kết quả nhánh K-Means được lưu trong `outputs/kmeans/`; Pipeline riêng lưu tại
`models/kmeans_pipeline.joblib`, metadata tại `models/kmeans_metadata.json`.
Không dùng Accuracy/Precision/Recall để đánh giá phân cụm khi chưa có nhãn cụm
đúng do chuyên gia cung cấp; silhouette cao cũng không tự chứng minh nhóm có ý
nghĩa giáo dục.

## Tệp kết quả quan trọng

- `outputs/step_01/`: báo cáo kiểm tra dữ liệu.
- `outputs/step_02/`: dữ liệu sạch và báo cáo làm sạch.
- `outputs/step_03/`: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`,
  phân bố nhãn và metadata chia tập.
- `outputs/step_04/thong_tin_huan_luyen.json`: thông tin cấu hình huấn luyện.
- `outputs/step_05/`: chỉ số và báo cáo đánh giá trên tập kiểm tra.
- `outputs/step_06/`: các bảng xác suất, Top 4 và đối chiếu tổ hợp.
- `models/logistic_regression_pipeline.joblib`: Pipeline huấn luyện.
- `models/metadata.json`: thứ tự và tên cột đặc trưng/nhãn đã dùng.

Các script dựa trên đường dẫn tương đối tới thư mục dự án, không phụ thuộc vào
đường dẫn ổ đĩa máy cá nhân.

## Cấu hình 5 tổ hợp chính thức

Chỉ sau khi có danh sách môn chính xác, mở `src/subject_combinations.py` và khai
báo từ điển `OFFICIAL_COMBINATIONS` theo mẫu có sẵn trong file. Tên môn trong
tổ hợp phải tương ứng với tên nhãn trong CSV. Khi chưa cấu hình, ứng dụng sẽ
thông báo chưa thể đối chiếu; không tự tạo tổ hợp mới hoặc tự chọn nguyện vọng.

## Lưu ý diễn giải và lỗi thường gặp

- Nếu chưa có `du_lieu_da_lam_sach.csv`, hãy chạy bước 2.
- Nếu thiếu tập train/test, hãy chạy bước 3 trước bước 4.
- Nếu chưa có mô hình, hãy chạy bước 4 trước bước 5–7 hoặc mở giao diện.
- Nếu bước 3 báo nhãn thiếu hoặc nhãn không phải yes/no, cần xử lý nguồn nhãn;
  không thay nhãn bằng dữ liệu giả.
- Nếu kết quả Accuracy cao nhưng một lớp hiếm hoặc Recall thấp, Accuracy không
  đủ để kết luận. Xem toàn bộ chỉ số từng nhãn và Confusion Matrix.
- Điểm xác suất Logistic Regression chưa được hiệu chỉnh. Xếp hạng không có nghĩa
  xác suất lựa chọn ngoài đời chính xác, không phải mức độ phù hợp với học sinh.
