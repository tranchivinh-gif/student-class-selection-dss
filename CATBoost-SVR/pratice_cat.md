# Tìm hiểu CatBoost và thực hành với dự án

> Tài liệu này giải thích CatBoost ở mức nền tảng, sau đó hướng dẫn chạy và tự thay đổi một vài cấu hình trong notebook của dự án. Bài thực hành dự báo **điểm Toán mô phỏng**; đây không phải điểm thi thật hay mô hình đã kiểm chứng trên học sinh thật.

## 1. Trước hết: mô hình đang học việc gì?

Trong bài toán hiện tại, mỗi dòng dữ liệu là một hồ sơ học sinh mô phỏng:

- **Đầu vào `X`:** điểm học tập các môn/lớp, giờ tự học, chuyên cần và điểm thi thử Toán.
- **Đầu ra `y`:** `diem_thi_toan_mo_phong`, một con số từ 0 đến 10 được tạo theo quy tắc mô phỏng.
- **Mục tiêu học:** tìm một hàm dự đoán `f(X)` sao cho điểm dự đoán gần với `y` nhất trên dữ liệu chưa dùng để huấn luyện.

Đây là **hồi quy** vì đầu ra là số liên tục. Vì vậy dự án dùng `CatBoostRegressor`. `CatBoostClassifier` dành cho đầu ra dạng lớp, ví dụ `yes/no`; đó là một bài toán khác.

## 2. Cây quyết định: viên gạch cơ bản

Một cây quyết định đặt các câu hỏi lần lượt về dữ liệu. Ví dụ:

```text
Điểm Toán lớp 9 <= 6.5?
├── Có: xét tiếp điểm Toán lớp 8 <= 5.5?
└── Không: đi đến một nhóm hồ sơ khác
```

Mỗi lần tách dữ liệu thành hai nhánh được gọi là một **split**. Với hồi quy, mô hình chọn các split giúp các điểm trong từng nhóm có nhãn gần nhau hơn, theo hàm mất mát đã chọn. Mỗi lá cây đưa ra một giá trị đóng góp cho dự đoán.

Một cây nông dễ hiểu nhưng có thể bỏ sót quan hệ phức tạp. Cây quá sâu có thể ghi nhớ cả nhiễu trong train. CatBoost dùng nhiều cây nên mỗi cây riêng lẻ không phải toàn bộ câu trả lời.

## 3. Boosting: vì sao có nhiều cây?

CatBoost thuộc nhóm **gradient boosting trên cây**. Hãy hình dung quy trình như sau:

1. Bắt đầu bằng một dự đoán chung, thường gần mức trung bình của nhãn.
2. So sánh dự đoán với điểm thật để tính lỗi theo hàm mất mát.
3. Huấn luyện cây tiếp theo để tìm các mẫu lỗi mà mô hình hiện tại còn mắc.
4. Cộng một phần đóng góp của cây mới vào tổng dự đoán.
5. Lặp lại cho đến khi đủ số vòng hoặc validation không còn cải thiện.

Ở mức trực giác, nếu một hồ sơ có điểm thật là 7 nhưng dự đoán hiện tại là 6, mô hình còn thiếu khoảng 1 điểm. Cây sau có thể học một quy tắc giúp sửa một phần lỗi đó. Trong CatBoost, việc sửa lỗi được tính bằng gradient của hàm mất mát; không phải lúc nào cũng chỉ lấy phần dư thô rồi cộng nguyên vẹn.

Có thể hình dung tổng dự đoán là:

```text
dự đoán cuối = dự đoán ban đầu + đóng góp cây 1 + đóng góp cây 2 + ...
```

Đây là điểm khác quan trọng với **Random Forest**: Random Forest thường huấn luyện nhiều cây tương đối độc lập trên các mẫu/đặc trưng ngẫu nhiên rồi lấy trung bình; boosting xây cây tuần tự, cây sau phụ thuộc vào sai số hiện tại của tổ hợp cây trước.

## 4. CatBoost có gì đặc trưng?

Tên CatBoost là viết tắt của **Categorical Boosting**. Hai ý tưởng nổi bật của thuật toán là xử lý đặc trưng phân loại và Ordered Boosting.

### 4.1. Xử lý biến phân loại

Một biến phân loại có các giá trị như `loại_trường = công_lập/tư_thục` hoặc `khu_vực = nội_thành/ngoại_thành`. Nếu mã hóa tùy tiện các nhãn này thành 0, 1, 2 thì mô hình có thể hiểu nhầm rằng chúng có thứ tự và khoảng cách số học.

CatBoost hỗ trợ khai báo cột phân loại và tự tạo biểu diễn số phù hợp, gồm các thống kê dựa trên nhãn và một số tổ hợp đặc trưng. Cách tính có sắp thứ tự được thiết kế để giảm việc một hồ sơ vô tình dùng chính nhãn của nó trong thống kê tạo đặc trưng. Khi dùng CatBoost, vẫn phải khai báo đúng cột phân loại; thư viện không thể tự biết ý nghĩa của từng cột.

**Trong notebook dự án hiện tại, 55 đầu vào đều là số.** Do đó phần xử lý biến phân loại không được minh họa ở bài thực hành này. CatBoost không bắt buộc phải có biến phân loại mới chạy được.

### 4.2. Ordered Boosting

Trong boosting thông thường, dự đoán dùng để tính bước sửa lỗi có thể bị lệch do mô hình đã được huấn luyện trên cùng các nhãn. CatBoost dùng các hoán vị và dự đoán theo phần đứng trước trong thứ tự để giảm dạng lệch này. Cơ chế này giúp giảm một nguồn prediction shift, nhưng **không tự động ngăn mọi loại rò rỉ dữ liệu**.

Vẫn cần tự chia train/validation/test đúng cách, không đưa nhãn vào đặc trưng, và không để các hồ sơ có quan hệ gần nhau xuất hiện ở cả train lẫn test. Dự án có nhóm `nhom_ho_so_goc` gồm các hồ sơ phụ thuộc nên phải giữ nhóm trong cùng một phân vùng.

### 4.3. Cây đối xứng mặc định

CatBoost mặc định dùng cây đối xứng (symmetric/oblivious tree): tại mỗi tầng, cùng một điều kiện chia được áp dụng cho các nút ở tầng đó. Cấu trúc này giúp đường đi trong cây gọn và dự đoán hiệu quả, nhưng cũng là lựa chọn thiết kế riêng của CatBoost. Không nên hiểu rằng mọi cây CatBoost đều là cây bất đối xứng như một số thư viện boosting khác.

## 5. Những tham số chính cần hiểu

Các giá trị dưới đây là cấu hình trong `analys/catboost.ipynb` và `train_models.py`, không phải kết luận rằng đã tối ưu:

| Tham số | Giá trị dự án | Ý nghĩa |
|---|---:|---|
| `iterations` | 1.000 tối đa | Số vòng boosting/số cây tối đa. Early stopping có thể làm mô hình dừng sớm hơn. |
| `depth` | 8 | Độ sâu cây; tăng độ sâu làm cây biểu diễn quan hệ phức tạp hơn nhưng có thể tăng khớp quá mức và thời gian. |
| `learning_rate` | 0,05 | Mức đóng góp mỗi vòng. Giá trị nhỏ thường cần nhiều cây hơn. |
| `loss_function` | `RMSE` | Đại lượng mô hình tối ưu trong huấn luyện; RMSE phạt lỗi lớn mạnh hơn MAE. |
| `eval_metric` | `RMSE` | Chỉ số theo dõi trên validation để chọn vòng dừng. |
| `early_stopping_rounds` | 100 | Dừng nếu validation không cải thiện sau số vòng quy định. |
| `random_seed` | 42 | Giúp lần chạy có thể tái lập trong cùng điều kiện/phần mềm. |

Một thay đổi tham số có thể làm kết quả khác. Khi so sánh, chỉ đổi một nhóm yếu tố mỗi lần và ghi lại cấu hình, seed, số cây thực tế và metric validation/test.

## 6. MAE, RMSE, R² dùng để đọc kết quả

- **MAE:** sai số tuyệt đối trung bình, cùng đơn vị điểm. Nếu MAE = 0,755 thì dự đoán lệch trung bình khoảng 0,755 điểm trong tập đánh giá. Không gọi đó là “độ chính xác 92,45%”.
- **RMSE:** căn trung bình bình phương sai số; nhạy với những lỗi lớn hơn MAE.
- **R²:** so mức sai số của mô hình với một mốc dự đoán bằng trung bình nhãn. R² = 1 là khớp hoàn hảo theo định nghĩa, 0 tương đương mức mốc trung bình, có thể âm nếu mô hình tệ hơn mốc đó.

Luôn so với một **baseline** đơn giản. Notebook dùng `toan_lop9` làm mốc: nếu mô hình học máy không cải thiện được mốc dễ hiểu này, độ phức tạp thêm vào có thể chưa đem lại giá trị.

## 7. Chạy bài thực hành trong dự án

### Bước 1: mở đúng thư mục và chuẩn bị thư viện

Mở terminal tại thư mục `CATBoost-SVR`:

```powershell
cd D:\Nam4\MLAI\student-class-selection-dss\CATBoost-SVR
python --version
python -m pip install -r requirements.txt
```

Notebook cần môi trường có Jupyter hỗ trợ. Nếu dùng Jupyter trong VS Code, cài extension Jupyter và chọn đúng Python interpreter. Nếu thiếu Jupyter trong môi trường Python hiện tại, cài thêm:

```powershell
python -m pip install jupyter
```

### Bước 2: mở notebook và chạy lần lượt

Mở `analys/catboost.ipynb` trong VS Code hoặc Jupyter, chọn kernel Python vừa cài, rồi chọn **Run All**. Notebook tự tìm `dulieuhocsinh_50000_mo_phong.csv` trong thư mục dự án hoặc thư mục cha.

Không dùng riêng `dulieuhocsinh_10000.csv` cho bài toán điểm thi: tệp đó không có cột điểm thi đích. Bài thực hành này dùng nhãn `diem_thi_toan_mo_phong` trong bộ 50.000 dòng.

### Bước 3: đọc từng khối kết quả

1. **Thông tin dữ liệu:** kiểm tra 50.000 dòng và các phân vùng train 30.000, validation 10.000, test 10.000.
2. **Kiểm tra nhóm:** ba số nhóm trùng phải bằng 0. Đây là kiểm tra chống rò rỉ giữa hồ sơ gốc và các biến thể của nó.
3. **Tạo đầu vào và nhãn:** notebook chọn 55 đặc trưng, loại mã định danh/nhóm và cột tuyển sinh tương lai; nhãn là điểm Toán mô phỏng.
4. **Huấn luyện:** CatBoost học trên train. Validation dùng theo dõi RMSE và early stopping; test chưa được dùng để dừng vòng lặp.
5. **Đánh giá cuối:** so CatBoost với `toan_lop9` trên cùng validation/test. Chỉ số thấp hơn tốt hơn với MAE/RMSE; cao hơn tốt hơn với R².
6. **Feature importance:** xem đặc trưng nào mô hình sử dụng nhiều. Đây là mức đóng góp theo mô hình, không chứng minh quan hệ nhân quả.

Đầu ra đã lưu trong notebook hiện có: 256 cây (best iteration 255); trên test MAE 0,755, RMSE 0,951, R² 0,866. Mốc `toan_lop9` có MAE 1,065, RMSE 1,352, R² 0,730. Khi tự chạy lại, giá trị có thể thay đổi nếu khác phiên bản thư viện, dữ liệu hoặc cấu hình.

## 8. Tự tay thay đổi một tham số

Trong cell tạo `CatBoostRegressor`, thử đổi `depth` từ 8 thành 6, chạy lại các cell từ lúc tạo và huấn luyện mô hình. Sau đó ghi **validation MAE/RMSE/R²**, số cây và best iteration. Tiếp theo có thể quay về depth 8 và thử learning rate 0,03, nhưng mỗi lần chỉ thay một tham số.

| Lần chạy | `depth` | `learning_rate` | `iterations` tối đa | Best iteration | Validation MAE | Validation RMSE | Ghi chú |
|---|---:|---:|---:|---:|---:|---:|---|
| Cấu hình gốc | 8 | 0,05 | 1.000 | 255* |  |  |  |
| Thử nghiệm 1 | 6 | 0,05 | 1.000 |  |  |  |  |
| Thử nghiệm 2 | 8 | 0,03 | 1.000 |  |  |  |  |

\* Với cấu hình huấn luyện notebook hiện tại; `best_iteration` được đếm từ 0 và số cây thực tế là best iteration cộng 1.

### Quy tắc để không “học thuộc” tập test

- Có thể dùng validation để chọn tham số, vì đây là vai trò của validation.
- Không xem metric test sau mỗi lần thử rồi chọn cấu hình có test tốt nhất; như vậy test đã trở thành một phần của quá trình tuning.
- Sau khi chốt cấu hình bằng validation, chạy đánh giá test cuối cùng một lần và ghi lại kết quả.
- Nếu thử nhiều seed, mô tả đó là phân tích độ ổn định; không coi mỗi seed là một tập test độc lập.

## 9. Thử dự đoán một hồ sơ

Sau khi notebook đã huấn luyện xong, trong một cell mới chạy:

```python
sample = X_test.iloc[[0]]
actual = y_test.iloc[0]
predicted = float(model.predict(sample)[0])

print("Điểm thi Toán mô phỏng:", round(float(actual), 2))
print("Dự đoán CatBoost:", round(predicted, 2))
print("Sai số tuyệt đối:", round(abs(float(actual) - predicted), 2))
```

`X_test.iloc[[0]]` giữ dữ liệu ở dạng bảng hai chiều mà CatBoost mong đợi. Dòng này chỉ minh họa một hồ sơ test; không dùng riêng một ví dụ để đánh giá chất lượng toàn mô hình.

## 10. Lưu mô hình khi thực hành

Notebook học tập hiện không cần ghi đè mô hình hệ thống. Nếu muốn lưu bản thử riêng:

```python
from pathlib import Path

practice_path = Path("models/practice_catboost_toan.cbm")
practice_path.parent.mkdir(parents=True, exist_ok=True)
model.save_model(str(practice_path), format="cbm")
print("Đã lưu:", practice_path.resolve())
```

Tên `practice_catboost_toan.cbm` giúp không ghi đè `models/catboost_toan.cbm` đang được ứng dụng sử dụng. Muốn nạp lại:

```python
from catboost import CatBoostRegressor

loaded_model = CatBoostRegressor()
loaded_model.load_model("models/practice_catboost_toan.cbm", format="cbm")
```

## 11. Bài tập tự kiểm tra

1. Giải thích vì sao `diem_thi_toan_mo_phong` là `y`, còn `toan_lop9` là một cột đầu vào và đồng thời được dùng làm baseline.
2. Nếu tăng `depth`, điều gì có thể xảy ra với độ phức tạp, sai số train và khả năng khái quát?
3. Vì sao validation được dùng khi huấn luyện nhưng test phải được giữ cho đánh giá cuối?
4. Feature importance 58,60% của `toan_lop9` cho phép kết luận điều gì, và không cho phép kết luận điều gì?
5. Vì sao điểm test tốt trên dữ liệu mô phỏng chưa đủ để khẳng định mô hình dự báo tốt cho học sinh thật?

## 12. Giới hạn cần nhớ trong đề tài

Điểm thi đích của bộ 50.000 dòng là nhãn mô phỏng, được tạo từ các đặc trưng có quan hệ với kết quả. Mô hình có thể học tốt cơ chế sinh nhãn mà vẫn chưa chắc hoạt động tốt trên dữ liệu thực. Do đó, nên viết “CatBoost đạt các chỉ số … trên tập test mô phỏng” thay vì “CatBoost dự đoán đúng …% cho học sinh”.

Các mô hình CatBoostRegressor/SVR hiện tại dự báo điểm Toán, Ngữ văn và Ngoại ngữ. Chúng không trực tiếp chọn bốn môn tự chọn phù hợp. Muốn nghiên cứu mục tiêu đó cần xác định nhãn/tiêu chí phù hợp và thu thập dữ liệu có ý nghĩa tương ứng.

## Tài liệu tham khảo

1. CatBoost, [CatBoost: unbiased boosting with categorical features](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html), NeurIPS 2018.
2. CatBoost documentation, [Categorical features](https://catboost.ai/docs/en/features/categorical-features).
3. CatBoost documentation, [Choosing the tree structure](https://catboost.ai/docs/en/concepts/algorithm-main-stages_choose-tree-structure).
4. CatBoost documentation, [Parameter tuning](https://catboost.ai/docs/en/concepts/parameter-tuning).
