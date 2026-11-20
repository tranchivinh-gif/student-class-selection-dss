# Bộ dữ liệu học sinh tổng hợp SSRC–2026

Đây là thư mục gốc của thiết kế dữ liệu tổng hợp độc lập được mô tả trong
`Bao_cao_thiet_ke_bo_sinh_du_lieu_SSRC_2026_cap_nhat.docx`.

Cấu hình bộ sinh và sáu tổ hợp nhãn C1–C6 đã được xác nhận. Kết quả benchmark
tổng hợp không phải bằng chứng về học sinh thật.

## Cấu trúc

- `configs/`: cấu hình bộ sinh và thí nghiệm.
- `src/`: điểm vào sinh dữ liệu, kiểm tra, huấn luyện, đánh giá, độ nhạy và SHAP.
- `data/seed_<seed>/`: nơi lưu riêng dữ liệu của từng seed.
- `results/csv/`, `results/tables/`, `results/figures/`: đầu ra thí nghiệm.
- `logs/runs/`: seed, cấu hình, phiên bản thư viện và thời gian chạy.

Mỗi bộ dữ liệu dự kiến có 10.000 bản ghi tổng hợp, 32 điểm cả năm (8 môn,
lớp 6–9) và nhãn C1–C6. Năm seed được giữ và đánh giá riêng.

Tên cột CSV dùng tiếng Việt: `ma_hoc_sinh`, các cột điểm có dạng
`diem_<mon>_ca_nam_lop_<lop>`, và nhãn lớp là `nhan_lop`. Không có cột học kỳ 1
hoặc học kỳ 2.

## Sinh và kiểm tra dữ liệu

Thiết kế C1–C6 đã được xác nhận; có thể sinh dữ liệu chính thức:

```powershell
python src\generate_data
python src\validate_data
```

Chọn một seed bằng `--seed 20261108`; lặp lại `--seed` để chọn nhiều seed.
Dùng `--force` chỉ khi cố ý thay thế CSV và manifest hiện có. Có thể kiểm thử
nhanh riêng bằng `--records 100 --output-root <thu-muc-tam>`, không làm thay đổi
quy mô thí nghiệm 10.000 bản ghi.

### Pilot thăm dò 1.000 bản ghi

Pilot mặc định dùng seed đầu tiên, được đánh dấu là thăm dò và lưu tách khỏi dữ
liệu chính thức:

```powershell
python src\generate_data.py --pilot
python src\validate_data.py --pilot
python src\eda.py
```

CSV pilot nằm trong `data\pilot\seed_<seed>\`, manifest trong
`logs\pilot_runs\`, còn bảng, biểu đồ và ghi chú diễn giải nằm trong
`results\pilot\seed_<seed>\`. Dùng `--seed 20261109` ở mỗi lệnh để chọn seed
khác. Pilot tối đa 1.000 bản ghi và không được dùng làm bằng chứng hiệu năng
mô hình cuối cùng hoặc bằng chứng về học sinh thật.
