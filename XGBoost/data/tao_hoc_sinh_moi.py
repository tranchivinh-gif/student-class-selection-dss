"""Tạo 10.000 học sinh giả lập độc lập để thử mô hình đa nhãn 4/9.

Chỉ sử dụng thư viện chuẩn; tuyệt đối không mở CSV dữ liệu hiện có.
Đặt script ở thư mục gốc dự án và chạy: python tao_hoc_sinh_moi.py

LƯU Ý: Đối chiếu hai cấu hình MON_HOC và NHAN_MON_TU_CHON với header
thực tế trước khi chạy. Không thể xác nhận tên cột gốc nếu không có header.
"""

import csv
import math
import random
from collections import Counter
from pathlib import Path


# ------------------- CẤU HÌNH -------------------
SEED = 20261021
SO_HOC_SINH = 10_000
SO_MON_CHON = 4
CAC_LOP = (6, 7, 8, 9)
DUONG_DAN = Path("XGBoost/data/hoc_sinh_moi.csv")

# Giữ đúng tên và thứ tự môn như header gốc của dự án.
MON_HOC = (
    "toan",
    "ngu_van",
    "ngoai_ngu",
    "khoa_hoc_tu_nhien",
    "lich_su_dia_ly",
    "giao_duc_cong_dan",
    "cong_nghe",
    "tin_hoc",
    "giao_duc_the_chat",
    "nghe_thuat",
)

# Khóa = tên cột nhãn; giá trị = môn THCS dùng để tính độ liên hệ điểm.
# Cần đối chiếu tên 9 khóa này với header thật (không đọc CSV).
NHAN_MON_TU_CHON = {
    "vat_ly": "khoa_hoc_tu_nhien",
    "hoa_hoc": "khoa_hoc_tu_nhien",
    "sinh_hoc": "khoa_hoc_tu_nhien",
    "cong_nghe": "cong_nghe",
    "tin_hoc_tu_chon": "tin_hoc",
    "dia_ly": "lich_su_dia_ly",
    "giao_duc_kinh_te_va_phap_luat": "giao_duc_cong_dan",
    "am_nhac_tu_chon": "nghe_thuat",
    "mi_thuat_tu_chon": "nghe_thuat",
}

# Các nhóm năng lực chung: giúp môn gần nhau có quan hệ nhẹ về điểm.
NHOM_MON = {
    "toan": "tu_nhien",
    "ngu_van": "xa_hoi",
    "ngoai_ngu": "ngon_ngu",
    "khoa_hoc_tu_nhien": "tu_nhien",
    "lich_su_dia_ly": "xa_hoi",
    "giao_duc_cong_dan": "xa_hoi",
    "cong_nghe": "ky_thuat",
    "tin_hoc": "ky_thuat",
    "giao_duc_the_chat": "nang_khieu",
    "nghe_thuat": "nang_khieu",
}

NHOM_NHAN = {
    "vat_ly": "tu_nhien",
    "hoa_hoc": "tu_nhien",
    "sinh_hoc": "tu_nhien",
    "cong_nghe": "ky_thuat",
    "tin_hoc_tu_chon": "ky_thuat",
    "dia_ly": "xa_hoi",
    "giao_duc_kinh_te_va_phap_luat": "xa_hoi",
    "am_nhac_tu_chon": "nang_khieu",
    "mi_thuat_tu_chon": "nang_khieu",
}


# ------------------- TẠO DỮ LIỆU -------------------
def gioi_han_diem(x):
    return round(max(0.0, min(10.0, x)), 2)


def tao_diem(rng):
    """Tạo điểm trước khi sinh nhãn: không có đường truyền nhãn -> điểm."""
    nang_luc_chung = rng.gauss(6.3, 1.15)
    nang_luc_nhom = {nhom: rng.gauss(0, 0.75) for nhom in sorted(set(NHOM_MON.values()))}
    nang_luc_mon = {mon: rng.gauss(0, 0.8) for mon in MON_HOC}
    tang_truong_chung = rng.gauss(0.05, 0.18)
    tang_truong_mon = {mon: rng.gauss(0, 0.12) for mon in MON_HOC}
    bien_dong_nam = {lop: rng.gauss(0, 0.28) for lop in CAC_LOP}

    diem = {}
    for mon in MON_HOC:
        for lop in CAC_LOP:
            diem[f"{mon}_lop{lop}"] = gioi_han_diem(
                nang_luc_chung
                + 0.65 * nang_luc_nhom[NHOM_MON[mon]]
                + nang_luc_mon[mon]
                + (lop - 6) * (tang_truong_chung + tang_truong_mon[mon])
                + bien_dong_nam[lop]
                + rng.gauss(0, 0.55)
            )
    return diem


def tao_nhan(rng, diem):
    """Chọn 4/9 bằng điểm + sở thích ẩn + nhiễu, không có luật cứng."""
    so_thich_nhom = {nhom: rng.gauss(0, 0.85) for nhom in sorted(set(NHOM_NHAN.values()))}
    muc_do_yeu_thich = {ten: rng.gauss(0, 0.9) for ten in NHAN_MON_TU_CHON}

    diem_uu_tien = []
    for ten_nhan, mon_lien_quan in NHAN_MON_TU_CHON.items():
        hoc_luc = (diem[f"{mon_lien_quan}_lop8"] + diem[f"{mon_lien_quan}_lop9"]) / 2
        uu_tien = (
            0.43 * (hoc_luc - 6.0)
            + 0.75 * so_thich_nhom[NHOM_NHAN[ten_nhan]]
            + 0.8 * muc_do_yeu_thich[ten_nhan]
            + rng.gauss(0, 1.05)
        )
        diem_uu_tien.append((uu_tien, ten_nhan))

    # Top-4 sau khi cộng nhiễu mới là nhãn, không hề tác động ngược vào điểm.
    duoc_chon = {ten for _, ten in sorted(diem_uu_tien, reverse=True)[:SO_MON_CHON]}
    return {ten: int(ten in duoc_chon) for ten in NHAN_MON_TU_CHON}


def tao_header():
    cot_diem = [f"{mon}_lop{lop}" for mon in MON_HOC for lop in CAC_LOP]
    return ["ma_hoc_sinh", "nam_tot_nghiep", *cot_diem, *NHAN_MON_TU_CHON]


def kiem_tra_cau_hinh():
    assert len(NHAN_MON_TU_CHON) == 9, "Phải có đúng 9 nhãn."
    assert len(MON_HOC) == len(set(MON_HOC)), "Trùng tên môn học."
    assert set(NHOM_MON) == set(MON_HOC), "Thiếu nhóm năng lực cho môn học."
    assert set(NHOM_NHAN) == set(NHAN_MON_TU_CHON), "Thiếu nhóm sở thích."
    assert set(NHAN_MON_TU_CHON.values()) <= set(MON_HOC), "Nhãn trỏ đến môn không tồn tại."
    assert len(tao_header()) == len(set(tao_header())), "Trùng tên cột."


def ghi_csv():
    kiem_tra_cau_hinh()
    rng = random.Random(SEED)
    duong_dan = DUONG_DAN
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    with duong_dan.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=tao_header(), extrasaction="raise")
        writer.writeheader()
        for i in range(1, SO_HOC_SINH + 1):
            # Mã tăng tuần tự bảo đảm duy nhất, không phải thông tin nhãn.
            dong = {
                "ma_hoc_sinh": f"HS{i:06d}",
                "nam_tot_nghiep": rng.choice((2025, 2026, 2027, 2028, 2029)),
            }
            diem = tao_diem(rng)
            dong.update(diem)
            dong.update(tao_nhan(rng, diem))
            writer.writerow(dong)
    return duong_dan


# ------------------- KIỂM TRA FILE VỪA GHI -------------------
def kiem_tra_file(duong_dan):
    header = tao_header()
    cot_diem = [f"{mon}_lop{lop}" for mon in MON_HOC for lop in CAC_LOP]
    cot_nhan = list(NHAN_MON_TU_CHON)
    ma_da_gap = set()
    phan_bo = Counter()
    so_dong = 0

    # Chỉ đọc lại tệp MỚI VỪA TẠO để hậu kiểm; không đọc hoc_sinh.csv cũ.
    with duong_dan.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == header, "Sai tên, thứ tự hoặc số lượng cột."

        for so_dong, dong in enumerate(reader, start=1):
            assert None not in dong, f"Dòng {so_dong}: có cột thừa."
            assert all(dong[c] is not None and dong[c].strip() for c in header), (
                f"Dòng {so_dong}: có ô trống."
            )
            ma = dong["ma_hoc_sinh"]
            assert ma not in ma_da_gap, f"Mã học sinh bị trùng: {ma}"
            ma_da_gap.add(ma)

            for cot in cot_diem:
                diem = float(dong[cot])
                assert math.isfinite(diem) and 0 <= diem <= 10, (
                    f"Dòng {so_dong}: điểm {cot} ngoài [0, 10]."
                )

            for cot in cot_nhan:
                assert dong[cot] in {"0", "1"}, f"Dòng {so_dong}: nhãn {cot} không phải 0/1."
            assert sum(int(dong[cot]) for cot in cot_nhan) == SO_MON_CHON, (
                f"Dòng {so_dong}: không chọn đúng 4 môn."
            )
            phan_bo.update(cot for cot in cot_nhan if dong[cot] == "1")

    assert so_dong == SO_HOC_SINH, f"Sai số dòng: {so_dong}"
    assert len(ma_da_gap) == SO_HOC_SINH, "Phải có 10.000 mã học sinh duy nhất."
    assert sum(phan_bo.values()) == SO_HOC_SINH * SO_MON_CHON
    print(f"Số bản ghi: {so_dong:,}")
    print(f"Mã duy nhất: {len(ma_da_gap):,}")
    print(f"Số cột: {len(header)} (điểm: {len(cot_diem)}, nhãn: {len(cot_nhan)})")
    print("Phân bố nhãn (số học sinh / tỉ lệ):")
    for cot in cot_nhan:
        print(f"  {cot}: {phan_bo[cot]:5,d} / {phan_bo[cot] / so_dong:6.2%}")
    print(f"Đường dẫn: {duong_dan.resolve()}")
    print("Kiểm tra: ĐẠT")


if __name__ == "__main__":
    duong_dan_moi = ghi_csv()
    kiem_tra_file(duong_dan_moi)
