"""Làm sạch dữ liệu đã kiểm tra mà không tự ý loại bỏ học sinh."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "dulieuhocsinh_10000.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step_02"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV = PROCESSED_DIR / "du_lieu_da_lam_sach.csv"
SCORE_COLUMN_PATTERN = re.compile(r"^.+_lop[6-9]$", re.IGNORECASE)


def identify_score_columns(data: pd.DataFrame) -> list[str]:
    """Nhận diện cột điểm theo hậu tố môn_lopN đã thấy trong CSV."""
    return [
        column
        for column in data.columns
        if SCORE_COLUMN_PATTERN.fullmatch(str(column).strip()) is not None
    ]


def identify_yes_no_columns(data: pd.DataFrame) -> list[str]:
    """Tìm cột chỉ gồm yes/no, không phân biệt hoa thường và khoảng trắng."""
    columns: list[str] = []
    for column in data.columns:
        values = data[column].dropna().astype(str).str.strip().str.casefold()
        unique_values = set(values.unique())
        if unique_values and unique_values.issubset({"yes", "no"}):
            columns.append(column)
    return columns


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy CSV gốc: {INPUT_CSV}\n"
            "Hãy đặt CSV vào thư mục data/raw/ trước khi chạy."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    rows_before = len(data)
    duplicate_rows_before = int(data.duplicated().sum())

    stripped_columns = [str(column).strip() for column in data.columns]
    if len(stripped_columns) != len(set(stripped_columns)):
        raise ValueError(
            "Sau khi bỏ khoảng trắng đầu/cuối, có tên cột bị trùng. "
            "Hãy kiểm tra tiêu đề CSV và sửa tên cột trước khi tiếp tục."
        )
    column_name_changes = [
        (old, new)
        for old, new in zip(data.columns, stripped_columns)
        if old != new
    ]
    data.columns = stripped_columns

    score_columns = identify_score_columns(data)
    label_columns = identify_yes_no_columns(data)
    if not score_columns:
        raise ValueError(
            "Không nhận diện được cột điểm theo quy ước tên *_lop6 đến *_lop9. "
            "Kiểm tra lại tên cột trong báo cáo bước 1."
        )
    if not label_columns:
        raise ValueError(
            "Không nhận diện được cột yes/no hợp lệ. "
            "Không thể chuẩn hóa nhãn lựa chọn."
        )

    score_report: list[dict[str, object]] = []
    for column in score_columns:
        original = data[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid_count = int((numeric.isna() & original.notna()).sum())
        out_of_range_count = int(((numeric < 0) | (numeric > 10)).sum())
        data[column] = numeric
        score_report.append(
            {
                "ten_cot_diem": column,
                "so_thieu": int(numeric.isna().sum()),
                "so_gia_tri_khong_phai_so": invalid_count,
                "so_diem_ngoai_0_10": out_of_range_count,
                "min_quan_sat": numeric.min(),
                "max_quan_sat": numeric.max(),
                "tu_dong_xoa_hoac_cat_diem": False,
            }
        )

    label_report: list[dict[str, object]] = []
    for column in label_columns:
        original = data[column]
        normalized = original.astype("string").str.strip().str.casefold()
        unexpected_mask = original.notna() & ~normalized.isin(["yes", "no"])
        unexpected_values = sorted(
            {str(value) for value in original[unexpected_mask].unique()}
        )
        if unexpected_values:
            raise ValueError(
                f"Cột nhãn {column!r} có giá trị khác yes/no: "
                f"{unexpected_values}. Không tự chuyển đổi các giá trị này."
            )

        data[column] = normalized
        counts = normalized.value_counts(dropna=True)
        label_report.append(
            {
                "ten_cot_nhan": column,
                "so_yes": int(counts.get("yes", 0)),
                "so_no": int(counts.get("no", 0)),
                "so_thieu": int(normalized.isna().sum()),
                "gia_tri_khac_yes_no": 0,
                "quy_tac_chuan_hoa": "yes/Yes -> yes; no/No -> no; bỏ khoảng trắng",
            }
        )

    data.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(score_report).to_csv(
        OUTPUT_DIR / "kiem_tra_cot_diem.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(label_report).to_csv(
        OUTPUT_DIR / "tom_tat_nhan_sau_chuan_hoa.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rows_after = len(data)
    report_lines = [
        "BÁO CÁO LÀM SẠCH DỮ LIỆU - BƯỚC 2",
        f"File gốc: {INPUT_CSV.relative_to(PROJECT_ROOT)}",
        f"File sau xử lý: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}",
        f"Số dòng trước: {rows_before}",
        f"Số dòng sau: {rows_after}",
        f"Số dòng trùng hoàn toàn trước xử lý: {duplicate_rows_before}",
        "Số dòng bị tự động xóa: 0",
        f"Số cột điểm được nhận diện: {len(score_columns)}",
        f"Số cột nhãn yes/no được nhận diện: {len(label_columns)}",
        f"Tổng ô thiếu sau xử lý: {int(data.isna().sum().sum())}",
        "",
        "TÊN CỘT ĐƯỢC BỎ KHOẢNG TRẮNG ĐẦU/CUỐI:",
        *(
            [f"- {old!r} -> {new!r}" for old, new in column_name_changes]
            or ["- Không có"]
        ),
        "",
        "KIỂM TRA ĐIỂM:",
        *[
            (
                f"- {item['ten_cot_diem']}: thiếu={item['so_thieu']}, "
                f"không phải số={item['so_gia_tri_khong_phai_so']}, "
                f"ngoài 0-10={item['so_diem_ngoai_0_10']}"
            )
            for item in score_report
        ],
        "",
        "KIỂM TRA VÀ CHUẨN HÓA NHÃN:",
        *[
            (
                f"- {item['ten_cot_nhan']}: yes={item['so_yes']}, "
                f"no={item['so_no']}, thiếu={item['so_thieu']}"
            )
            for item in label_report
        ],
        "",
        "GIẢI THÍCH:",
        "- Dữ liệu điểm được đổi sang kiểu số nếu có thể; điểm ngoài 0-10 được "
        "báo cáo, không tự xóa hoặc tự cắt về khoảng hợp lệ.",
        "- Nhãn yes/no được chuẩn hóa thành chữ thường; ô nhãn bị thiếu được giữ "
        "thiếu, không đoán lựa chọn của học sinh.",
        "- Điểm thiếu được giữ nguyên ở đây. Nếu có, bước huấn luyện phải học "
        "giá trị điền thiếu từ tập huấn luyện trong Pipeline, không học từ tập kiểm tra.",
        "- Bản CSV gốc trong data/raw/ không bị thay đổi.",
    ]
    (OUTPUT_DIR / "bao_cao_lam_sach.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print("\n".join(report_lines))
    print("\nĐã lưu:")
    print(f"- {OUTPUT_CSV}")
    print(f"- {OUTPUT_DIR / 'bao_cao_lam_sach.txt'}")
    print(f"- {OUTPUT_DIR / 'kiem_tra_cot_diem.csv'}")
    print(f"- {OUTPUT_DIR / 'tom_tat_nhan_sau_chuan_hoa.csv'}")


if __name__ == "__main__":
    main()
