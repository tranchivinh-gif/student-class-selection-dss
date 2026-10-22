"""Kiểm tra cấu trúc và chất lượng dữ liệu CSV đầu vào."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "dulieuhocsinh_10000.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step_01"
ID_COLUMN_CANDIDATES = {"ma_hoc_sinh", "student_id", "studentid", "id"}
SCORE_COLUMN_PATTERN = re.compile(r"^(?P<subject>.+)_lop(?P<grade>[6-9])$", re.IGNORECASE)


def identify_score_columns(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Nhận diện cột điểm theo quy ước tên môn_lopN và kiểm tra khả năng đổi sang số."""
    detected: list[str] = []
    invalid_columns: list[str] = []

    for column in data.columns:
        if SCORE_COLUMN_PATTERN.fullmatch(str(column).strip()) is None:
            continue

        detected.append(column)
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid_count = int((converted.isna() & data[column].notna()).sum())
        if invalid_count:
            invalid_columns.append(column)

    return detected, invalid_columns


def identify_yes_no_columns(data: pd.DataFrame) -> list[str]:
    """Tìm cột chỉ chứa yes/no (không phân biệt hoa thường và khoảng trắng)."""
    result: list[str] = []
    for column in data.columns:
        normalized = data[column].dropna().astype(str).str.strip().str.casefold()
        unique_values = set(normalized.unique())
        if unique_values and unique_values.issubset({"yes", "no"}):
            result.append(column)
    return result


def format_value_counts(series: pd.Series) -> str:
    """Định dạng các giá trị duy nhất và tần số để đưa vào báo cáo."""
    counts = series.value_counts(dropna=False)
    if counts.empty:
        return "(không có giá trị)"
    return "; ".join(f"{value!r}: {count}" for value, count in counts.items())


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy CSV đầu vào: {INPUT_CSV}\n"
            "Hãy đặt file CSV gốc vào thư mục data/raw/."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            "Không đọc được CSV với mã hóa UTF-8. Hãy kiểm tra mã hóa của file "
            "trước khi tiếp tục; chương trình không tự đổi mã hóa để tránh sai tên cột."
        ) from error

    score_columns, invalid_score_columns = identify_score_columns(data)
    yes_no_columns = identify_yes_no_columns(data)
    id_columns = [
        column
        for column in data.columns
        if str(column).strip().casefold() in ID_COLUMN_CANDIDATES
    ]
    missing_cells = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())

    column_rows: list[dict[str, object]] = []
    for column in data.columns:
        column_rows.append(
            {
                "ten_cot": column,
                "kieu_du_lieu": str(data[column].dtype),
                "so_gia_tri_thieu": int(data[column].isna().sum()),
                "so_gia_tri_khac_nhau": int(data[column].nunique(dropna=True)),
                "la_cot_diem": column in score_columns,
                "la_cot_yes_no": column in yes_no_columns,
                "la_cot_ma_hoc_sinh": column in id_columns,
            }
        )

    column_report = pd.DataFrame(column_rows)
    column_report.to_csv(
        OUTPUT_DIR / "thong_tin_cot.csv", index=False, encoding="utf-8-sig"
    )

    label_rows: list[dict[str, object]] = []
    for column in yes_no_columns:
        normalized = data[column].dropna().astype(str).str.strip().str.casefold()
        counts = normalized.value_counts()
        label_rows.append(
            {
                "ten_cot_nhan": column,
                "gia_tri_goc_va_so_luong": format_value_counts(data[column]),
                "so_yes": int(counts.get("yes", 0)),
                "so_no": int(counts.get("no", 0)),
                "so_thieu": int(data[column].isna().sum()),
                "co_du_hai_lop": bool(counts.get("yes", 0) > 0 and counts.get("no", 0) > 0),
            }
        )

    pd.DataFrame(
        label_rows,
        columns=[
            "ten_cot_nhan",
            "gia_tri_goc_va_so_luong",
            "so_yes",
            "so_no",
            "so_thieu",
            "co_du_hai_lop",
        ],
    ).to_csv(
        OUTPUT_DIR / "tom_tat_nhan_yes_no.csv",
        index=False,
        encoding="utf-8-sig",
    )

    score_lines: list[str] = []
    for column in score_columns:
        numeric_values = pd.to_numeric(data[column], errors="coerce")
        invalid_count = int((numeric_values.isna() & data[column].notna()).sum())
        outside_range = int(((numeric_values < 0) | (numeric_values > 10)).sum())
        score_lines.append(
            f"- {column}: kiểu={data[column].dtype}, "
            f"min={numeric_values.min()}, max={numeric_values.max()}, "
            f"không đổi được sang số={invalid_count}, ngoài khoảng 0-10={outside_range}"
        )

    label_lines: list[str] = []
    for column in yes_no_columns:
        label_lines.append(f"- {column}: {format_value_counts(data[column])}")

    structurally_ready = bool(score_columns and yes_no_columns)
    conclusion = (
        "ĐỦ ĐIỀU KIỆN VỀ CẤU TRÚC để xem xét bước 2: đã tìm thấy cột điểm "
        "và cột yes/no."
        if structurally_ready
        else "CHƯA ĐỦ ĐIỀU KIỆN: cần xem lại quy ước tên cột điểm và/hoặc nhãn yes/no."
    )
    if invalid_score_columns:
        conclusion += (
            "\nCẦN XEM XÉT: một số cột điểm có giá trị không phải số: "
            + ", ".join(invalid_score_columns)
            + "."
        )
    if not yes_no_columns:
        conclusion += "\nKhông tìm thấy cột yes/no hợp lệ; không thể huấn luyện nhãn lựa chọn."

    report_lines = [
        "BÁO CÁO KIỂM TRA DỮ LIỆU",
        f"File: {INPUT_CSV.relative_to(PROJECT_ROOT)}",
        f"Kích thước: {data.shape[0]} dòng x {data.shape[1]} cột",
        f"Tổng ô thiếu: {missing_cells}",
        f"Số dòng trùng hoàn toàn: {duplicate_rows}",
        f"Số cột nhận diện là mã học sinh: {len(id_columns)}"
        + (f" ({', '.join(id_columns)})" if id_columns else ""),
        "",
        "DANH SÁCH TÊN CỘT (giữ nguyên như trong CSV):",
        *[f"{index}. {column}" for index, column in enumerate(data.columns, start=1)],
        "",
        f"SỐ CỘT ĐIỂM NHẬN DIỆN: {len(score_columns)}",
        *score_lines,
        "",
        f"SỐ CỘT YES/NO NHẬN DIỆN: {len(yes_no_columns)}",
        *label_lines,
        "",
        "KẾT LUẬN:",
        conclusion,
        (
            "Lưu ý: phát hiện yes/no chỉ xác nhận định dạng hai lớp, chưa chứng minh "
            "các giá trị này thật sự ghi nhận lựa chọn môn của học sinh."
        ),
    ]

    report_path = OUTPUT_DIR / "bao_cao_tong_quan.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print("\nĐã lưu báo cáo:")
    print(f"- {report_path}")
    print(f"- {OUTPUT_DIR / 'thong_tin_cot.csv'}")
    print(f"- {OUTPUT_DIR / 'tom_tat_nhan_yes_no.csv'}")


if __name__ == "__main__":
    main()
