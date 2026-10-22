"""Data validation and cleaning for the student elective-subject project."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_SOURCE = DATA_DIR / "hoc_sinh.csv"
CLEAN_DATA_PATH = DATA_DIR / "hoc_sinh_clean.csv"
REPORTS_DIR = PROJECT_DIR / "reports"
DATA_REPORT_PATH = REPORTS_DIR / "data_report.md"
STUDENT_ID_COLUMN = "ma_hoc_sinh"
GRADE_COLUMN_PATTERN = re.compile(r".+_lop[6-9]$", re.IGNORECASE)


def prepare_dataframe(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, object]]:
    """Validate a source frame and return cleaned data, features, labels, and audit."""
    frame = source.copy()
    if frame.empty:
        raise ValueError("Dữ liệu nguồn không có dòng học sinh.")
    original_columns = list(frame.columns)
    normalized_columns = [str(column).strip() for column in original_columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError("Tên cột bị trùng sau khi loại bỏ khoảng trắng.")
    frame.columns = normalized_columns

    if STUDENT_ID_COLUMN not in frame.columns:
        raise ValueError(f"Thiếu cột định danh bắt buộc: {STUDENT_ID_COLUMN}")

    feature_columns = [
        column for column in frame.columns
        if GRADE_COLUMN_PATTERN.fullmatch(column)
    ]
    if not feature_columns:
        raise ValueError("Không tìm thấy cột điểm lớp 6-9 theo mẫu *_lop[6-9].")

    label_columns = [
        column for column in frame.columns
        if column != STUDENT_ID_COLUMN and column not in feature_columns
    ]
    if not label_columns:
        raise ValueError("Không tìm thấy cột nhãn môn tự chọn.")

    missing_by_column = frame.isna().sum()
    blank_by_column = frame.astype("string").apply(
        lambda column: column.str.strip().eq("").sum()
    )
    missing_by_column = missing_by_column.add(blank_by_column, fill_value=0).astype(int)
    duplicate_rows = int(frame.duplicated().sum())
    duplicate_ids = int(frame[STUDENT_ID_COLUMN].duplicated().sum())

    invalid_scores: dict[str, int] = {}
    for column in feature_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.isna() | ~values.between(0, 10)
        if invalid.any():
            invalid_scores[column] = int(invalid.sum())
        frame[column] = values

    invalid_labels: dict[str, list[str]] = {}
    for column in label_columns:
        normalized = frame[column].astype("string").str.strip().str.casefold()
        unexpected = sorted(set(normalized.dropna()) - {"yes", "no", "1", "0"})
        if normalized.isna().any() or unexpected:
            invalid_labels[column] = unexpected or ["giá trị thiếu"]
        frame[column] = normalized.map({"yes": 1, "no": 0, "1": 1, "0": 0}).astype("Int64")

    audit: dict[str, object] = {
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "source_columns": original_columns,
        "feature_columns": feature_columns,
        "label_columns": label_columns,
        "missing_by_column": {
            column: int(count)
            for column, count in missing_by_column.items()
            if count > 0
        },
        "duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "invalid_scores": invalid_scores,
        "invalid_labels": invalid_labels,
        "estimated_columns": [],
    }

    problems = []
    if audit["missing_by_column"]:
        problems.append(f"có giá trị thiếu: {audit['missing_by_column']}")
    if duplicate_rows:
        problems.append(f"có {duplicate_rows} dòng trùng lặp")
    if duplicate_ids:
        problems.append(f"có {duplicate_ids} mã học sinh trùng lặp")
    if invalid_scores:
        problems.append(f"điểm thiếu/ngoài [0, 10]: {invalid_scores}")
    if invalid_labels:
        problems.append(f"nhãn không hợp lệ: {invalid_labels}")
    if problems:
        raise ValueError("Dữ liệu cần được xem xét thủ công; không tự động xóa hay sửa: "
                         + "; ".join(problems))

    frame = frame[
        [STUDENT_ID_COLUMN, *feature_columns, *label_columns]
    ].copy()
    frame[label_columns] = frame[label_columns].astype("int8")
    audit["label_counts"] = {
        column: {
            "0": int((frame[column] == 0).sum()),
            "1": int((frame[column] == 1).sum()),
        }
        for column in label_columns
    }
    return frame, feature_columns, label_columns, audit


def _render_report(
    audit: dict[str, object],
    source_path: Path,
    source_hash: str,
) -> str:
    lines = [
        "# Báo cáo chuẩn bị dữ liệu",
        "",
        f"- Tệp nguồn: `{source_path.name}`",
        f"- SHA-256 tệp nguồn: `{source_hash}`",
        f"- Số dòng: {audit['row_count']:,}",
        f"- Số cột nguồn: {audit['column_count']}",
        f"- Số đặc trưng điểm lớp 6-9: {len(audit['feature_columns'])}",
        f"- Số nhãn môn tự chọn: {len(audit['label_columns'])}",
        f"- Dòng trùng lặp: {audit['duplicate_rows']}",
        f"- Mã học sinh trùng: {audit['duplicate_ids']}",
        "",
        "## Kiểm tra chất lượng",
        "",
        "- Giá trị thiếu: không có."
        if not audit["missing_by_column"]
        else f"- Giá trị thiếu: `{audit['missing_by_column']}`",
        "- Điểm ngoài khoảng [0, 10]: không có."
        if not audit["invalid_scores"]
        else f"- Điểm bất thường: `{audit['invalid_scores']}`",
        "- Nhãn không thuộc Yes/No hoặc 0/1: không có."
        if not audit["invalid_labels"]
        else f"- Nhãn không hợp lệ: `{audit['invalid_labels']}`",
        "",
        "## Thay đổi và giới hạn đầu vào",
        "",
        "- Chỉ loại bỏ khoảng trắng đầu/cuối tên cột và nhãn; nhãn `Yes/No` "
        "được mã hóa thành `1/0`; nhãn `0/1` được giữ nguyên.",
        "- Không xóa dòng hoặc sửa điểm/nhãn gốc.",
        "- `ma_hoc_sinh` được giữ trong CSV đã làm sạch để truy vết nhưng sẽ "
        "không được đưa vào đặc trưng mô hình.",
        "- Không tạo đặc trưng từ nhãn hoặc từ thống kê toàn tập.",
    ]
    lines.extend(
        [
            "",
            "## Phân bố nhãn",
            "",
            "| Môn tự chọn | No (0) | Yes (1) |",
            "|---|---:|---:|",
        ]
    )
    for label, counts in audit["label_counts"].items():
        lines.append(f"| {label} | {counts['0']:,} | {counts['1']:,} |")
    return "\n".join(lines) + "\n"


def prepare_data(
    source_path: Path = DEFAULT_SOURCE,
    output_path: Path = CLEAN_DATA_PATH,
    report_path: Path = DATA_REPORT_PATH,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, object]]:
    """Read, validate, clean, and report without modifying the source file."""
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu nguồn: {source_path}")

    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source = pd.read_csv(source_path, encoding="utf-8-sig")
    cleaned, features, labels, audit = prepare_dataframe(source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, encoding="utf-8")
    report_path.write_text(
        _render_report(audit, source_path, source_hash),
        encoding="utf-8",
    )
    return cleaned, features, labels, audit


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=CLEAN_DATA_PATH)
    parser.add_argument("--report", type=Path, default=DATA_REPORT_PATH)
    args = parser.parse_args()
    cleaned, features, labels, audit = prepare_data(args.input, args.output, args.report)
    print(
        f"Đã chuẩn bị {len(cleaned):,} dòng; "
        f"{len(features)} đặc trưng và {len(labels)} nhãn."
    )
    print(f"Báo cáo: {args.report}")


if __name__ == "__main__":
    main()
