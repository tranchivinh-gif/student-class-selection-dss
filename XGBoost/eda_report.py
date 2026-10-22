"""Describe the raw CSV before cleaning or training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from preprocessing import DEFAULT_SOURCE, GRADE_COLUMN_PATTERN, PROJECT_DIR, STUDENT_ID_COLUMN


REPORT_PATH = PROJECT_DIR / "reports" / "eda_report.md"


def create_eda_report(input_path: Path = DEFAULT_SOURCE, report_path: Path = REPORT_PATH) -> str:
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu nguồn: {input_path}")
    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    columns = [str(column).strip() for column in frame.columns]
    features = [column for column in columns if GRADE_COLUMN_PATTERN.fullmatch(column)]
    labels = [column for column in columns if column != STUDENT_ID_COLUMN and column not in features]
    missing = frame.isna().sum() + frame.astype("string").apply(
        lambda series: series.str.strip().eq("").sum()
    )
    lines = [
        "# Khám phá dữ liệu nguồn (EDA)", "",
        f"- Tệp: `{input_path.name}`", f"- Số học sinh: {len(frame):,}",
        f"- Số cột: {len(columns)}", f"- Cột điểm: {len(features)}",
        f"- Cột môn tự chọn: {len(labels)}",
        f"- Dòng trùng: {int(frame.duplicated().sum()):,}",
        f"- Mã học sinh trùng: {int(frame[STUDENT_ID_COLUMN].duplicated().sum()):,}"
        if STUDENT_ID_COLUMN in frame else "- Thiếu cột mã học sinh.",
        "", "## Ô trống theo cột", "",
    ]
    nonempty_missing = missing[missing > 0]
    lines += [f"- `{column}`: {int(count):,}" for column, count in nonempty_missing.items()] or ["Không có."]
    lines += ["", "## Phân bố số môn đã chọn", ""]
    if labels:
        normalized = frame[[original for original in frame if str(original).strip() in labels]].astype("string").apply(
            lambda series: series.str.strip().str.casefold().isin(["yes", "1"])
        )
        counts = normalized.sum(axis=1).value_counts().sort_index()
        lines += [f"- {int(subjects)} môn: {int(count):,} học sinh" for subjects, count in counts.items()]
        lines += ["", "Phân bố chỉ đếm các ô Yes/1; các giá trị khác cần được kiểm tra ở bước xử lý dữ liệu."]
    lines += ["", "Báo cáo này chỉ mô tả dữ liệu nguồn; chưa loại hay sửa dòng.", ""]
    report = "\n".join(lines)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    create_eda_report(args.input, args.report)
    print(f"Báo cáo EDA: {args.report}")


if __name__ == "__main__":
    main()
