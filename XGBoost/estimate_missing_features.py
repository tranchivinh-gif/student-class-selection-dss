"""Create a separate CSV copy with explicitly estimated literature scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from preprocessing import DATA_DIR


RAW_SOURCE = DATA_DIR / "olds" / "dulieuhocsinh_10000.csv"
DERIVED_SOURCE = DATA_DIR / "olds" / "dulieuhocsinh_10000_uoc_tinh.csv"
SOURCE_COLUMNS = ("ngu_van_lop6", "ngu_van_lop7")
ESTIMATED_COLUMNS = ("ngu_van_lop8", "ngu_van_lop9")


def estimate_literature_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Linearly project the grade-6-to-7 change and clamp estimates to [0, 10]."""
    if frame.empty:
        raise ValueError("Dữ liệu nguồn không có dòng học sinh.")
    if any(column in frame.columns for column in ESTIMATED_COLUMNS):
        raise ValueError(
            "Nguồn đã chứa ngu_van_lop8 hoặc ngu_van_lop9; "
            "không ghi đè hay thay thế dữ liệu có sẵn."
        )
    missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Thiếu cột điểm để ước tính: {missing}")

    grade6 = pd.to_numeric(frame[SOURCE_COLUMNS[0]], errors="coerce")
    grade7 = pd.to_numeric(frame[SOURCE_COLUMNS[1]], errors="coerce")
    valid = (
        grade6.notna()
        & grade7.notna()
        & grade6.between(0, 10)
        & grade7.between(0, 10)
    )
    if not valid.all():
        raise ValueError(
            "Không thể ước tính: điểm Ngữ văn lớp 6/7 bị thiếu, không phải số "
            "hoặc ngoài khoảng [0, 10]."
        )

    change = grade7 - grade6
    estimated = frame.copy()
    estimated[ESTIMATED_COLUMNS[0]] = (grade7 + change).clip(0, 10).round(1)
    estimated[ESTIMATED_COLUMNS[1]] = (grade7 + 2 * change).clip(0, 10).round(1)
    return estimated


def create_estimated_dataset(
    source_path: Path = RAW_SOURCE,
    output_path: Path = DERIVED_SOURCE,
) -> Path:
    source_path = Path(source_path)
    output_path = Path(output_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu nguồn: {source_path}")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Tệp đầu ra phải khác tệp dữ liệu gốc.")

    source = pd.read_csv(source_path, encoding="utf-8-sig")
    estimated = estimate_literature_scores(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    estimated.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_SOURCE)
    parser.add_argument("--output", type=Path, default=DERIVED_SOURCE)
    args = parser.parse_args()
    output_path = create_estimated_dataset(args.input, args.output)
    print(f"Đã tạo bản sao có điểm ước tính: {output_path}")
    print(
        "Lưu ý: ngu_van_lop8 và ngu_van_lop9 là điểm giả lập, "
        "không phải điểm thực tế."
    )


if __name__ == "__main__":
    main()
