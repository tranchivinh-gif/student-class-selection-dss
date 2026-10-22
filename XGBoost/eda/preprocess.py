"""Keep students who selected exactly four elective subjects."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


XGBOOST_DIR = Path(__file__).resolve().parents[1]
if str(XGBOOST_DIR) not in sys.path:
    sys.path.insert(0, str(XGBOOST_DIR))

from preprocessing import prepare_dataframe


DEFAULT_INPUT = XGBOOST_DIR / "data" / "hoc_sinh.csv"
DEFAULT_OUTPUT = XGBOOST_DIR / "data" / "hoc_sinh_4_mon.csv"
SUBJECT_COUNT = 4


def preprocess_data(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate and normalize source data, then keep exactly four selected subjects."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu nguồn: {input_path}")

    source = pd.read_csv(input_path, encoding="utf-8-sig")
    cleaned, _, label_columns, _ = prepare_dataframe(source)
    selected_subjects = cleaned[label_columns].sum(axis=1)
    keep_rows = selected_subjects.eq(SUBJECT_COUNT)
    audit = {
        "input_rows": len(cleaned),
        "kept_rows": int(keep_rows.sum()),
        "removed_fewer_than_four": int(selected_subjects.lt(SUBJECT_COUNT).sum()),
        "removed_more_than_four": int(selected_subjects.gt(SUBJECT_COUNT).sum()),
    }
    filtered = cleaned.loc[keep_rows].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False, encoding="utf-8")
    return filtered, audit


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    filtered, audit = preprocess_data(args.input, args.output)
    print(
        f"Đầu vào: {audit['input_rows']:,} dòng; "
        f"giữ lại {audit['kept_rows']:,} dòng có đúng {SUBJECT_COUNT} môn."
    )
    print(
        "Đã loại: "
        f"{audit['removed_fewer_than_four']:,} dòng dưới {SUBJECT_COUNT} môn, "
        f"{audit['removed_more_than_four']:,} dòng trên {SUBJECT_COUNT} môn."
    )
    print(f"Tệp kết quả: {args.output.resolve()}")


if __name__ == "__main__":
    main()
