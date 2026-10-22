"""Generate multi-label subject predictions using a saved model artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from preprocessing import PROJECT_DIR
from selection import SUBJECT_COUNT, select_top_subjects


MODEL_PATH = PROJECT_DIR / "models" / "xgboost_model.pkl"


def load_model(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình: {model_path}. Hãy chạy train.py trước."
        )
    artifact = joblib.load(model_path)
    required = {"classifier", "feature_columns", "target_columns", "subject_count"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Tệp mô hình thiếu metadata: {sorted(missing)}")
    return artifact


def predict_frame(
    frame: pd.DataFrame,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    """Predict from score columns only; IDs and any supplied labels are ignored."""
    if frame.empty:
        raise ValueError("Tệp đầu vào không có dòng học sinh.")
    feature_columns: list[str] = artifact["feature_columns"]
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Thiếu cột điểm đầu vào: {missing}")

    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        bad_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Điểm thiếu hoặc không phải số ở cột: {bad_columns}")
    invalid = ~features.apply(lambda column: column.between(0, 10)).all()
    if invalid.any():
        bad_columns = invalid[invalid].index.tolist()
        raise ValueError(f"Điểm ngoài khoảng [0, 10] ở cột: {bad_columns}")

    classifier = artifact["classifier"]
    target_columns: list[str] = artifact["target_columns"]
    probabilities = np.column_stack(
        [item[:, 1] for item in classifier.predict_proba(features)]
    )
    predictions = select_top_subjects(probabilities, int(artifact["subject_count"]))

    output: dict[str, Any] = {}
    for index, label in enumerate(target_columns):
        output[f"predicted::{label}"] = predictions[:, index]
        output[f"probability::{label}"] = probabilities[:, index]

    result = pd.DataFrame(output, index=frame.index)
    if "ma_hoc_sinh" in frame.columns:
        result.insert(0, "ma_hoc_sinh", frame["ma_hoc_sinh"])
    return result


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    predictions = predict_frame(frame, load_model(args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Đã dự đoán {len(predictions):,} dòng; lưu tại {args.output}")
    print(f"Mỗi học sinh được đề xuất đúng {SUBJECT_COUNT} môn có xác suất cao nhất.")
    print("Xác suất là mức độ dự đoán lựa chọn theo dữ liệu, không phải độ phù hợp.")


if __name__ == "__main__":
    main()
