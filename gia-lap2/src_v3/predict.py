"""Predict the most likely complete four-subject combinations from a CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .model import CombinationClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/src_v3/champion_combination.joblib"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("combination_predictions.csv"))
    parser.add_argument("--alternatives", type=int, default=3)
    args = parser.parse_args()
    model = CombinationClassifier.load(args.model)
    data = pd.read_csv(args.input, encoding="utf-8-sig")
    result = model.recommend(data, alternatives=args.alternatives)
    result["subjects"] = result["subjects"].map(" | ".join)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Đã lưu {len(result)} dự đoán tại {args.output}")


if __name__ == "__main__":
    main()
