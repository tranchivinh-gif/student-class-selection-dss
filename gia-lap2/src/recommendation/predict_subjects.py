"""Command-line prediction with a saved subject ranker."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .subject_ranker import SubjectRanker


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank elective subjects for new students.")
    parser.add_argument("--model", type=Path, required=True, help="Saved .joblib ranker")
    parser.add_argument("--input", type=Path, required=True, help="CSV containing grade columns")
    parser.add_argument("--output", type=Path, default=Path("subject_recommendations.csv"))
    parser.add_argument("--primary-k", type=int, default=4)
    parser.add_argument("--additional-k", type=int, default=2)
    args = parser.parse_args()

    ranker = SubjectRanker.load(args.model)
    students = pd.read_csv(args.input, encoding="utf-8-sig")
    recommendations = ranker.recommend_grouped(
        students,
        primary_k=args.primary_k,
        additional_k=args.additional_k,
    )

    if "ma_hoc_sinh" in students.columns:
        recommendations.insert(
            0,
            "ma_hoc_sinh",
            recommendations["student_row"].map(students["ma_hoc_sinh"].reset_index(drop=True)),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(recommendations)} ranked recommendations to {args.output}")


if __name__ == "__main__":
    main()
