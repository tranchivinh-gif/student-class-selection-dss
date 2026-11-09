"""Run the single, final evaluation on the locked Independent Test set."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .subject_ranker import SubjectRanker, evaluate_per_subject, evaluate_ranking


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final subject-ranker evaluation.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/subject_rankers/catboost_subject_ranker.joblib"),
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/data-splits/independent_test.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/subject_rankers/independent_test"),
    )
    return parser.parse_args()


def run_evaluation(args: argparse.Namespace) -> dict[str, float]:
    ranker = SubjectRanker.load(args.model)
    test = pd.read_csv(args.test_data, encoding="utf-8-sig", dtype={"ma_hoc_sinh": str})
    missing_targets = [c for c in ranker.target_columns if c not in test.columns]
    if missing_targets:
        raise ValueError(f"Independent Test is missing targets: {missing_targets}")

    y_true = test[ranker.target_columns]
    if not y_true.sum(axis=1).eq(4).all():
        raise ValueError("Every Independent Test row must contain exactly four selected subjects.")
    scores = ranker.predict_scores(test)
    metrics = evaluate_ranking(y_true, scores, ks=(1, 3, 4, 5, 6))

    order = np.argsort(-scores, axis=1, kind="stable")
    truth = y_true.to_numpy(dtype=int)
    for k in (4, 5, 6):
        hits = np.take_along_axis(truth, order[:, :k], axis=1).sum(axis=1)
        metrics[f"Full_Coverage@{k}"] = float((hits == 4).mean())
        metrics[f"Average_True_Subjects@{k}"] = float(hits.mean())

    per_subject = evaluate_per_subject(y_true, scores, ranker.target_columns, ks=(4, 6))
    recommendations = ranker.recommend_grouped(test, primary_k=4, additional_k=2)
    if "ma_hoc_sinh" in test.columns:
        recommendations.insert(
            0,
            "ma_hoc_sinh",
            recommendations["student_row"].map(test["ma_hoc_sinh"].reset_index(drop=True)),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        args.output_dir / "final_ranking_metrics.csv", index=False, encoding="utf-8-sig"
    )
    per_subject.to_csv(
        args.output_dir / "final_metrics_per_subject.csv", index=False, encoding="utf-8-sig"
    )
    recommendations.to_csv(
        args.output_dir / "final_recommendations_top4_plus2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "evaluation": "Independent Test - final locked evaluation",
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "model_path": str(args.model),
        "model_name": ranker.model_name,
        "test_path": str(args.test_data),
        "n_test_students": len(test),
        "primary_suggestions": 4,
        "additional_suggestions": 2,
        "warning": "Do not tune the model or decision rules on these results.",
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    metrics = run_evaluation(_parse_args())
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
