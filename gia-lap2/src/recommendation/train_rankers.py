"""Train and evaluate the subject-ranking baseline, XGBoost and CatBoost.

Example (run from the repository root):
    python -m src.recommendation.train_rankers

The script only reads Development and the fixed V1 folds. Independent Test is
intentionally not touched during model comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .subject_ranker import (
    DEFAULT_TARGET_COLUMNS,
    SubjectRanker,
    evaluate_ranking,
    infer_feature_columns,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train elective-subject ranking models.")
    parser.add_argument("--data", type=Path, default=Path("data/data-splits/development.csv"))
    parser.add_argument("--fold-dir", type=Path, default=Path("data/data-splits/cv_folds_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/subject_rankers"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("baseline", "xgboost", "catboost"),
        default=("baseline", "xgboost", "catboost"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _read_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"ma_hoc_sinh": str})
    return frame["ma_hoc_sinh"].str.strip().tolist()


def run_training(args: argparse.Namespace) -> pd.DataFrame:
    data = pd.read_csv(args.data, encoding="utf-8-sig", dtype={"ma_hoc_sinh": str})
    data["ma_hoc_sinh"] = data["ma_hoc_sinh"].str.strip()
    if not data["ma_hoc_sinh"].is_unique:
        raise ValueError("Development contains duplicate student IDs.")

    feature_columns = infer_feature_columns(data.columns)
    if len(feature_columns) != 96:
        raise ValueError(f"Expected 96 grade features, found {len(feature_columns)}.")
    missing_targets = [c for c in DEFAULT_TARGET_COLUMNS if c not in data.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    indexed = data.set_index("ma_hoc_sinh", drop=False)
    records: list[dict[str, float | int | str]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        for fold in range(1, args.folds + 1):
            train_ids = _read_ids(args.fold_dir / f"fold_{fold}_train_ids.csv")
            validation_ids = _read_ids(args.fold_dir / f"fold_{fold}_validation_ids.csv")
            if not set(train_ids).isdisjoint(validation_ids):
                raise ValueError(f"Training/validation leakage in fold {fold}.")

            train = indexed.loc[train_ids]
            validation = indexed.loc[validation_ids]
            ranker = SubjectRanker(
                model_name=model_name,
                feature_columns=feature_columns,
                random_state=args.random_state,
            ).fit(train[feature_columns], train[DEFAULT_TARGET_COLUMNS])
            scores = ranker.predict_scores(validation[feature_columns])
            metrics = evaluate_ranking(validation[DEFAULT_TARGET_COLUMNS], scores)
            records.append({"Model": model_name, "Fold": fold, **metrics})
            print(f"{model_name:9s} fold={fold}: NDCG@4={metrics['NDCG@4']:.4f}, Recall@4={metrics['Recall@4']:.4f}")

        final_ranker = SubjectRanker(
            model_name=model_name,
            feature_columns=feature_columns,
            random_state=args.random_state,
        ).fit(data[feature_columns], data[DEFAULT_TARGET_COLUMNS])
        final_ranker.save(args.output_dir / f"{model_name}_subject_ranker.joblib")

    fold_metrics = pd.DataFrame.from_records(records)
    fold_metrics.to_csv(args.output_dir / "ranking_metrics_5fold.csv", index=False, encoding="utf-8-sig")
    metric_columns = [c for c in fold_metrics.columns if c not in {"Model", "Fold"}]
    summary = fold_metrics.groupby("Model")[metric_columns].agg(["mean", "std"])
    summary.to_csv(args.output_dir / "ranking_summary.csv", encoding="utf-8-sig")

    metadata = {
        "training_source": str(args.data),
        "independent_test_used": False,
        "models": list(args.models),
        "n_folds": args.folds,
        "n_features": len(feature_columns),
        "target_columns": DEFAULT_TARGET_COLUMNS,
        "primary_metric": "NDCG@4",
        "secondary_metric": "Recall@4",
        "score_meaning": "recommendation score learned from historical subject choices",
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return fold_metrics


def main() -> None:
    run_training(_parse_args())


if __name__ == "__main__":
    main()
