"""Complete training workflow for a newly supplied labelled CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .config import DEFAULT_OUTPUT, FEATURES, ID_COLUMN, RANDOM_SEED, TARGETS
from .data import label_combination, load_and_validate, split_dataset
from .metrics import evaluate, per_subject
from .model import SubjectRankingModel


def train_pipeline(
    data_path: Path,
    output_dir: Path = DEFAULT_OUTPUT,
    folds: int = 5,
    holdout_size: float = 0.1,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_and_validate(data_path)
    split = split_dataset(data, holdout_size)
    split.development.to_csv(output_dir / "development.csv", index=False, encoding="utf-8-sig")
    split.holdout.to_csv(output_dir / "locked_holdout.csv", index=False, encoding="utf-8-sig")

    combinations = label_combination(split.development)
    can_stratify = combinations.value_counts().min() >= folds
    splitter = (
        StratifiedKFold(folds, shuffle=True, random_state=RANDOM_SEED)
        if can_stratify else KFold(folds, shuffle=True, random_state=RANDOM_SEED)
    )
    split_iter = splitter.split(split.development, combinations if can_stratify else None)
    fold_indices = list(split_iter)
    records = []
    for algorithm in ("baseline", "xgboost", "catboost"):
        for fold, (train_idx, val_idx) in enumerate(fold_indices, 1):
            train = split.development.iloc[train_idx]
            validation = split.development.iloc[val_idx]
            model = SubjectRankingModel(algorithm).fit(train[FEATURES], train[TARGETS])
            metrics = evaluate(validation[TARGETS], model.predict_scores(validation[FEATURES]))
            records.append({"model": algorithm, "fold": fold, **metrics})
            print(f"{algorithm} fold {fold}: NDCG@4={metrics['NDCG@4']:.4f}")

    cv = pd.DataFrame(records)
    cv.to_csv(output_dir / "cv_metrics.csv", index=False, encoding="utf-8-sig")
    summary = cv.groupby("model").mean(numeric_only=True)
    summary.to_csv(output_dir / "cv_summary.csv", encoding="utf-8-sig")
    champion = str(summary["NDCG@4"].idxmax())

    final = SubjectRankingModel(champion).fit(
        split.development[FEATURES], split.development[TARGETS]
    )
    holdout_scores = final.predict_scores(split.holdout[FEATURES])
    holdout_metrics = evaluate(split.holdout[TARGETS], holdout_scores)
    pd.DataFrame([holdout_metrics]).to_csv(
        output_dir / "holdout_metrics.csv", index=False, encoding="utf-8-sig"
    )
    per_subject(split.holdout[TARGETS], holdout_scores, TARGETS).to_csv(
        output_dir / "holdout_per_subject.csv", index=False, encoding="utf-8-sig"
    )

    # Deployment artifact is retrained on all supplied data only after evaluation.
    deployed = SubjectRankingModel(champion).fit(data[FEATURES], data[TARGETS])
    deployed.save(output_dir / "champion.joblib")
    metadata = {
        "champion": champion,
        "selection_metric": "NDCG@4",
        "rows": len(data),
        "development_rows": len(split.development),
        "holdout_rows": len(split.holdout),
        "holdout_used_for_selection": False,
        "features": FEATURES,
        "targets": TARGETS,
        "holdout_metrics": holdout_metrics,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Champion: {champion}; artifact: {output_dir / 'champion.joblib'}")
    return champion


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the complete subject recommender.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-size", type=float, default=0.1)
    args = parser.parse_args()
    train_pipeline(args.data, args.output, args.folds, args.holdout_size)


if __name__ == "__main__":
    main()
