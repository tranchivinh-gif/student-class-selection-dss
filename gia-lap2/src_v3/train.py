"""Cross-validate and train a direct four-subject combination classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .config import DEFAULT_OUTPUT, FEATURES, RANDOM_SEED
from .data import load_and_validate
from .metrics import evaluate
from .model import CombinationClassifier


def run(data_path: Path, output: Path, folds: int = 5, holdout_size: float = .1) -> str:
    output.mkdir(parents=True, exist_ok=True)
    data = load_and_validate(data_path)
    labels = data["combination_key"]
    can_holdout_stratify = labels.value_counts().min() >= 2
    development, holdout = train_test_split(
        data, test_size=holdout_size, random_state=RANDOM_SEED, shuffle=True,
        stratify=labels if can_holdout_stratify else None,
    )
    development, holdout = development.reset_index(drop=True), holdout.reset_index(drop=True)
    development.to_csv(output / "development.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(output / "locked_holdout.csv", index=False, encoding="utf-8-sig")

    dev_labels = development["combination_key"]
    stratified = dev_labels.value_counts().min() >= folds
    splitter = (
        StratifiedKFold(folds, shuffle=True, random_state=RANDOM_SEED)
        if stratified else KFold(folds, shuffle=True, random_state=RANDOM_SEED)
    )
    indices = list(splitter.split(development, dev_labels if stratified else None))
    records = []
    for algorithm in ("baseline", "xgboost", "catboost"):
        for fold, (train_idx, validation_idx) in enumerate(indices, 1):
            train, validation = development.iloc[train_idx], development.iloc[validation_idx]
            model = CombinationClassifier(algorithm).fit(train[FEATURES], train["combination_key"])
            predictions = model.predict(validation[FEATURES])
            metrics = evaluate(
                validation["combination_key"], predictions,
                model.predict_proba(validation[FEATURES]), model.classes_,
            )
            records.append({"model": algorithm, "fold": fold, **metrics})
            print(f"{algorithm} fold {fold}: Exact={metrics['Exact_Match_Accuracy']:.4f}")

    cv = pd.DataFrame(records)
    cv.to_csv(output / "cv_metrics.csv", index=False, encoding="utf-8-sig")
    summary = cv.groupby("model").mean(numeric_only=True)
    summary.to_csv(output / "cv_summary.csv", encoding="utf-8-sig")
    champion = str(summary["Exact_Match_Accuracy"].idxmax())

    evaluation_model = CombinationClassifier(champion).fit(
        development[FEATURES], development["combination_key"]
    )
    holdout_prediction = evaluation_model.predict(holdout[FEATURES])
    holdout_metrics = evaluate(
        holdout["combination_key"], holdout_prediction,
        evaluation_model.predict_proba(holdout[FEATURES]), evaluation_model.classes_,
    )
    pd.DataFrame([holdout_metrics]).to_csv(
        output / "holdout_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({
        "actual": holdout["combination_key"],
        "predicted": holdout_prediction,
        "correct": holdout["combination_key"].to_numpy() == holdout_prediction,
    }).to_csv(output / "holdout_predictions.csv", index=False, encoding="utf-8-sig")

    deployed = CombinationClassifier(champion).fit(data[FEATURES], data["combination_key"])
    deployed.save(output / "champion_combination.joblib")
    metadata = {
        "task": "direct four-subject combination classification",
        "champion": champion,
        "selection_metric": "Exact_Match_Accuracy",
        "n_combinations": int(labels.nunique()),
        "holdout_used_for_selection": False,
        "holdout_metrics": holdout_metrics,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Champion: {champion}; Exact holdout={holdout_metrics['Exact_Match_Accuracy']:.4f}")
    return champion


def main() -> None:
    parser = argparse.ArgumentParser(description="Train direct combination classifiers.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-size", type=float, default=.1)
    args = parser.parse_args()
    run(args.data, args.output, args.folds, args.holdout_size)


if __name__ == "__main__":
    main()
