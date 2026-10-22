"""Tune and train a multi-label XGBoost model using train/validation/test splits."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from evaluate import calculate_metrics
from preprocessing import (
    CLEAN_DATA_PATH,
    PROJECT_DIR,
    prepare_dataframe,
)
from selection import SUBJECT_COUNT, select_top_subjects


MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
REPORTS_DIR = PROJECT_DIR / "reports"
TRAINING_REPORT_PATH = REPORTS_DIR / "training_report.json"
PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE_FROM_TRAIN_VALIDATION = 0.15 / 0.85

PARAMETER_CANDIDATES: list[dict[str, Any]] = [
    {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 140,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
    },
    {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 180,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    {
        "max_depth": 2,
        "learning_rate": 0.05,
        "n_estimators": 240,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 240,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
    {
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 320,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    },
]


def create_classifier(parameters: dict[str, Any]) -> MultiOutputClassifier:
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=min(4, os.cpu_count() or 1),
        random_state=RANDOM_STATE,
        verbosity=0,
        **parameters,
    )
    return MultiOutputClassifier(base_model, n_jobs=1)


def split_indices(row_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic 70/15/15 row indices without using label values."""
    indices = np.arange(row_count)
    train_validation, test = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=VALIDATION_SIZE_FROM_TRAIN_VALIDATION,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    return train, validation, test


def _check_class_counts(labels: pd.DataFrame, train_indices: np.ndarray) -> None:
    rare = {
        column: labels.iloc[train_indices][column].value_counts().to_dict()
        for column in labels.columns
        if labels.iloc[train_indices][column].nunique() < 2
    }
    if rare:
        raise ValueError(
            "Tập Train thiếu một trong hai lớp cho các nhãn sau; "
            f"không thể huấn luyện đáng tin cậy: {rare}"
        )


def train_model(source_path: Path = CLEAN_DATA_PATH) -> dict[str, Any]:
    """Tune on Validation and save a model and untouched Test predictions."""
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Chưa có dữ liệu sạch: {source_path}. Hãy chạy preprocessing.py trước.")
    cleaned, feature_columns, target_columns, audit = prepare_dataframe(
        pd.read_csv(source_path, encoding="utf-8-sig")
    )
    features = cleaned[feature_columns]
    labels = cleaned[target_columns]
    train_indices, validation_indices, test_indices = split_indices(len(cleaned))
    _check_class_counts(labels, train_indices)
    if len(target_columns) < SUBJECT_COUNT:
        raise ValueError(f"Cần ít nhất {SUBJECT_COUNT} cột môn để đề xuất đúng {SUBJECT_COUNT} môn.")

    validation_trials: list[dict[str, Any]] = []
    best_parameters: dict[str, Any] | None = None
    best_validation_macro_f1 = -1.0
    best_validation_micro_f1 = -1.0
    best_validation_exact_match = -1.0
    y_train = labels.iloc[train_indices]
    y_validation = labels.iloc[validation_indices]

    for parameters in PARAMETER_CANDIDATES:
        candidate = create_classifier(parameters)
        candidate.fit(features.iloc[train_indices], y_train)
        validation_probabilities = np.column_stack(
            [item[:, 1] for item in candidate.predict_proba(features.iloc[validation_indices])]
        )
        validation_predictions = select_top_subjects(validation_probabilities)
        metrics = calculate_metrics(
            y_validation.to_numpy(),
            validation_predictions,
            target_columns,
        )
        trial = {
            "parameters": parameters,
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "exact_match_accuracy": metrics["exact_match_accuracy"],
        }
        validation_trials.append(trial)
        score = (trial["macro_f1"], trial["exact_match_accuracy"], trial["micro_f1"])
        best_score = (best_validation_macro_f1, best_validation_exact_match, best_validation_micro_f1)
        if score > best_score:
            best_parameters = parameters
            best_validation_macro_f1, best_validation_exact_match, best_validation_micro_f1 = score

    if best_parameters is None:
        raise RuntimeError("Không chọn được cấu hình XGBoost trên Validation.")

    train_validation_indices = np.concatenate((train_indices, validation_indices))
    final_model = create_classifier(best_parameters)
    final_model.fit(
        features.iloc[train_validation_indices],
        labels.iloc[train_validation_indices],
    )

    test_features = features.iloc[test_indices]
    y_test = labels.iloc[test_indices].to_numpy(dtype=int)
    test_probabilities = np.column_stack(
        [probabilities[:, 1] for probabilities in final_model.predict_proba(test_features)]
    )
    test_predictions = select_top_subjects(test_probabilities)

    result_frame = pd.DataFrame(index=np.arange(len(test_indices)))
    for index, label in enumerate(target_columns):
        result_frame[f"actual::{label}"] = y_test[:, index]
        result_frame[f"predicted::{label}"] = test_predictions[:, index]
        result_frame[f"probability::{label}"] = test_probabilities[:, index]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")
    artifact = {
        "classifier": final_model,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "student_id_column": "ma_hoc_sinh",
        "best_params": best_parameters,
        "subject_count": SUBJECT_COUNT,
        "split_counts": {
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "test": int(len(test_indices)),
        },
        "random_state": RANDOM_STATE,
        "estimated_feature_columns": [],
    }
    joblib.dump(artifact, MODEL_PATH)

    top_frequency_indices = np.argsort(-y_train.mean(axis=0).to_numpy(), kind="stable")[:SUBJECT_COUNT]
    majority_predictions = np.zeros_like(y_test)
    majority_predictions[:, top_frequency_indices] = 1
    baseline_metrics = calculate_metrics(
        y_test, majority_predictions, target_columns
    )
    training_report = {
        "source_file": Path(source_path).name,
        "data_audit": audit,
        "split_counts": artifact["split_counts"],
        "random_state": RANDOM_STATE,
        "selection_metric": "validation_macro_f1_with_exact_match_tiebreak",
        "subject_count": SUBJECT_COUNT,
        "baseline": "four_most_frequent_labels_in_train",
        "selected_parameters": best_parameters,
        "validation_trials": validation_trials,
        "baseline_metrics": baseline_metrics,
        "estimated_feature_columns": artifact["estimated_feature_columns"],
    }
    TRAINING_REPORT_PATH.write_text(
        json.dumps(training_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Train/Validation/Test: {len(train_indices)}/"
        f"{len(validation_indices)}/{len(test_indices)}"
    )
    print(f"Model: {MODEL_PATH}")
    print(f"Dự đoán Test: {PREDICTIONS_PATH}")
    return artifact


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CLEAN_DATA_PATH)
    args = parser.parse_args()
    train_model(args.input)


if __name__ == "__main__":
    main()
