"""Metrics for exact combination prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def evaluate(y_true, y_pred, probabilities=None, classes=None) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=str)
    predicted = np.asarray(y_pred, dtype=str)
    result = {
        "Exact_Match_Accuracy": float(accuracy_score(truth, predicted)),
        "Balanced_Accuracy": float(balanced_accuracy_score(truth, predicted)),
        "Macro_F1_Combination": float(f1_score(truth, predicted, average="macro", zero_division=0)),
    }
    if probabilities is not None and classes is not None:
        order = np.argsort(-probabilities, axis=1, kind="stable")
        class_array = np.asarray(classes)
        for k in (2, 3):
            top = class_array[order[:, :k]]
            result[f"Top_{k}_Combination_Accuracy"] = float(
                np.mean([actual in candidates for actual, candidates in zip(truth, top)])
            )
    return result
