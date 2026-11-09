"""Ranking metrics used for model selection and final reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def evaluate(y_true: np.ndarray, scores: np.ndarray, ks=(4, 6)) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, axis=1, kind="stable")
    result: dict[str, float] = {}
    for k in ks:
        hits = np.take_along_axis(truth, order[:, :k], axis=1).sum(axis=1)
        result[f"NDCG@{k}"] = float(ndcg_score(truth, scores, k=k))
        result[f"Recall@{k}"] = float(np.mean(hits / truth.sum(axis=1)))
        result[f"Full_Coverage@{k}"] = float(np.mean(hits == truth.sum(axis=1)))
        result[f"Average_Hits@{k}"] = float(hits.mean())
    return result


def per_subject(y_true: np.ndarray, scores: np.ndarray, targets: list[str]) -> pd.DataFrame:
    truth = np.asarray(y_true, dtype=int)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(len(order))[:, None], order] = np.arange(1, scores.shape[1] + 1)
    records = []
    for index, subject in enumerate(targets):
        selected = truth[:, index] == 1
        selected_ranks = ranks[selected, index]
        records.append({
            "subject": subject,
            "positive_samples": int(selected.sum()),
            "mean_rank": float(selected_ranks.mean()) if selected.any() else None,
            "recall_at_4": float((selected_ranks <= 4).mean()) if selected.any() else None,
            "recall_at_6": float((selected_ranks <= 6).mean()) if selected.any() else None,
        })
    return pd.DataFrame(records)
