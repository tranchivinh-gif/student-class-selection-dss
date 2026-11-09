"""Rank elective subjects from a student's grade history.

The module treats every subject as a binary target, obtains one score per
subject, then sorts all subjects by that score.  Scores are recommendation
scores learned from historical choices; they are not career-advice scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import ndcg_score


DEFAULT_TARGET_COLUMNS = [
    "Vật lí",
    "Hóa học",
    "Sinh học",
    "Địa lí",
    "Giáo dục kinh tế và pháp luật (GDKT&PL)",
    "Tin học",
    "Công nghệ",
    "Âm nhạc",
    "Mĩ thuật",
]


def infer_feature_columns(columns: Iterable[str]) -> list[str]:
    """Return the 96 grade columns used by the current project."""
    return [
        column
        for column in columns
        if column.startswith(("L6_", "L7_", "L8_", "L9_"))
        and column.endswith(("_hki", "_hkii", "_ca_nam"))
    ]


def _recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, scores.shape[1])
    top_k = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    hits = np.take_along_axis(y_true, top_k, axis=1).sum(axis=1)
    relevant = y_true.sum(axis=1)
    return float(np.mean(np.divide(hits, relevant, out=np.zeros_like(hits, dtype=float), where=relevant > 0)))


def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, scores.shape[1])
    top_k = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    hits = np.take_along_axis(y_true, top_k, axis=1).sum(axis=1)
    return float(np.mean(hits / k))


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked_true = np.take_along_axis(y_true, order, axis=1)
    precision = np.cumsum(ranked_true, axis=1) / np.arange(1, scores.shape[1] + 1)
    relevant = y_true.sum(axis=1)
    ap = np.divide(
        (precision * ranked_true).sum(axis=1),
        relevant,
        out=np.zeros(len(y_true), dtype=float),
        where=relevant > 0,
    )
    return float(ap.mean())


def evaluate_ranking(
    y_true: pd.DataFrame | np.ndarray,
    scores: np.ndarray,
    ks: Sequence[int] = (1, 3, 4, 5),
) -> dict[str, float]:
    """Evaluate a complete subject ranking.

    NDCG rewards correct subjects appearing near the top. Recall@K answers
    how many of a student's actual choices occur in the first K suggestions.
    """
    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(scores, dtype=float)
    if truth.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: y_true={truth.shape}, scores={predicted.shape}")
    if not np.isfinite(predicted).all():
        raise ValueError("Scores contain NaN or infinite values.")

    result: dict[str, float] = {"MAP": _average_precision(truth, predicted)}
    for k in ks:
        if k < 1:
            raise ValueError("Every K must be at least 1.")
        effective_k = min(k, truth.shape[1])
        result[f"Precision@{k}"] = _precision_at_k(truth, predicted, effective_k)
        result[f"Recall@{k}"] = _recall_at_k(truth, predicted, effective_k)
        result[f"NDCG@{k}"] = float(ndcg_score(truth, predicted, k=effective_k))
    return result


def evaluate_per_subject(
    y_true: pd.DataFrame | np.ndarray,
    scores: np.ndarray,
    target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
    ks: Sequence[int] = (4, 6),
) -> pd.DataFrame:
    """Report ranking quality separately for every subject."""
    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(scores, dtype=float)
    if truth.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: y_true={truth.shape}, scores={predicted.shape}")
    if truth.shape[1] != len(target_columns):
        raise ValueError("target_columns does not match the number of score columns.")

    order = np.argsort(-predicted, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(len(order))[:, None]
    ranks[rows, order] = np.arange(1, predicted.shape[1] + 1)

    records: list[dict[str, Any]] = []
    for index, subject in enumerate(target_columns):
        positive = truth[:, index] == 1
        positive_count = int(positive.sum())
        subject_ranks = ranks[positive, index]
        record: dict[str, Any] = {
            "subject": subject,
            "actual_positive": positive_count,
            "prevalence": float(positive.mean()),
            "mean_rank_when_selected": float(subject_ranks.mean()) if positive_count else np.nan,
            "median_rank_when_selected": float(np.median(subject_ranks)) if positive_count else np.nan,
        }
        for k in ks:
            effective_k = min(k, truth.shape[1])
            hits = int((subject_ranks <= effective_k).sum())
            record[f"hits_at_{k}"] = hits
            record[f"recall_at_{k}"] = hits / positive_count if positive_count else np.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def _make_estimator(model_name: str, random_state: int, parameters: Mapping[str, Any]) -> Any:
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        defaults = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
        }
        defaults.update(parameters)
        return XGBClassifier(**defaults)

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        defaults = {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "Logloss",
            "l2_leaf_reg": 3.0,
            "random_seed": random_state,
            "thread_count": -1,
            "verbose": False,
            "allow_writing_files": False,
        }
        defaults.update(parameters)
        return CatBoostClassifier(**defaults)

    raise ValueError("model_name must be one of: baseline, xgboost, catboost")


@dataclass
class SubjectRanker:
    """Fit one binary learner per subject and return a ranked recommendation."""

    model_name: str
    feature_columns: Sequence[str]
    target_columns: Sequence[str] = field(default_factory=lambda: DEFAULT_TARGET_COLUMNS.copy())
    random_state: int = 42
    model_parameters: Mapping[str, Any] = field(default_factory=dict)
    estimators_: list[Any] = field(default_factory=list, init=False, repr=False)
    prior_scores_: np.ndarray | None = field(default=None, init=False, repr=False)
    is_fitted_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.model_name = self.model_name.lower().strip()
        if self.model_name not in {"baseline", "xgboost", "catboost"}:
            raise ValueError("model_name must be one of: baseline, xgboost, catboost")
        self.feature_columns = list(self.feature_columns)
        self.target_columns = list(self.target_columns)
        if not self.feature_columns or not self.target_columns:
            raise ValueError("feature_columns and target_columns cannot be empty.")

    def _validate_x(self, x: pd.DataFrame | Mapping[str, float]) -> pd.DataFrame:
        frame = pd.DataFrame([x]) if isinstance(x, Mapping) else x.copy()
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"Missing {len(missing)} required features: {preview}")
        frame = frame.loc[:, self.feature_columns].apply(pd.to_numeric, errors="raise")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Input features contain missing or infinite values.")
        return frame

    def fit(self, x: pd.DataFrame, y: pd.DataFrame | np.ndarray) -> "SubjectRanker":
        features = self._validate_x(x)
        targets = np.asarray(y, dtype=int)
        if targets.shape != (len(features), len(self.target_columns)):
            raise ValueError(
                f"Expected y shape {(len(features), len(self.target_columns))}, got {targets.shape}"
            )
        if not np.isin(targets, [0, 1]).all():
            raise ValueError("Targets must contain only 0 and 1.")

        self.prior_scores_ = targets.mean(axis=0)
        self.estimators_ = []
        if self.model_name != "baseline":
            base = _make_estimator(self.model_name, self.random_state, self.model_parameters)
            for target_index in range(targets.shape[1]):
                estimator = clone(base)
                estimator.fit(features, targets[:, target_index])
                self.estimators_.append(estimator)
        self.is_fitted_ = True
        return self

    @staticmethod
    def _positive_probability(estimator: Any, x: pd.DataFrame) -> np.ndarray:
        probabilities = estimator.predict_proba(x)
        classes = np.asarray(estimator.classes_)
        positive = np.flatnonzero(classes == 1)
        if len(positive) == 0:
            return np.zeros(len(x), dtype=float)
        return np.asarray(probabilities)[:, positive[0]]

    def predict_scores(self, x: pd.DataFrame | Mapping[str, float]) -> np.ndarray:
        """Return one recommendation score per subject, without truncating to Top-4."""
        if not self.is_fitted_:
            raise RuntimeError("The ranker must be fitted or loaded before prediction.")
        features = self._validate_x(x)
        if self.model_name == "baseline":
            assert self.prior_scores_ is not None
            return np.tile(self.prior_scores_, (len(features), 1))
        return np.column_stack(
            [self._positive_probability(estimator, features) for estimator in self.estimators_]
        )

    def recommend(
        self,
        x: pd.DataFrame | Mapping[str, float],
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """Return long-form rankings for one or more students."""
        scores = self.predict_scores(x)
        k = len(self.target_columns) if top_k is None else top_k
        if not 1 <= k <= len(self.target_columns):
            raise ValueError(f"top_k must be between 1 and {len(self.target_columns)}")

        records: list[dict[str, Any]] = []
        for row_index, row_scores in enumerate(scores):
            order = np.argsort(-row_scores, kind="stable")[:k]
            for rank, target_index in enumerate(order, start=1):
                records.append(
                    {
                        "student_row": row_index,
                        "rank": rank,
                        "subject": self.target_columns[target_index],
                        "recommendation_score": float(row_scores[target_index]),
                    }
                )
        return pd.DataFrame.from_records(records)

    def recommend_grouped(
        self,
        x: pd.DataFrame | Mapping[str, float],
        primary_k: int = 4,
        additional_k: int = 2,
    ) -> pd.DataFrame:
        """Return primary suggestions followed by additional considerations."""
        if primary_k < 1 or additional_k < 0:
            raise ValueError("primary_k must be positive and additional_k cannot be negative.")
        total = primary_k + additional_k
        if total > len(self.target_columns):
            raise ValueError(f"primary_k + additional_k cannot exceed {len(self.target_columns)}")
        recommendations = self.recommend(x, top_k=total)
        recommendations.insert(
            2,
            "recommendation_group",
            np.where(
                recommendations["rank"] <= primary_k,
                "Đề xuất phù hợp nhất",
                "Có thể cân nhắc thêm",
            ),
        )
        return recommendations

    def save(self, path: str | Path) -> Path:
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted ranker.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "SubjectRanker":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"{path} does not contain a SubjectRanker.")
        return loaded
