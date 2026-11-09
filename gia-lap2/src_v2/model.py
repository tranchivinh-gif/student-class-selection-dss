"""Self-contained baseline, XGBoost and CatBoost ranking models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import ADDITIONAL_K, FEATURES, PRIMARY_K, RANDOM_SEED, TARGETS


@dataclass
class SubjectRankingModel:
    algorithm: str
    random_state: int = RANDOM_SEED
    estimators: list[Any] = field(default_factory=list, init=False, repr=False)
    priors: np.ndarray | None = field(default=None, init=False, repr=False)
    fitted: bool = field(default=False, init=False)

    def _new_estimator(self) -> Any:
        if self.algorithm == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=1.0, colsample_bytree=1.0,
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", random_state=self.random_state, n_jobs=-1,
            )
        if self.algorithm == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.05,
                loss_function="Logloss", l2_leaf_reg=3.0,
                random_seed=self.random_state, thread_count=-1,
                verbose=False, allow_writing_files=False,
            )
        raise ValueError("algorithm phải là baseline, xgboost hoặc catboost")

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "SubjectRankingModel":
        x = self._validate_x(x)
        labels = y[TARGETS].to_numpy(dtype=int)
        self.priors = labels.mean(axis=0)
        self.estimators = []
        if self.algorithm != "baseline":
            for index in range(len(TARGETS)):
                unique_labels = np.unique(labels[:, index])
                if len(unique_labels) == 1:
                    # Small real-world datasets can contain a subject that is
                    # always selected (or never selected) in a training fold.
                    # Tree classifiers cannot fit a one-class target, so keep
                    # its exact constant probability instead.
                    self.estimators.append(float(unique_labels[0]))
                    continue
                estimator = self._new_estimator()
                estimator.fit(x, labels[:, index])
                self.estimators.append(estimator)
        self.fitted = True
        return self

    def _validate_x(self, x: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in FEATURES if column not in x.columns]
        if missing:
            raise ValueError(f"Thiếu {len(missing)} đặc trưng điểm.")
        values = x[FEATURES].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("Đặc trưng có giá trị thiếu hoặc vô hạn.")
        return values

    def predict_scores(self, x: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Mô hình chưa được huấn luyện.")
        x = self._validate_x(x)
        if self.algorithm == "baseline":
            return np.tile(self.priors, (len(x), 1))
        output = []
        for estimator in self.estimators:
            if np.isscalar(estimator):
                output.append(np.full(len(x), float(estimator)))
                continue
            classes = np.asarray(estimator.classes_)
            positive = np.flatnonzero(classes == 1)
            output.append(
                estimator.predict_proba(x)[:, positive[0]]
                if len(positive) else np.zeros(len(x))
            )
        return np.column_stack(output)

    def recommend(self, x: pd.DataFrame) -> pd.DataFrame:
        scores = self.predict_scores(x)
        records = []
        for row, row_scores in enumerate(scores):
            for rank, index in enumerate(np.argsort(-row_scores, kind="stable")[:6], 1):
                records.append({
                    "student_row": row,
                    "rank": rank,
                    "group": "Đề xuất phù hợp nhất" if rank <= PRIMARY_K else "Có thể cân nhắc thêm",
                    "subject": TARGETS[index],
                    "recommendation_score": float(row_scores[index]),
                })
        return pd.DataFrame(records)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "SubjectRankingModel":
        return joblib.load(path)
