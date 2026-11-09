"""Baseline, XGBoost and CatBoost models that predict one complete combination."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import FEATURES, RANDOM_SEED
from .data import combination_name, key_to_subjects


@dataclass
class CombinationClassifier:
    algorithm: str
    random_state: int = RANDOM_SEED
    classes_: list[str] = field(default_factory=list, init=False)
    estimator_: Any = field(default=None, init=False, repr=False)
    most_frequent_: str | None = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    def _validate_x(self, x: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in FEATURES if column not in x.columns]
        if missing:
            raise ValueError(f"Thiếu {len(missing)} đặc trưng điểm.")
        values = x[FEATURES].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("Dữ liệu đầu vào có giá trị thiếu hoặc vô hạn.")
        return values

    def fit(self, x: pd.DataFrame, combinations: pd.Series) -> "CombinationClassifier":
        x = self._validate_x(x)
        labels = combinations.astype(str).reset_index(drop=True)
        self.classes_ = sorted(labels.unique().tolist())
        self.most_frequent_ = str(labels.value_counts().idxmax())
        if self.algorithm == "baseline":
            self.fitted_ = True
            return self

        class_to_index = {key: index for index, key in enumerate(self.classes_)}
        encoded = labels.map(class_to_index).to_numpy()
        if self.algorithm == "xgboost":
            from xgboost import XGBClassifier
            self.estimator_ = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9,
                objective="multi:softprob", eval_metric="mlogloss",
                tree_method="hist", random_state=self.random_state, n_jobs=-1,
            )
        elif self.algorithm == "catboost":
            from catboost import CatBoostClassifier
            self.estimator_ = CatBoostClassifier(
                iterations=600, depth=6, learning_rate=0.05,
                loss_function="MultiClass", l2_leaf_reg=3.0,
                random_seed=self.random_state, thread_count=-1,
                verbose=False, allow_writing_files=False,
            )
        else:
            raise ValueError("algorithm phải là baseline, xgboost hoặc catboost")
        self.estimator_.fit(x, encoded)
        self.fitted_ = True
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Mô hình chưa được huấn luyện.")
        x = self._validate_x(x)
        if self.algorithm == "baseline":
            return np.repeat(self.most_frequent_, len(x))
        indices = np.asarray(self.estimator_.predict(x)).astype(int).reshape(-1)
        return np.asarray([self.classes_[index] for index in indices])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        x = self._validate_x(x)
        if self.algorithm == "baseline":
            output = np.zeros((len(x), len(self.classes_)))
            output[:, self.classes_.index(self.most_frequent_)] = 1.0
            return output
        return np.asarray(self.estimator_.predict_proba(x), dtype=float)

    def recommend(self, x: pd.DataFrame, alternatives: int = 3) -> pd.DataFrame:
        probabilities = self.predict_proba(x)
        records = []
        for row_index, row_probabilities in enumerate(probabilities):
            order = np.argsort(-row_probabilities, kind="stable")[:alternatives]
            for rank, class_index in enumerate(order, 1):
                key = self.classes_[class_index]
                records.append({
                    "student_row": row_index,
                    "rank": rank,
                    "combination_key": key,
                    "combination": combination_name(key),
                    "subjects": key_to_subjects(key),
                    "score": float(row_probabilities[class_index]),
                })
        return pd.DataFrame(records)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "CombinationClassifier":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("Artifact không phải CombinationClassifier.")
        return model
