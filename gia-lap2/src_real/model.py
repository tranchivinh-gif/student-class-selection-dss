from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd

from .config import ALLOWED_COMBINATIONS, FEATURES, RANDOM_SEED, TARGETS
from .data import combination_name, subjects_from_key


@dataclass
class RealCombinationModel:
    random_state: int = RANDOM_SEED
    algorithm: str = "catboost"
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURES))
    estimator: Any = field(default=None, init=False, repr=False)
    classes_: list[str] = field(default_factory=lambda: list(ALLOWED_COMBINATIONS), init=False)
    class_probabilities_: np.ndarray | None = field(default=None, init=False, repr=False)

    def _features(self) -> list[str]:
        # Artifacts created before feature-set support do not contain this field.
        return list(getattr(self, "feature_columns", FEATURES))

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "RealCombinationModel":
        labels = y.astype(str)
        unknown = sorted(set(labels) - set(self.classes_))
        if unknown:
            raise ValueError(f"Tổ hợp không được hỗ trợ: {unknown}")
        encoded = labels.map({key: i for i, key in enumerate(self.classes_)})
        self.class_probabilities_ = (
            encoded.value_counts(normalize=True).reindex(range(len(self.classes_)), fill_value=0)
            .sort_index().to_numpy(float)
        )
        if self.algorithm == "baseline":
            self.estimator = None
            return self
        if self.algorithm == "catboost":
            from catboost import CatBoostClassifier
            self.estimator = CatBoostClassifier(
                iterations=500, depth=5, learning_rate=0.04,
                loss_function="MultiClass", l2_leaf_reg=5.0,
                random_seed=self.random_state, verbose=False,
                allow_writing_files=False, thread_count=-1,
            )
        elif self.algorithm == "xgboost":
            from xgboost import XGBClassifier
            self.estimator = XGBClassifier(
                n_estimators=350, max_depth=5, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=self.random_state, tree_method="hist", n_jobs=-1,
            )
        else:
            raise ValueError("algorithm phải là baseline, xgboost hoặc catboost.")
        self.estimator.fit(x[self._features()], encoded)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.algorithm == "baseline":
            if self.class_probabilities_ is None:
                raise RuntimeError("Mô hình chưa được huấn luyện.")
            return np.tile(self.class_probabilities_, (len(x), 1))
        if self.estimator is None:
            raise RuntimeError("Mô hình chưa được huấn luyện.")
        return np.asarray(self.estimator.predict_proba(x[self._features()]), dtype=float)

    def feature_contributions(self, x: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
        """Return SHAP contributions for the predicted combination of one student."""
        if self.algorithm == "baseline":
            return pd.DataFrame(columns=["feature", "contribution", "direction"])
        if self.estimator is None or len(x) != 1:
            raise ValueError("SHAP yêu cầu model đã fit và đúng một học sinh.")
        import shap

        feature_columns = self._features()
        values = shap.TreeExplainer(self.estimator).shap_values(x[feature_columns])
        predicted_class = int(self.predict_proba(x).argmax(axis=1)[0])
        if isinstance(values, list):
            contribution = np.asarray(values[predicted_class])[0]
        else:
            array = np.asarray(values)
            if array.ndim == 3 and array.shape[-1] == len(self.classes_):
                contribution = array[0, :, predicted_class]
            elif array.ndim == 3 and array.shape[0] == len(self.classes_):
                contribution = array[predicted_class, 0, :]
            elif array.ndim == 2:
                contribution = array[0]
            else:
                raise ValueError(f"Không nhận dạng được SHAP shape: {array.shape}")
        result = pd.DataFrame({"feature": feature_columns, "contribution": contribution})
        result["direction"] = np.where(result["contribution"] >= 0, "Tăng", "Giảm")
        return result.reindex(result["contribution"].abs().sort_values(ascending=False).index).head(top_n)

    def recommend(self, x: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
        top_k = max(1, min(int(top_k), len(self.classes_)))
        probabilities = self.predict_proba(x)
        records = []
        ids = x.get("ma_hoc_sinh", pd.Series(range(len(x)))).astype(str).tolist()
        for row, scores in enumerate(probabilities):
            for rank, index in enumerate(np.argsort(-scores)[:top_k], 1):
                key = self.classes_[index]
                records.append({
                    "ma_hoc_sinh": ids[row], "rank": rank,
                    "combination_key": key, "combination": combination_name(key),
                    "subjects": " | ".join(subjects_from_key(key)),
                    "probability": float(scores[index]),
                })
        return pd.DataFrame(records)

    def subject_probabilities(self, x: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.predict_proba(x)
        membership = np.asarray([[int(bit) for bit in key] for key in self.classes_])
        result = probabilities @ membership
        return pd.DataFrame(result, columns=TARGETS, index=x.index)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "RealCombinationModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("Artifact không phải RealCombinationModel.")
        return model

