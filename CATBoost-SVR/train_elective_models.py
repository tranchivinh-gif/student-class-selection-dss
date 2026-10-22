"""Train multi-label subject-interest models from the 10k source dataset.

The yes/no columns are treated as recorded interest labels, not confirmed
enrollments. CatBoostClassifier and SVC predict nine binary interests; the
interface ranks their probabilities and displays the top four.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dulieuhocsinh_10000.csv"
MODEL_DIR = ROOT / "models"
RANDOM_SEED = 42

TARGET_LABELS = {
    "Vật lý": "Vật lí",
    "Hóa học": "Hóa học",
    "Sinh học": "Sinh học",
    "Công nghệ": "Công nghệ",
    "Tin học": "Tin học",
    "Địa lý": "Địa lí",
    "Giáo dục kinh tế và pháp luật": "Giáo dục kinh tế và pháp luật",
    "Âm nhạc": "Âm nhạc",
    "Mỹ thuật": "Mĩ thuật",
}


def binary_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probability)),
    }


def positive_probability(model, features: pd.DataFrame) -> np.ndarray:
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    positive_index = int(np.flatnonzero(np.asarray(classes) == 1)[0])
    return model.predict_proba(features)[:, positive_index]


def top_four_match(labels: pd.DataFrame, probabilities: np.ndarray) -> float:
    top_indices = np.argpartition(-probabilities, kth=3, axis=1)[:, :4]
    matched = np.take_along_axis(labels.to_numpy(dtype=int), top_indices, axis=1)
    return float(matched.sum(axis=1).mean() / 4)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy bộ dữ liệu: {DATA_PATH}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    target_cols = list(TARGET_LABELS)
    feature_cols = [
        column for column in df.columns
        if column not in target_cols and column != "ma_hoc_sinh"
    ]
    if len(feature_cols) != 50:
        raise ValueError(f"Mong đợi 50 đặc trưng điểm học tập, nhận được {len(feature_cols)}.")
    if df["ma_hoc_sinh"].duplicated().any():
        raise ValueError("Mã học sinh bị trùng; cần xử lý nhóm trước khi chia tập.")

    labels = df[target_cols].apply(
        lambda column: column.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    )
    if labels.isna().any().any():
        raise ValueError("Có nhãn yes/no không hợp lệ.")
    labels = labels.astype(int)
    features = df[feature_cols].apply(pd.to_numeric, errors="raise")
    if features.isna().any().any():
        raise ValueError("Dữ liệu đặc trưng có giá trị thiếu.")

    # Chia theo hồ sơ gốc; mỗi học sinh chỉ có một dòng trong tệp 10k.
    all_indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        all_indices, test_size=0.20, random_state=RANDOM_SEED, shuffle=True
    )
    train_idx, validation_idx = train_test_split(
        train_val_idx, test_size=0.25, random_state=RANDOM_SEED, shuffle=True
    )
    X_train, X_validation, X_test = (
        features.iloc[train_idx], features.iloc[validation_idx], features.iloc[test_idx]
    )
    y_train, y_validation, y_test = (
        labels.iloc[train_idx], labels.iloc[validation_idx], labels.iloc[test_idx]
    )

    summary: dict[str, object] = {
        "dataset": DATA_PATH.name,
        "dataset_kind": "simulated_or_unverified_source",
        "label_semantics": (
            "Recorded yes/no interest fields. The dataset documentation does not "
            "confirm that these values are actual Grade 10 course registrations."
        ),
        "target_scale": ["no", "yes"],
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "test_rows": int(len(test_idx)),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "targets": [
            {"key": column, "label": TARGET_LABELS[column]}
            for column in target_cols
        ],
        "random_seed": RANDOM_SEED,
        "top_four_rule": (
            "Return the four subjects with the largest per-subject yes probabilities. "
            "This fixed top-four rule does not imply the training labels were four actual selections."
        ),
        "algorithms": {},
    }

    probability_sets: dict[str, np.ndarray] = {}
    model_specs = {
        "catboost": "CatBoostClassifier",
        "svc": "SVC (RBF, scaled inputs)",
    }
    for algorithm, model_name in model_specs.items():
        per_target: dict[str, object] = {}
        test_probability_columns: list[np.ndarray] = []

        for target_index, target_col in enumerate(target_cols):
            if algorithm == "catboost":
                model = CatBoostClassifier(
                    iterations=700,
                    depth=6,
                    learning_rate=0.05,
                    loss_function="Logloss",
                    eval_metric="F1",
                    random_seed=RANDOM_SEED,
                    verbose=False,
                    allow_writing_files=False,
                )
                model.fit(
                    X_train,
                    y_train[target_col],
                    eval_set=(X_validation, y_validation[target_col]),
                    early_stopping_rounds=70,
                    verbose=False,
                )
                artifact = MODEL_DIR / f"elective_catboost_{target_index + 1:02d}.cbm"
                model.save_model(str(artifact), format="cbm")
                selected_iteration = int(model.get_best_iteration())
                warning_messages: list[str] = []
            else:
                model = make_pipeline(
                    StandardScaler(),
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        random_state=RANDOM_SEED,
                    ),
                )
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    model.fit(X_train, y_train[target_col])
                artifact = MODEL_DIR / f"elective_svc_{target_index + 1:02d}.joblib"
                joblib.dump(model, artifact)
                selected_iteration = None
                warning_messages = [str(item.message) for item in captured]

            test_probability = positive_probability(model, X_test)
            validation_probability = positive_probability(model, X_validation)
            test_probability_columns.append(test_probability)
            per_target[target_col] = {
                "label": TARGET_LABELS[target_col],
                "artifact": artifact.name,
                "test_metrics": binary_metrics(y_test[target_col].to_numpy(), test_probability),
                "validation_metrics": binary_metrics(
                    y_validation[target_col].to_numpy(), validation_probability
                ),
                "positive_rate_test": float(y_test[target_col].mean()),
                "selected_iteration": selected_iteration,
                "warnings": warning_messages,
            }

        test_probabilities = np.column_stack(test_probability_columns)
        probability_sets[algorithm] = test_probabilities
        summary["algorithms"][algorithm] = {
            "label": model_name,
            "targets": per_target,
            "top_four_mean_interest_match": top_four_match(y_test, test_probabilities),
            "exact_match_accuracy_at_0_5": float(
                np.mean((test_probabilities >= 0.5) == y_test.to_numpy(dtype=bool))
            ),
        }
        print(
            f"{model_name}: top-4 mean interest match="
            f"{summary['algorithms'][algorithm]['top_four_mean_interest_match']:.3f}; "
            f"exact-match@0.5={summary['algorithms'][algorithm]['exact_match_accuracy_at_0_5']:.3f}"
        )
        for target_col in target_cols:
            metrics = summary["algorithms"][algorithm]["targets"][target_col]["test_metrics"]
            print(
                f"  {TARGET_LABELS[target_col]}: F1={metrics['f1']:.3f}, "
                f"AUC={metrics['roc_auc']:.3f}, accuracy={metrics['accuracy']:.3f}"
            )

    # Save unlabeled feature examples from the held-out split for the web form;
    # targets are included only for a clearly marked comparison with the recorded yes/no fields.
    sample_rows = df.iloc[test_idx].sample(n=5, random_state=RANDOM_SEED)
    summary["demo_profiles"] = [
        {
            "id": index,
            "label": f"Hồ sơ mẫu {index}",
            "features": {
                column: float(row[column])
                for column in feature_cols
            },
            "recorded_interest_labels": {
                column: int(labels.loc[row.name, column])
                for column in target_cols
            },
        }
        for index, (_, row) in enumerate(sample_rows.iterrows(), start=1)
    ]
    summary["defaults"] = {
        column: round(float(features.iloc[train_idx][column].median()), 2)
        for column in feature_cols
    }
    summary["feature_groups"] = [
        {
            "key": base,
            "label": base.replace("_", " ").capitalize(),
            "features": [column for column in feature_cols if column.startswith(base + "_lop")],
        }
        for base in dict.fromkeys(column.rsplit("_lop", 1)[0] for column in feature_cols)
    ]
    (MODEL_DIR / "elective_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Đã lưu 18 mô hình phân loại và models/elective_metadata.json")


if __name__ == "__main__":
    main()
