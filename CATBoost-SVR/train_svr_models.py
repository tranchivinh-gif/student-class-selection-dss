"""Train one scaled RBF-SVR pipeline per simulated exam subject."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dulieuhocsinh_50000_mo_phong.csv"
MODEL_DIR = ROOT / "models"
CATBOOST_METADATA_PATH = MODEL_DIR / "metadata.json"
RANDOM_SEED = 42
TRAIN_GROUPS = 300
PARAMETERS = {
    "kernel": "rbf",
    "C": 10.0,
    "epsilon": 0.1,
    "gamma": "scale",
    "tol": 0.01,
    "max_iter": 20_000,
}


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy bộ dữ liệu: {DATA_PATH}")
    if not CATBOOST_METADATA_PATH.exists():
        raise FileNotFoundError("Hãy chạy train_models.py để tạo schema mô hình trước.")

    catboost_metadata = json.loads(CATBOOST_METADATA_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    train_df = df.loc[df["tap_du_lieu"] == "train"].copy()
    validation_df = df.loc[df["tap_du_lieu"] == "validation"].copy()
    test_df = df.loc[df["tap_du_lieu"] == "test"].copy()

    group_sets = {
        name: set(df.loc[df["tap_du_lieu"] == name, "nhom_ho_so_goc"].unique())
        for name in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if group_sets[left] & group_sets[right]:
            raise ValueError(f"Nhóm hồ sơ bị chia qua {left}/{right}.")

    selected_groups = (
        train_df["nhom_ho_so_goc"].drop_duplicates().sample(n=TRAIN_GROUPS, random_state=RANDOM_SEED)
    )
    svr_train_df = train_df.loc[train_df["nhom_ho_so_goc"].isin(selected_groups)].copy()
    expected_rows = TRAIN_GROUPS * 5
    if len(svr_train_df) != expected_rows:
        raise ValueError(f"Mong đợi {expected_rows} dòng train theo nhóm; nhận được {len(svr_train_df)}.")

    svr_metadata: dict[str, object] = {
        "dataset": DATA_PATH.name,
        "dataset_kind": "simulated",
        "train_rows": int(len(svr_train_df)),
        "train_groups": TRAIN_GROUPS,
        "full_train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "random_seed": RANDOM_SEED,
        "parameters": PARAMETERS,
        "subjects": {},
        "limitations": (
            "SVR được huấn luyện trên 1.500 dòng thuộc 300 nhóm lấy mẫu từ tập train "
            "để giới hạn thời gian huấn luyện. Dữ liệu và nhãn đều là mô phỏng."
        ),
    }

    for subject_key, catboost_spec in catboost_metadata["subjects"].items():
        feature_cols = catboost_spec["features"]
        target = catboost_spec["target"]
        baseline = catboost_spec["baseline"]
        model = make_pipeline(
            StandardScaler(),
            SVR(
                kernel=PARAMETERS["kernel"],
                C=PARAMETERS["C"],
                epsilon=PARAMETERS["epsilon"],
                gamma=PARAMETERS["gamma"],
                tol=PARAMETERS["tol"],
                max_iter=PARAMETERS["max_iter"],
            ),
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(svr_train_df[feature_cols], svr_train_df[target])

        svr_estimator = model.named_steps["svr"]
        if svr_estimator.fit_status_ != 0:
            raise RuntimeError(
                f"SVR chưa hội tụ cho {catboost_spec['label']} "
                f"(max_iter={PARAMETERS['max_iter']}); không lưu mô hình này."
            )

        test_pred = model.predict(test_df[feature_cols])
        validation_pred = model.predict(validation_df[feature_cols])
        test_metrics = calculate_metrics(test_df[target], test_pred)
        validation_metrics = calculate_metrics(validation_df[target], validation_pred)
        baseline_metrics = calculate_metrics(test_df[target], test_df[baseline].to_numpy())

        model_path = MODEL_DIR / f"svr_{subject_key}.joblib"
        joblib.dump(model, model_path)
        svr_metadata["subjects"][subject_key] = {
            "label": catboost_spec["label"],
            "target": target,
            "mock_score": catboost_spec["mock_score"],
            "baseline": baseline,
            "features": feature_cols,
            "model_file": model_path.name,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "baseline_test_metrics": baseline_metrics,
            "support_vectors": int(len(svr_estimator.support_)),
            "converged": True,
            "warnings": [str(item.message) for item in captured],
        }
        print(
            f"{catboost_spec['label']}: MAE={test_metrics['mae']:.3f}, "
            f"RMSE={test_metrics['rmse']:.3f}, R²={test_metrics['r2']:.3f}; "
            f"support vectors={len(svr_estimator.support_):,}"
        )

    metadata_path = MODEL_DIR / "svr_metadata.json"
    metadata_path.write_text(json.dumps(svr_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu 3 pipeline SVR và metadata trong {MODEL_DIR}")


if __name__ == "__main__":
    main()
