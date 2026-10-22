"""Train one CatBoostRegressor per simulated exam subject."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dulieuhocsinh_50000_mo_phong.csv"
MODEL_DIR = ROOT / "models"
RANDOM_SEED = 42

SUBJECTS = {
    "toan": {
        "label": "Toán",
        "target": "diem_thi_toan_mo_phong",
        "mock_score": "diem_thi_thu_toan",
        "baseline": "toan_lop9",
    },
    "ngu_van": {
        "label": "Ngữ văn",
        "target": "diem_thi_ngu_van_mo_phong",
        "mock_score": "diem_thi_thu_ngu_van",
        "baseline": "ngu_van_lop9",
    },
    "ngoai_ngu": {
        "label": "Ngoại ngữ",
        "target": "diem_thi_ngoai_ngu_mo_phong",
        "mock_score": "diem_thi_thu_ngoai_ngu",
        "baseline": "ngoai_ngu_lop9",
    },
}

SUBJECT_LABELS = {
    "toan": "Toán",
    "ngu_van": "Ngữ văn",
    "ngoai_ngu": "Ngoại ngữ",
    "khoa_hoc_tu_nhien": "Khoa học tự nhiên",
    "lich_su_dia_ly": "Lịch sử và Địa lý",
    "giao_duc_cong_dan": "Giáo dục công dân",
    "tin_hoc": "Tin học",
    "cong_nghe": "Công nghệ",
    "giao_duc_the_chat": "Giáo dục thể chất",
    "am_nhac": "Âm nhạc",
    "mi_thuat": "Mỹ thuật",
    "noi_dung_giao_duc_dia_phuong": "Nội dung giáo dục địa phương",
    "hoat_dong_trai_nghiem_huong_nghiep": "Hoạt động trải nghiệm, hướng nghiệp",
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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Chỉ dùng điểm học tập và thông tin được giả định có trước kỳ thi.
    grade_features = [
        column
        for column in df.columns
        if column.endswith(tuple(f"_lop{grade}" for grade in range(6, 10)))
    ]
    common_features = grade_features + ["gio_tu_hoc_tuan", "ti_le_chuyen_can"]

    partitions = {
        name: df.loc[df["tap_du_lieu"] == name].copy()
        for name in ("train", "validation", "test")
    }
    group_sets = {
        name: set(part["nhom_ho_so_goc"].unique())
        for name, part in partitions.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise ValueError(f"Nhóm hồ sơ bị chia qua {left}/{right}: {len(overlap)} nhóm")

    train_df = partitions["train"]
    validation_df = partitions["validation"]
    test_df = partitions["test"]

    metadata: dict[str, object] = {
        "dataset": DATA_PATH.name,
        "dataset_kind": "simulated",
        "target_scale": [0, 10],
        "split_counts": {name: int(len(part)) for name, part in partitions.items()},
        "split_group_counts": {name: int(len(group_sets[name])) for name in group_sets},
        "random_seed": RANDOM_SEED,
        "features_common": common_features,
        "subjects": {},
        "defaults": {},
        "feature_groups": [],
        "demo_profiles": [],
        "limitations": (
            "Các nhãn điểm thi và hồ sơ trong bộ dữ liệu là mô phỏng; kết quả không phải "
            "bằng chứng hiệu quả trên tuyển sinh hoặc học sinh thực tế."
        ),
    }

    for feature in common_features:
        if feature.endswith(tuple(f"_lop{grade}" for grade in range(6, 10))):
            base, _grade = feature.rsplit("_lop", 1)
            group = next((item for item in metadata["feature_groups"] if item["key"] == base), None)
            if group is None:
                group = {"key": base, "label": SUBJECT_LABELS.get(base, base), "features": []}
                metadata["feature_groups"].append(group)
            group["features"].append(feature)

    metadata["defaults"] = {
        feature: round(float(train_df[feature].median()), 3)
        for feature in common_features
    }
    metadata["defaults"]["gio_tu_hoc_tuan"] = round(float(train_df["gio_tu_hoc_tuan"].median()), 1)
    metadata["defaults"]["ti_le_chuyen_can"] = round(float(train_df["ti_le_chuyen_can"].median()), 3)

    config = {
        "iterations": 1000,
        "depth": 8,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": RANDOM_SEED,
        "early_stopping_rounds": 100,
    }
    for subject_key, spec in SUBJECTS.items():
        feature_cols = common_features + [spec["mock_score"]]
        model = CatBoostRegressor(
            iterations=config["iterations"],
            depth=config["depth"],
            learning_rate=config["learning_rate"],
            loss_function=config["loss_function"],
            eval_metric=config["eval_metric"],
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            train_df[feature_cols],
            train_df[spec["target"]],
            eval_set=(validation_df[feature_cols], validation_df[spec["target"]]),
            early_stopping_rounds=config["early_stopping_rounds"],
            verbose=False,
        )

        test_pred = model.predict(test_df[feature_cols])
        test_metrics = calculate_metrics(test_df[spec["target"]], test_pred)
        baseline_metrics = calculate_metrics(test_df[spec["target"]], test_df[spec["baseline"]].to_numpy())
        model_path = MODEL_DIR / f"catboost_{subject_key}.cbm"
        model.save_model(str(model_path), format="cbm")

        subject_metadata = {
            **spec,
            "features": feature_cols,
            "model_file": model_path.name,
            "best_iteration": int(model.get_best_iteration()),
            "tree_count": int(model.tree_count_),
            "test_metrics": test_metrics,
            "baseline_test_metrics": baseline_metrics,
            "top_features": [
                {"name": feature_cols[index], "importance": float(importance)}
                for index, importance in sorted(
                    enumerate(model.feature_importances_), key=lambda pair: pair[1], reverse=True
                )[:10]
            ],
        }
        metadata["subjects"][subject_key] = subject_metadata
        print(
            f"{spec['label']}: MAE={test_metrics['mae']:.3f}, "
            f"RMSE={test_metrics['rmse']:.3f}, R²={test_metrics['r2']:.3f}; "
            f"mốc MAE={baseline_metrics['mae']:.3f}; cây={model.tree_count_}"
        )

    # Lấy hồ sơ test thuộc các nhóm nguồn khác nhau để thử giao diện.
    demo_rows = (
        test_df.drop_duplicates("nhom_ho_so_goc")
        .sample(n=5, random_state=RANDOM_SEED)
        .sort_values("ma_hoc_sinh")
    )
    for index, (_, row) in enumerate(demo_rows.iterrows(), start=1):
        profile_features = {feature: float(row[feature]) for feature in common_features}
        targets = {key: float(row[spec["target"]]) for key, spec in SUBJECTS.items()}
        mock_scores = {key: float(row[spec["mock_score"]]) for key, spec in SUBJECTS.items()}
        metadata["demo_profiles"].append(
            {
                "id": index,
                "label": f"Hồ sơ mẫu {index}",
                "features": profile_features,
                "mock_scores": mock_scores,
                "targets": targets,
            }
        )

    metadata_path = MODEL_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu 3 mô hình và metadata trong {MODEL_DIR}")


if __name__ == "__main__":
    main()
