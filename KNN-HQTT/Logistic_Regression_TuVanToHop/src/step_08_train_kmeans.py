"""Chọn số cụm và huấn luyện K-Means chỉ bằng 80% tập huấn luyện."""

from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    KMEANS_METADATA_PATH,
    KMEANS_MODEL_PATH,
    KMEANS_OUTPUT_DIR,
    STEP_03_DIR,
)


K_CANDIDATES = range(2, 9)
RANDOM_STATE = 42
SILHOUETTE_SAMPLE_SIZE = 2000


def make_pipeline(cluster_count: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=cluster_count,
                    random_state=RANDOM_STATE,
                    n_init=20,
                ),
            ),
        ]
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    X_train_path = STEP_03_DIR / "X_train.csv"
    split_metadata_path = STEP_03_DIR / "dataset_metadata.json"
    for path in (X_train_path, split_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Thiếu {path.name}; hãy chạy bước 3 để tạo tập 80/20 trước."
            )

    X_train = pd.read_csv(X_train_path, encoding="utf-8-sig")
    with split_metadata_path.open(encoding="utf-8") as file:
        split_metadata = json.load(file)
    feature_columns = split_metadata["feature_columns"]
    if X_train.columns.tolist() != feature_columns:
        raise ValueError("Cột X_train không khớp metadata của bước 3.")
    if len(X_train) <= max(K_CANDIDATES):
        raise ValueError("Tập huấn luyện quá nhỏ để thử các số cụm đã cấu hình.")

    scores: list[dict[str, object]] = []
    best_k: int | None = None
    best_score = -1.0
    sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(X_train))

    for cluster_count in K_CANDIDATES:
        candidate = make_pipeline(cluster_count)
        candidate.fit(X_train)
        scaled_train = candidate.named_steps["scaler"].transform(
            candidate.named_steps["imputer"].transform(X_train)
        )
        assignments = candidate.named_steps["kmeans"].labels_
        distinct_clusters = np.unique(assignments)
        if len(distinct_clusters) < 2:
            raise RuntimeError(
                f"K-Means với k={cluster_count} chỉ tạo một cụm; "
                "không thể tính silhouette."
            )
        score = float(
            silhouette_score(
                scaled_train,
                assignments,
                sample_size=sample_size,
                random_state=RANDOM_STATE,
            )
        )
        scores.append(
            {
                "so_cum_k": cluster_count,
                "silhouette_train_mau": score,
                "inertia_train": float(candidate.named_steps["kmeans"].inertia_),
                "so_cum_thuc_te": len(distinct_clusters),
            }
        )
        if score > best_score:
            best_score = score
            best_k = cluster_count

    if best_k is None:
        raise RuntimeError("Không chọn được số cụm từ kết quả huấn luyện.")

    final_pipeline = make_pipeline(best_k)
    final_pipeline.fit(X_train)
    cluster_counts = pd.Series(final_pipeline.named_steps["kmeans"].labels_).value_counts()
    cluster_sizes = {
        str(cluster): int(cluster_counts.get(cluster, 0))
        for cluster in range(best_k)
    }

    KMEANS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    KMEANS_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    score_path = KMEANS_OUTPUT_DIR / "chon_so_cum_tren_train.csv"
    pd.DataFrame(scores).to_csv(score_path, index=False, encoding="utf-8-sig")
    joblib.dump(final_pipeline, KMEANS_MODEL_PATH)

    metadata = {
        "model_type": "KMeans",
        "feature_columns": feature_columns,
        "n_clusters": best_k,
        "random_state": RANDOM_STATE,
        "n_init": 20,
        "selection_metric": "silhouette_score sampled from training set",
        "selected_train_silhouette": best_score,
        "silhouette_sample_size": sample_size,
        "training_samples": len(X_train),
        "train_cluster_sizes": cluster_sizes,
        "preprocessing": [
            "SimpleImputer(strategy='median')",
            "StandardScaler()",
        ],
        "model_path": str(
            KMEANS_MODEL_PATH.relative_to(KMEANS_MODEL_PATH.parents[1])
        ),
        "split_random_state": split_metadata["random_state"],
        "split_test_size": split_metadata["test_size"],
    }
    with KMEANS_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    report_lines = [
        "HUẤN LUYỆN K-MEANS HOÀN TẤT",
        f"Số mẫu train: {len(X_train)}",
        f"Số đặc trưng điểm: {len(feature_columns)}",
        f"Số cụm được chọn: {best_k}",
        f"Silhouette train (mẫu tối đa {sample_size}): {best_score:.4f}",
        f"Kích thước cụm train: {cluster_sizes}",
        "Chọn k bằng silhouette trên tập huấn luyện; tập test không được dùng "
        "để chọn k hoặc fit imputer/scaler.",
        "K-Means là phân cụm không giám sát, không dự đoán lựa chọn môn.",
        f"Mô hình lưu tại: {KMEANS_MODEL_PATH}",
        f"Điểm so sánh k lưu tại: {score_path}",
    ]
    (KMEANS_OUTPUT_DIR / "bao_cao_huan_luyen.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
