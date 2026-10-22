"""Đánh giá cụm K-Means trên tập kiểm tra chưa dùng khi huấn luyện."""

from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from common import (
    KMEANS_METADATA_PATH,
    KMEANS_MODEL_PATH,
    KMEANS_OUTPUT_DIR,
    STEP_03_DIR,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    required_paths = [
        KMEANS_MODEL_PATH,
        KMEANS_METADATA_PATH,
        STEP_03_DIR / "X_test.csv",
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Thiếu {path.name}; hãy chạy bước 3 và bước 8 trước."
            )

    with KMEANS_METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)
    X_test = pd.read_csv(required_paths[2], encoding="utf-8-sig")
    features = metadata["feature_columns"]
    if X_test.columns.tolist() != features:
        raise ValueError("Cột X_test không khớp metadata K-Means.")

    pipeline = joblib.load(KMEANS_MODEL_PATH)
    predicted_clusters = pipeline.predict(X_test)
    transformed_test = pipeline[:-1].transform(X_test)
    unique_clusters = np.unique(predicted_clusters)
    if len(unique_clusters) < 2 or len(unique_clusters) >= len(X_test):
        test_silhouette = float("nan")
        silhouette_note = (
            "Không tính được silhouette: tập test cần có từ 2 cụm trở lên "
            "và ít cụm hơn số mẫu."
        )
    else:
        test_silhouette = float(
            silhouette_score(transformed_test, predicted_clusters)
        )
        silhouette_note = (
            "Silhouette mô tả mức độ gần cụm trong không gian điểm đã chuẩn hóa; "
            "không đo khả năng dự đoán môn lựa chọn."
        )

    cluster_counts = pd.Series(predicted_clusters).value_counts()
    report = pd.DataFrame(
        [
            {
                "so_mau_test": len(X_test),
                "so_cum_duoc_gan": len(unique_clusters),
                "silhouette_test": test_silhouette,
                "ghi_chu": silhouette_note,
            }
        ]
    )
    assignments = pd.DataFrame(
        {
            "so_thu_tu_mau_test": np.arange(len(X_test)),
            "cum_kmeans": predicted_clusters.astype(int),
        }
    )
    sizes = pd.DataFrame(
        [
            {"cum_kmeans": int(cluster), "so_mau_test": int(count)}
            for cluster, count in cluster_counts.items()
        ]
    ).sort_values("cum_kmeans")

    KMEANS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(
        KMEANS_OUTPUT_DIR / "danh_gia_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    assignments.to_csv(
        KMEANS_OUTPUT_DIR / "gan_cum_tap_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sizes.to_csv(
        KMEANS_OUTPUT_DIR / "kich_thuoc_cum_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "ĐÁNH GIÁ K-MEANS TRÊN TẬP TEST",
        f"Số mẫu test: {len(X_test)}",
        f"Số cụm được gán trong test: {len(unique_clusters)}",
        (
            f"Silhouette test: {test_silhouette:.4f}"
            if np.isfinite(test_silhouette)
            else "Silhouette test: không tính được"
        ),
        silhouette_note,
        "K-Means là học không giám sát nên không có Accuracy/Precision/Recall "
        "nếu không có nhãn cụm đúng do chuyên gia cung cấp.",
        "Mã cụm chỉ là nhãn định danh tùy ý, không hàm ý tốt/xấu hoặc phù hợp/không phù hợp.",
    ]
    (KMEANS_OUTPUT_DIR / "bao_cao_danh_gia.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"Kích thước cụm test:\n{sizes.to_string(index=False)}")


if __name__ == "__main__":
    main()
