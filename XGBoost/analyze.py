"""Analyze held-out predictions and feature importance without retraining."""

from __future__ import annotations

import sys
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from evaluate import PREDICTIONS_PATH
from preprocessing import PROJECT_DIR


REPORTS_DIR = PROJECT_DIR / "reports"
MODEL_PATH = PROJECT_DIR / "models" / "xgboost_model.pkl"
IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
IMPORTANCE_PLOT = REPORTS_DIR / "feature_importance.png"
DISTRIBUTION_PLOT = REPORTS_DIR / "prediction_distribution.png"
REPORT_PATH = REPORTS_DIR / "analysis_report.md"


def _feature_importances(artifact: dict[str, Any]) -> pd.DataFrame:
    features = artifact["feature_columns"]
    target_columns = artifact["target_columns"]
    estimators = artifact["classifier"].estimators_
    records = []
    for label, estimator in zip(target_columns, estimators, strict=True):
        importances = estimator.feature_importances_
        records.extend(
            {
                "feature": feature,
                "label": label,
                "importance": float(importance),
            }
            for feature, importance in zip(features, importances, strict=True)
        )
    return pd.DataFrame(records)


def _render_charts(
    importance: pd.DataFrame,
    predictions: pd.DataFrame,
    target_columns: list[str],
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    aggregated = (
        importance.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(15)
        .sort_values("importance")
    )
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(aggregated["feature"], aggregated["importance"], color="#2c7fb8")
    axis.set_title("Top 15 features by mean XGBoost importance")
    axis.set_xlabel("Mean feature importance across labels")
    figure.tight_layout()
    figure.savefig(IMPORTANCE_PLOT, dpi=160)
    plt.close(figure)

    actual_counts = [
        int(predictions[f"actual::{label}"].sum()) for label in target_columns
    ]
    predicted_counts = [
        int(predictions[f"predicted::{label}"].sum()) for label in target_columns
    ]
    positions = range(len(target_columns))
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        [position - 0.2 for position in positions],
        actual_counts,
        width=0.4,
        label="Actual Yes",
        color="#41ab5d",
    )
    axis.bar(
        [position + 0.2 for position in positions],
        predicted_counts,
        width=0.4,
        label="Predicted Yes",
        color="#f16913",
    )
    axis.set_xticks(list(positions), target_columns, rotation=35, ha="right")
    axis.set_ylabel("Number of positive predictions")
    axis.set_title("Actual vs predicted subject choices on the test set")
    axis.legend()
    figure.tight_layout()
    figure.savefig(DISTRIBUTION_PLOT, dpi=160)
    plt.close(figure)


def analyze() -> str:
    if not MODEL_PATH.is_file() or not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError("Cần chạy train.py trước khi phân tích.")
    artifact = joblib.load(MODEL_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    target_columns = artifact["target_columns"]
    required = {
        f"{kind}::{label}"
        for label in target_columns
        for kind in ("actual", "predicted")
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Thiếu cột kết quả test: {sorted(missing)}")

    importance = _feature_importances(artifact)
    importance.to_csv(IMPORTANCE_PATH, index=False, encoding="utf-8")
    _render_charts(importance, predictions, target_columns)

    aggregated = (
        importance.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
    )
    lines = [
        "# Báo cáo phân tích mô hình",
        "",
        f"- Mẫu phân tích: {len(predictions):,} kết quả trên tập Test.",
        "- Phân tích chỉ đọc mô hình đã huấn luyện và dự đoán Test; không huấn "
        "luyện lại hoặc thay đổi dữ liệu.",
    ]
    estimated_columns = artifact.get("estimated_feature_columns", [])
    if estimated_columns:
        lines.extend(
            [
                "",
                "**CẢNH BÁO: các phân tích dùng đặc trưng ước tính "
                f"({', '.join(estimated_columns)}), không phải điểm thật. "
                "Không xem đây là kết luận về hiệu năng thực tế.**",
            ]
        )
    lines.extend(
        [
            "",
            "## Tỷ lệ dự đoán và lỗi theo môn",
            "",
            "| Môn | Actual Yes | Predicted Yes | False Positive | False Negative | Tỷ lệ sai |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in target_columns:
        truth = predictions[f"actual::{label}"].astype(int)
        predicted = predictions[f"predicted::{label}"].astype(int)
        false_positive = int(((truth == 0) & (predicted == 1)).sum())
        false_negative = int(((truth == 1) & (predicted == 0)).sum())
        error_rate = float((truth != predicted).mean())
        lines.append(
            f"| {label} | {int(truth.sum()):,} | {int(predicted.sum()):,} "
            f"| {false_positive:,} | {false_negative:,} | {error_rate:.1%} |"
        )

    lines.extend(
        [
            "",
            "False Positive là dự đoán Yes khi nhãn thực tế No; False Negative "
            "là dự đoán No khi nhãn thực tế Yes.",
            "",
            "## Feature Importance",
            "",
            "Bảng xếp hạng dưới đây là trung bình importance của các mô hình "
            "nhị phân theo từng nhãn. Đây là mức đóng góp dựa trên chia nhánh "
            "trong mô hình, không chứng minh quan hệ nhân quả.",
            "",
            "| Đặc trưng điểm | Mean importance |",
            "|---|---:|",
        ]
    )
    for feature, value in aggregated.head(15).items():
        lines.append(f"| {feature} | {value:.6f} |")
    lines.extend(
        [
            "",
            "Biểu đồ: `feature_importance.png` và `prediction_distribution.png`.",
            "",
            "Kết quả và importance chỉ phản ánh dữ liệu thu thập được. Không sử "
            "dụng điểm importance hoặc xác suất như tư vấn môn học phù hợp cho "
            "từng học sinh.",
            "",
        ]
    )
    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(analyze())


if __name__ == "__main__":
    main()
