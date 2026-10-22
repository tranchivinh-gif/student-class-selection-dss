"""Evaluate saved test predictions without retraining or tuning on the test set."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_DIR / "reports"
MODEL_PATH = PROJECT_DIR / "models" / "xgboost_model.pkl"
PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"
METRICS_PATH = REPORTS_DIR / "evaluation.json"
REPORT_PATH = REPORTS_DIR / "evaluation_report.md"


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
) -> dict[str, Any]:
    """Calculate per-label, micro/macro, and exact-match metrics."""
    truth = np.asarray(y_true, dtype=int)
    predictions = np.asarray(y_pred, dtype=int)
    if truth.ndim != 2 or predictions.shape != truth.shape:
        raise ValueError("y_true và y_pred phải là ma trận cùng kích thước.")
    if truth.shape[1] != len(target_columns):
        raise ValueError("Số cột nhãn không khớp target_columns.")

    per_label: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(target_columns):
        per_label[label] = {
            "precision": float(
                precision_score(truth[:, index], predictions[:, index], zero_division=0)
            ),
            "recall": float(
                recall_score(truth[:, index], predictions[:, index], zero_division=0)
            ),
            "f1": float(
                f1_score(truth[:, index], predictions[:, index], zero_division=0)
            ),
            "support": int(truth[:, index].sum()),
            "samples": int(truth.shape[0]),
        }

    return {
        "per_label": per_label,
        "micro_precision": float(
            precision_score(truth, predictions, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(truth, predictions, average="micro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(truth, predictions, average="micro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(truth, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(truth, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(truth, predictions, average="macro", zero_division=0)
        ),
        "exact_match_accuracy": float(accuracy_score(truth, predictions)),
        "sample_count": int(truth.shape[0]),
    }


def write_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
    best_params: dict[str, Any],
    validation_trials: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
    estimated_feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    metrics = calculate_metrics(y_true, y_pred, target_columns)
    result = {
        "test_metrics": metrics,
        "test_exactly_four_rate": float((np.asarray(y_true, dtype=int).sum(axis=1) == 4).mean()),
        "baseline_metrics": baseline_metrics,
        "selected_parameters": best_params,
        "validation_trials": validation_trials,
        "estimated_feature_columns": estimated_feature_columns or [],
        "test_set_policy": (
            "Test được giữ riêng trong lúc chọn tham số; chỉ đánh giá sau khi "
            "chọn cấu hình bằng Validation."
        ),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        render_report(result),
        encoding="utf-8",
    )
    return result


def render_report(result: dict[str, Any]) -> str:
    metrics = result["test_metrics"]
    baseline = result["baseline_metrics"]
    lines = [
        "# Báo cáo đánh giá XGBoost đa nhãn",
        "",
        f"- Số mẫu test: {metrics['sample_count']:,}",
        "- Mỗi học sinh được đề xuất đúng 4 môn có xác suất cao nhất.",
        f"- Tỷ lệ học sinh thực tế chọn đúng 4 môn: {result['test_exactly_four_rate']:.1%}. "
        "Đây là trần lý thuyết của Exact Match Accuracy khi luôn đề xuất 4 môn.",
        f"- Tham số chọn trên Validation: `{result['selected_parameters']}`",
        "- Test không được dùng để chọn tham số.",
    ]
    if result["estimated_feature_columns"]:
        lines.extend(
            [
                "",
                "**CẢNH BÁO: kết quả dưới đây chỉ mang tính minh họa.** Đặc trưng "
                f"`{', '.join(result['estimated_feature_columns'])}` là điểm "
                "ước tính, không phải số liệu học tập thực tế; các chỉ số này "
                "không chứng minh hiệu năng khi dùng điểm thật.",
            ]
        )
    lines.extend(
        [
            "",
            "## Chỉ số tổng thể trên Test",
            "",
            "| Chỉ số | XGBoost | Baseline 4 môn phổ biến nhất trong Train |",
            "|---|---:|---:|",
        ]
    )
    for key, label in (
        ("micro_precision", "Micro Precision"),
        ("micro_recall", "Micro Recall"),
        ("micro_f1", "Micro F1"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
        ("exact_match_accuracy", "Exact Match Accuracy"),
    ):
        lines.append(f"| {label} | {metrics[key]:.4f} | {baseline[key]:.4f} |")
    lines.extend(
        [
            "",
            "## Kết quả theo nhãn",
            "",
            "| Môn | Precision | Recall | F1 | Số mẫu Yes trong Test |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, item in metrics["per_label"].items():
        lines.append(
            f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} "
            f"| {item['f1']:.4f} | {item['support']:,} |"
        )
    lines.extend(
        [
            "",
            "## Chọn tham số và giới hạn",
            "",
            "Cấu hình được chọn theo Macro F1 trên Validation, dùng Exact Match "
            "và Micro F1 để phá hòa; không lựa chọn dựa trên Test.",
            "",
            "| Cấu hình | Validation Macro F1 | Validation Micro F1 | Validation Exact Match |",
            "|---|---:|---:|---:|",
        ]
    )
    for trial in result["validation_trials"]:
        lines.append(
            f"| `{trial['parameters']}` | {trial['macro_f1']:.4f} "
            f"| {trial['micro_f1']:.4f} | {trial['exact_match_accuracy']:.4f} |"
        )
    weak_labels = [
        label
        for label, item in metrics["per_label"].items()
        if item["support"] < 30 or item["samples"] - item["support"] < 30
    ]
    if weak_labels:
        lines.extend(
            [
                "",
                "Cảnh báo: một số nhãn có ít hơn 30 mẫu ở một lớp trong Test: "
                + ", ".join(weak_labels)
                + ". Các chỉ số của những nhãn này kém ổn định.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Mỗi nhãn có ít nhất 30 mẫu ở cả hai lớp trong Test; "
                "không phát hiện nhãn hiếm theo ngưỡng 30 mẫu.",
            ]
        )
    lines.extend(
        [
            "",
            "Mô hình dự đoán lựa chọn quan sát được trong dữ liệu, không đánh giá "
            "mức độ phù hợp hoặc năng lực học tập của học sinh. Hiệu năng cần "
            "được đối chiếu với baseline và dữ liệu mới trước khi sử dụng thực tế.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import joblib

    if not MODEL_PATH.is_file() or not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            "Cần chạy train.py trước để tạo mô hình và dự đoán trên tập Test."
        )
    training_path = REPORTS_DIR / "training_report.json"
    if not training_path.is_file():
        raise FileNotFoundError(f"Thiếu báo cáo huấn luyện: {training_path}")
    artifact = joblib.load(MODEL_PATH)
    target_columns = artifact["target_columns"]
    test_frame = pd.read_csv(PREDICTIONS_PATH)
    actual_columns = [f"actual::{column}" for column in target_columns]
    predicted_columns = [f"predicted::{column}" for column in target_columns]
    missing = set(actual_columns + predicted_columns) - set(test_frame.columns)
    if missing:
        raise ValueError(f"Thiếu cột trong test predictions: {sorted(missing)}")
    y_true = test_frame[actual_columns].to_numpy(dtype=int)
    y_pred = test_frame[predicted_columns].to_numpy(dtype=int)
    training_report = json.loads(training_path.read_text(encoding="utf-8"))
    result = write_evaluation(
        y_true,
        y_pred,
        target_columns,
        artifact["best_params"],
        training_report["validation_trials"],
        training_report["baseline_metrics"],
        artifact.get("estimated_feature_columns", []),
    )
    print(f"Test Macro F1: {result['test_metrics']['macro_f1']:.4f}")
    print(f"Test Micro F1: {result['test_metrics']['micro_f1']:.4f}")
    print(f"Báo cáo: {REPORT_PATH}")


if __name__ == "__main__":
    main()
