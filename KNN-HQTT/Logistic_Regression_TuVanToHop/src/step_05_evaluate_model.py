"""Đánh giá từng mô hình nhị phân trên tập kiểm tra 20%."""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import (
    MODEL_PATH,
    STEP_03_DIR,
    STEP_05_DIR,
    load_metadata,
    probability_frame,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Chưa có mô hình đã huấn luyện. Hãy chạy bước 4 trước."
        )
    X_test_path = STEP_03_DIR / "X_test.csv"
    y_test_path = STEP_03_DIR / "y_test.csv"
    if not X_test_path.is_file() or not y_test_path.is_file():
        raise FileNotFoundError("Thiếu tập kiểm tra bước 3. Hãy chạy lại bước 3.")

    metadata = load_metadata()
    X_test = pd.read_csv(X_test_path, encoding="utf-8-sig")
    y_test = pd.read_csv(y_test_path, encoding="utf-8-sig")
    labels = metadata["label_columns"]
    if X_test.columns.tolist() != metadata["feature_columns"]:
        raise ValueError("Cột X_test không khớp metadata mô hình.")
    if y_test.columns.tolist() != labels:
        raise ValueError("Cột y_test không khớp metadata mô hình.")

    pipeline = joblib.load(MODEL_PATH)
    probabilities, predictions = probability_frame(pipeline, X_test, labels)
    report_rows: list[dict[str, object]] = []
    detail_lines: list[str] = [
        "BÁO CÁO ĐÁNH GIÁ TRÊN TẬP KIỂM TRA",
        f"Số mẫu kiểm tra: {len(X_test)}",
        "",
    ]

    for label in labels:
        actual = y_test[label].astype(int)
        predicted = predictions[label].astype(int)
        probability = probabilities[label]
        matrix = confusion_matrix(actual, predicted, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        try:
            roc_auc = float(roc_auc_score(actual, probability))
        except ValueError:
            roc_auc = float("nan")

        report_rows.append(
            {
                "nhan": label,
                "accuracy": accuracy_score(actual, predicted),
                "precision": precision_score(
                    actual, predicted, zero_division=0
                ),
                "recall": recall_score(actual, predicted, zero_division=0),
                "f1_score": f1_score(actual, predicted, zero_division=0),
                "roc_auc": roc_auc,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "test_yes": int(actual.sum()),
                "test_no": int((actual == 0).sum()),
                "ty_le_yes_test": float(actual.mean()),
            }
        )
        detail_lines.extend(
            [
                f"NHÃN: {label}",
                classification_report(
                    actual,
                    predicted,
                    labels=[0, 1],
                    target_names=["no (0)", "yes (1)"],
                    zero_division=0,
                    digits=4,
                ).rstrip(),
                f"Confusion Matrix [[TN, FP], [FN, TP]]: {matrix.tolist()}",
                f"ROC-AUC: {roc_auc:.4f}" if np.isfinite(roc_auc) else "ROC-AUC: không tính được",
                "",
            ]
        )

    STEP_05_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(report_rows)
    metrics.to_csv(
        STEP_05_DIR / "chi_so_danh_gia.csv", index=False, encoding="utf-8-sig"
    )
    detail_lines.extend(
        [
            "Ý NGHĨA CHỈ SỐ",
            "- Accuracy: tỷ lệ dự đoán đúng tổng thể; có thể gây hiểu lầm nếu nhãn mất cân bằng.",
            "- Precision: trong các dự đoán yes, tỷ lệ thực sự là yes.",
            "- Recall: trong các trường hợp yes thực tế, tỷ lệ được mô hình tìm ra.",
            "- F1-score: trung bình điều hòa Precision và Recall.",
            "- ROC-AUC: khả năng xếp hạng yes cao hơn no trên nhiều ngưỡng; chỉ tính khi test có đủ hai lớp.",
            "- Confusion Matrix: TN đúng no, FP dự đoán yes sai, FN bỏ sót yes, TP đúng yes.",
            "- Điểm xác suất là ước lượng của mô hình, không mặc nhiên là xác suất thực tế đã hiệu chỉnh.",
            "- Kết quả dự đoán xu hướng lựa chọn trong nhãn dữ liệu, không đo năng lực hay mức độ phù hợp môn học.",
            "- Không kết luận mô hình tốt chỉ từ một chỉ số; cần xem từng nhãn và đối chiếu mục tiêu sử dụng.",
        ]
    )
    report_path = STEP_05_DIR / "bao_cao_danh_gia.txt"
    report_path.write_text("\n".join(detail_lines) + "\n", encoding="utf-8")

    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nĐã lưu:")
    print(f"- {STEP_05_DIR / 'chi_so_danh_gia.csv'}")
    print(f"- {report_path}")
    print("\nKhông kết luận mô hình tốt chỉ dựa trên Accuracy.")


if __name__ == "__main__":
    main()
