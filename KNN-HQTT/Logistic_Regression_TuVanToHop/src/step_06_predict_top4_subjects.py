"""Tạo dự đoán Top 4 cho một học sinh được chọn trong tập kiểm tra."""

from __future__ import annotations

import sys

import joblib
import pandas as pd

from common import (
    MODEL_PATH,
    STEP_03_DIR,
    build_ranked_results,
    compare_official_combinations,
    load_metadata,
    probability_frame,
    save_prediction_outputs,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Chưa có mô hình. Hãy chạy bước 4 trước.")
    test_path = STEP_03_DIR / "X_test.csv"
    if not test_path.is_file():
        raise FileNotFoundError("Chưa có X_test. Hãy chạy bước 3 trước.")

    metadata = load_metadata()
    X_test = pd.read_csv(test_path, encoding="utf-8-sig")
    if X_test.empty:
        raise ValueError("Tập kiểm tra rỗng, không có học sinh để dự đoán.")

    prompt = f"Chọn số thứ tự mẫu kiểm tra (0 đến {len(X_test) - 1}, mặc định 0): "
    raw_index = input(prompt).strip()
    try:
        row_index = int(raw_index) if raw_index else 0
    except ValueError as error:
        raise ValueError("Số thứ tự phải là số nguyên.") from error
    if not 0 <= row_index < len(X_test):
        raise ValueError(f"Số thứ tự phải nằm trong khoảng 0 đến {len(X_test) - 1}.")

    pipeline = joblib.load(MODEL_PATH)
    one_student = X_test.iloc[[row_index]]
    probabilities, predictions = probability_frame(
        pipeline, one_student, metadata["label_columns"]
    )
    ranked = build_ranked_results(probabilities.iloc[0], predictions.iloc[0])
    paths = save_prediction_outputs(ranked, output_prefix="mau_test")

    print("\nBẢNG XẾP HẠNG DỰ ĐOÁN")
    print(ranked.to_string(index=False, formatters={
        "diem_xac_suat_uoc_luong": lambda value: f"{value:.4f}"
    }))
    print("\nTOP 4 MÔN")
    print(ranked.head(4).to_string(index=False))
    comparison = compare_official_combinations(ranked.head(4)["ten_mon"].tolist())
    if comparison.empty:
        print(
            "\nChưa cấu hình 5 tổ hợp chính thức; không thể đối chiếu hoặc tự tạo "
            "tổ hợp mới. Hãy bổ sung danh sách chính xác vào src/subject_combinations.py."
        )
    else:
        print("\nĐỐI CHIẾU TỔ HỢP CHÍNH THỨC")
        print(comparison.to_string(index=False))
        if not comparison["khop_hoan_toan"].any():
            print(
                "Không khớp hoàn toàn. Tổ hợp đứng đầu chỉ là tổ hợp gần nhất "
                "theo số môn trùng, không phải kết luận phù hợp nhất."
            )
    print("\nĐã lưu:")
    for path in paths:
        print(f"- {path}")
    print(
        "\nCác giá trị là điểm xác suất mô hình ước lượng, không phải điểm môn, "
        "xác suất đỗ hay mức độ phù hợp thực tế."
    )


if __name__ == "__main__":
    main()
