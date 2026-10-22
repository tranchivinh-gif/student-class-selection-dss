"""Nhập điểm học sinh mới từ Terminal và dự đoán các nhãn lựa chọn."""

from __future__ import annotations

import sys

import joblib
import pandas as pd

from common import (
    MODEL_PATH,
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
    metadata = load_metadata()
    features = metadata["feature_columns"]
    print(
        f"Nhập đủ {len(features)} điểm theo đúng tên cột. "
        "Giá trị phải nằm trong khoảng 0 đến 10."
    )

    values: dict[str, float] = {}
    for feature in features:
        while True:
            raw_value = input(f"{feature}: ").strip()
            try:
                value = float(raw_value)
            except ValueError:
                print("Lỗi: hãy nhập một số, ví dụ 7 hoặc 8.5.")
                continue
            if not 0 <= value <= 10:
                print("Lỗi: điểm phải nằm trong khoảng 0 đến 10.")
                continue
            values[feature] = value
            break

    student = pd.DataFrame([values], columns=features)
    pipeline = joblib.load(MODEL_PATH)
    probabilities, predictions = probability_frame(
        pipeline, student, metadata["label_columns"]
    )
    ranked = build_ranked_results(probabilities.iloc[0], predictions.iloc[0])
    paths = save_prediction_outputs(ranked, output_prefix="hoc_sinh_moi")

    print("\nBẢNG XẾP HẠNG DỰ ĐOÁN")
    print(
        ranked.to_string(
            index=False,
            formatters={
                "diem_xac_suat_uoc_luong": lambda value: f"{value:.4f}"
            },
        )
    )
    print("\nTOP 4 MÔN")
    print(ranked.head(4).to_string(index=False))

    comparison = compare_official_combinations(ranked.head(4)["ten_mon"].tolist())
    if comparison.empty:
        print(
            "\nChưa cấu hình tổ hợp chính thức. Không đề xuất tổ hợp mới; "
            "hãy cung cấp 5 tổ hợp chính xác để bật đối chiếu."
        )
    else:
        print("\nĐỐI CHIẾU TỔ HỢP CHÍNH THỨC")
        print(comparison.to_string(index=False))
        if not comparison["khop_hoan_toan"].any():
            print(
                "Không khớp hoàn toàn; tổ hợp gần nhất chỉ được xếp theo số môn trùng."
            )

    print("\nĐã lưu kết quả:")
    for path in paths:
        print(f"- {path}")
    print(
        "Đây là điểm xác suất mô hình ước lượng cho xu hướng nhãn yes/no, "
        "không phải xác suất đỗ hay đánh giá mức độ phù hợp môn học."
    )


if __name__ == "__main__":
    main()
