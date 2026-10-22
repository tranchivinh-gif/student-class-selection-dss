"""Các đường dẫn và hàm dùng chung giữa các bước của dự án."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "du_lieu_da_lam_sach.csv"
STEP_03_DIR = PROJECT_ROOT / "outputs" / "step_03"
STEP_04_DIR = PROJECT_ROOT / "outputs" / "step_04"
STEP_05_DIR = PROJECT_ROOT / "outputs" / "step_05"
STEP_06_DIR = PROJECT_ROOT / "outputs" / "step_06"
KMEANS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "kmeans"
MODELS_DIR = PROJECT_ROOT / "models"
METADATA_PATH = MODELS_DIR / "metadata.json"
MODEL_PATH = MODELS_DIR / "logistic_regression_pipeline.joblib"
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_pipeline.joblib"
KMEANS_METADATA_PATH = MODELS_DIR / "kmeans_metadata.json"
SCORE_COLUMN_PATTERN = re.compile(r"^.+_lop[6-9]$", re.IGNORECASE)
ID_COLUMN_CANDIDATES = {"ma_hoc_sinh", "student_id", "studentid", "id"}


def identify_score_columns(data: pd.DataFrame) -> list[str]:
    """Nhận diện cột điểm theo quy ước tên môn_lopN có trong dữ liệu."""
    return [
        column
        for column in data.columns
        if SCORE_COLUMN_PATTERN.fullmatch(str(column).strip()) is not None
    ]


def identify_yes_no_columns(data: pd.DataFrame) -> list[str]:
    """Nhận diện nhãn yes/no dựa trên toàn bộ giá trị không thiếu của cột."""
    result: list[str] = []
    for column in data.columns:
        values = data[column].dropna().astype(str).str.strip().str.casefold()
        unique_values = set(values.unique())
        if unique_values and unique_values.issubset({"yes", "no"}):
            result.append(column)
    return result


def load_metadata() -> dict[str, Any]:
    """Đọc metadata đã lưu cùng mô hình và kiểm tra sự tồn tại."""
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            "Chưa có metadata mô hình. Hãy chạy lần lượt bước 1, 2, 3 và 4 "
            "trước khi đánh giá hoặc dự đoán."
        )
    with METADATA_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def probability_frame(
    pipeline: Any, features: pd.DataFrame, label_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lấy xác suất lớp 1 và nhãn dự đoán cho mỗi mô hình nhị phân."""
    classifiers = pipeline.named_steps["classifier"].estimators_
    if len(classifiers) != len(label_columns):
        raise ValueError(
            "Số mô hình con không khớp số nhãn trong metadata. "
            "Hãy huấn luyện lại từ bước 3."
        )

    probabilities: dict[str, np.ndarray] = {}
    classifier_features = pipeline[:-1].transform(features)
    for label, classifier in zip(label_columns, classifiers):
        class_positions = np.flatnonzero(classifier.classes_ == 1)
        if len(class_positions) != 1:
            raise ValueError(
                f"Mô hình nhãn {label!r} không có đúng một lớp dương (1). "
                "Không thể lấy xác suất lựa chọn."
            )
        probabilities[label] = classifier.predict_proba(classifier_features)[
            :, class_positions[0]
        ]

    probability_data = pd.DataFrame(probabilities, index=features.index)
    predicted_data = pd.DataFrame(
        pipeline.predict(features),
        columns=label_columns,
        index=features.index,
    ).astype(int)
    return probability_data, predicted_data


def build_ranked_results(
    probabilities: pd.Series, predictions: pd.Series
) -> pd.DataFrame:
    """Tạo bảng xếp hạng xác suất giảm dần cho một học sinh."""
    result = pd.DataFrame(
        {
            "ten_mon": probabilities.index,
            "diem_xac_suat_uoc_luong": probabilities.values,
            "nhan_du_doan": predictions.reindex(probabilities.index).map(
                {1: "Có xu hướng chọn", 0: "Không dự đoán chọn"}
            ).values,
        }
    )
    result = result.sort_values(
        ["diem_xac_suat_uoc_luong", "ten_mon"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    result.insert(0, "thu_hang", np.arange(1, len(result) + 1))
    return result


def compare_official_combinations(top_subjects: list[str]) -> pd.DataFrame:
    """Đối chiếu Top 4 với các tổ hợp được cấu hình chính thức."""
    from subject_combinations import OFFICIAL_COMBINATIONS

    if not OFFICIAL_COMBINATIONS:
        return pd.DataFrame(
            columns=[
                "ten_to_hop",
                "so_mon_trung",
                "so_mon_to_hop",
                "khop_hoan_toan",
                "trang_thai",
            ]
        )

    top_set = {subject.strip().casefold() for subject in top_subjects}
    rows: list[dict[str, object]] = []
    for name, subjects in OFFICIAL_COMBINATIONS.items():
        official_set = {subject.strip().casefold() for subject in subjects}
        overlap = len(top_set & official_set)
        rows.append(
            {
                "ten_to_hop": name,
                "so_mon_trung": overlap,
                "so_mon_to_hop": len(official_set),
                "khop_hoan_toan": top_set == official_set,
                "trang_thai": "Gần nhất theo số môn trùng" if overlap else "Không trùng",
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["so_mon_trung", "ten_to_hop"],
        ascending=[False, True],
        kind="stable",
    )
    best = int(result["so_mon_trung"].max())
    result.loc[result["so_mon_trung"] == best, "trang_thai"] = (
        "Đồng hạng cao nhất theo số môn trùng"
    )
    result.loc[result["khop_hoan_toan"], "trang_thai"] = "Khớp hoàn toàn"
    return result.reset_index(drop=True)


def save_prediction_outputs(
    ranked: pd.DataFrame, output_prefix: str = "du_doan"
) -> tuple[Path, Path, Path]:
    """Lưu toàn bộ xác suất, Top 4 và kết quả đối chiếu tổ hợp."""
    STEP_06_DIR.mkdir(parents=True, exist_ok=True)
    probabilities_path = STEP_06_DIR / f"{output_prefix}_xac_suat.csv"
    top4_path = STEP_06_DIR / f"{output_prefix}_top4.csv"
    combinations_path = STEP_06_DIR / f"{output_prefix}_doi_chieu_to_hop.csv"
    ranked.to_csv(probabilities_path, index=False, encoding="utf-8-sig")
    top4 = ranked.head(4)
    top4.to_csv(top4_path, index=False, encoding="utf-8-sig")
    compare_official_combinations(top4["ten_mon"].tolist()).to_csv(
        combinations_path, index=False, encoding="utf-8-sig"
    )
    return probabilities_path, top4_path, combinations_path
