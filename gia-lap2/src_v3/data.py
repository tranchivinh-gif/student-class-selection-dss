"""Validation and target construction for combination classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import FEATURES, ID_COLUMN, TARGETS


def combination_key(labels: pd.DataFrame) -> pd.Series:
    """Encode four selected subjects as a stable nine-bit string."""
    return labels[TARGETS].astype(int).astype(str).agg("".join, axis=1)


def combination_name(key: str) -> str:
    return " + ".join(subject for subject, selected in zip(TARGETS, key) if selected == "1")


def key_to_subjects(key: str) -> list[str]:
    if len(key) != len(TARGETS) or set(key) - {"0", "1"} or key.count("1") != 4:
        raise ValueError(f"Tổ hợp không hợp lệ: {key}")
    return [subject for subject, selected in zip(TARGETS, key) if selected == "1"]


def load_and_validate(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig", dtype={ID_COLUMN: str})
    required = [ID_COLUMN, *FEATURES, *TARGETS]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Thiếu {len(missing)} cột bắt buộc: {missing[:8]}")
    data = data[required].copy()
    data[ID_COLUMN] = data[ID_COLUMN].str.strip()
    if data[ID_COLUMN].isna().any() or data[ID_COLUMN].duplicated().any():
        raise ValueError("Mã học sinh phải đầy đủ và duy nhất.")
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="raise")
    if data[FEATURES].isna().any().any():
        raise ValueError("Dữ liệu điểm có giá trị thiếu.")
    if ((data[FEATURES] < 0) | (data[FEATURES] > 10)).any().any():
        raise ValueError("Điểm phải nằm trong khoảng 0–10.")
    data[TARGETS] = data[TARGETS].apply(pd.to_numeric, errors="raise").astype(int)
    if not data[TARGETS].isin([0, 1]).all().all():
        raise ValueError("Nhãn môn học chỉ được chứa 0 hoặc 1.")
    if not data[TARGETS].sum(axis=1).eq(4).all():
        raise ValueError("Mỗi học sinh phải chọn đúng bốn môn.")
    data = data.copy()  # Defragment after bulk numeric assignments.
    data["combination_key"] = combination_key(data)
    data["combination_name"] = data["combination_key"].map(combination_name)
    return data
