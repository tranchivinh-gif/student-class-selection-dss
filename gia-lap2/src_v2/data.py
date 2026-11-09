"""Load, validate and split a new labelled dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, ID_COLUMN, RANDOM_SEED, TARGETS


@dataclass
class DatasetSplit:
    development: pd.DataFrame
    holdout: pd.DataFrame
    stratified: bool


def load_and_validate(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig", dtype={ID_COLUMN: str})
    required = [ID_COLUMN, *FEATURES, *TARGETS]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Thiếu {len(missing)} cột bắt buộc: {missing[:8]}")
    data = data.loc[:, required].copy()
    data[ID_COLUMN] = data[ID_COLUMN].str.strip()
    if data[ID_COLUMN].isna().any() or data[ID_COLUMN].duplicated().any():
        raise ValueError("Mã học sinh phải đầy đủ và không trùng lặp.")
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="raise")
    if data[FEATURES].isna().any().any():
        raise ValueError("Dữ liệu điểm có giá trị thiếu.")
    invalid_score = ((data[FEATURES] < 0) | (data[FEATURES] > 10)).any().any()
    if invalid_score:
        raise ValueError("Điểm phải nằm trong khoảng 0–10.")
    data[TARGETS] = data[TARGETS].apply(pd.to_numeric, errors="raise").astype(int)
    if not data[TARGETS].isin([0, 1]).all().all():
        raise ValueError("Chín nhãn môn học chỉ được chứa 0 hoặc 1.")
    if not data[TARGETS].sum(axis=1).eq(4).all():
        raise ValueError("Mỗi học sinh phải có đúng bốn môn được chọn.")
    return data


def label_combination(data: pd.DataFrame) -> pd.Series:
    return data[TARGETS].astype(str).agg("".join, axis=1)


def split_dataset(
    data: pd.DataFrame,
    holdout_size: float = 0.1,
    random_state: int = RANDOM_SEED,
) -> DatasetSplit:
    combinations = label_combination(data)
    counts = combinations.value_counts()
    can_stratify = len(counts) > 1 and counts.min() >= 2
    development, holdout = train_test_split(
        data,
        test_size=holdout_size,
        random_state=random_state,
        shuffle=True,
        stratify=combinations if can_stratify else None,
    )
    return DatasetSplit(
        development=development.reset_index(drop=True),
        holdout=holdout.reset_index(drop=True),
        stratified=can_stratify,
    )
