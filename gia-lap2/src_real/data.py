from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import ALLOWED_COMBINATIONS, FEATURES, ID_COLUMN, TARGETS


def combination_key(labels: pd.DataFrame) -> pd.Series:
    return labels[TARGETS].astype(int).astype(str).agg("".join, axis=1)


def subjects_from_key(key: str) -> list[str]:
    return [name for name, selected in zip(TARGETS, key) if selected == "1"]


def combination_name(key: str) -> str:
    return " + ".join(subjects_from_key(key))


def load_training_data(path: str | Path, feature_columns: list[str] | None = None) -> pd.DataFrame:
    feature_columns = list(feature_columns or FEATURES)
    data = pd.read_csv(path, encoding="utf-8-sig", dtype={ID_COLUMN: str})
    required = [ID_COLUMN, *feature_columns, *TARGETS]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Thiếu {len(missing)} cột: {missing[:8]}")
    data = data[required].copy()
    if data[ID_COLUMN].isna().any() or data[ID_COLUMN].str.strip().eq("").any():
        raise ValueError("Mã học sinh không được để trống.")
    if data[ID_COLUMN].duplicated().any():
        raise ValueError("Mã học sinh bị trùng.")
    data[feature_columns] = data[feature_columns].apply(pd.to_numeric, errors="raise")
    values = data[feature_columns].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 10)).any():
        raise ValueError("Điểm phải đầy đủ và nằm trong khoảng 0–10.")
    data[TARGETS] = data[TARGETS].apply(pd.to_numeric, errors="raise").astype(int)
    if not data[TARGETS].isin([0, 1]).all().all():
        raise ValueError("Nhãn chỉ được là 0 hoặc 1.")
    data = data.copy()  # Defragment after assigning 96 numeric columns.
    data["combination_key"] = combination_key(data)
    unexpected = sorted(set(data["combination_key"]) - set(ALLOWED_COMBINATIONS))
    if unexpected:
        raise ValueError(f"Có tổ hợp ngoài bốn tổ hợp được hỗ trợ: {unexpected}")
    return data


def load_features(path: str | Path, feature_columns: list[str] | None = None) -> pd.DataFrame:
    feature_columns = list(feature_columns or FEATURES)
    data = pd.read_csv(path, encoding="utf-8-sig", dtype={ID_COLUMN: str})
    missing = [column for column in feature_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Thiếu {len(missing)} cột điểm: {missing[:8]}")
    values = data[feature_columns].apply(pd.to_numeric, errors="raise")
    array = values.to_numpy(float)
    if not np.isfinite(array).all() or ((array < 0) | (array > 10)).any():
        raise ValueError("Điểm phải đầy đủ và nằm trong khoảng 0–10.")
    return data

