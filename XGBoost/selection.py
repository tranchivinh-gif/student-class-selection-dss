"""Turn per-subject probabilities into exactly four subject choices."""

from __future__ import annotations

import numpy as np


SUBJECT_COUNT = 4


def select_top_subjects(probabilities: np.ndarray, count: int = SUBJECT_COUNT) -> np.ndarray:
    """Return binary choices for the highest-probability subjects in each row."""
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 2 or scores.shape[1] < count or count < 1:
        raise ValueError(f"Cần ít nhất {count} cột xác suất môn học.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Xác suất phải là số hữu hạn trong khoảng [0, 1].")
    selected = np.zeros(scores.shape, dtype=int)
    top_indices = np.argsort(-scores, axis=1, kind="stable")[:, :count]
    selected[np.arange(scores.shape[0])[:, None], top_indices] = 1
    return selected
