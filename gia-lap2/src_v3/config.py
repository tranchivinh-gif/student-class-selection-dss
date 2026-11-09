"""Schema shared by the direct-combination pipeline."""

from pathlib import Path

ID_COLUMN = "ma_hoc_sinh"
RANDOM_SEED = 42
DEFAULT_OUTPUT = Path("artifacts/src_v3")

TARGETS = [
    "Vật lí", "Hóa học", "Sinh học", "Địa lí",
    "Giáo dục kinh tế và pháp luật (GDKT&PL)",
    "Tin học", "Công nghệ", "Âm nhạc", "Mĩ thuật",
]
SUBJECT_SLUGS = [
    "van", "toan", "ngoai_ngu_1", "gdcd",
    "lich_su_dia_li", "khtn", "cong_nghe", "tin_hoc",
]
FEATURES = [
    f"L{grade}_{subject}_{period}"
    for grade in range(6, 10)
    for subject in SUBJECT_SLUGS
    for period in ("hki", "hkii", "ca_nam")
]
