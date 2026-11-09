"""Shared schema and defaults for the complete pipeline."""

from pathlib import Path

ID_COLUMN = "ma_hoc_sinh"
RANDOM_SEED = 42
PRIMARY_K = 4
ADDITIONAL_K = 2

TARGETS = [
    "Vật lí", "Hóa học", "Sinh học", "Địa lí",
    "Giáo dục kinh tế và pháp luật (GDKT&PL)",
    "Tin học", "Công nghệ", "Âm nhạc", "Mĩ thuật",
]

INPUT_SUBJECTS = [
    ("Ngữ văn", "van"), ("Toán", "toan"),
    ("Ngoại ngữ 1", "ngoai_ngu_1"), ("GDCD", "gdcd"),
    ("Lịch sử và Địa lí", "lich_su_dia_li"),
    ("Khoa học tự nhiên", "khtn"),
    ("Công nghệ", "cong_nghe"), ("Tin học", "tin_hoc"),
]
PERIODS = [("HKI", "hki"), ("HKII", "hkii"), ("Cả năm", "ca_nam")]
FEATURES = [
    f"L{grade}_{subject}_{period}"
    for grade in range(6, 10)
    for _, subject in INPUT_SUBJECTS
    for _, period in PERIODS
]

DEFAULT_OUTPUT = Path("artifacts/src_v2")
