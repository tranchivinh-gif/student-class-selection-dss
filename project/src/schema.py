"""Quy ước schema CSV dùng chung cho dữ liệu học sinh."""

from __future__ import annotations


SUBJECT_COLUMN_NAMES = {
    "mathematics": "toan",
    "literature": "ngu_van",
    "foreign_language": "tieng_anh",
    "civics": "giao_duc_cong_dan",
    "history_geography": "lich_su_dia_li",
    "natural_sciences": "khoa_hoc_tu_nhien",
    "technology": "cong_nghe",
    "informatics": "tin_hoc",
}

STUDENT_ID_COLUMN = "ma_hoc_sinh"
CLASS_LABEL_COLUMN = "nhan_lop"


def score_column(subject: str, grade: int) -> str:
    """Trả về tên cột điểm cả năm của một môn và một lớp."""
    try:
        translated_subject = SUBJECT_COLUMN_NAMES[subject]
    except KeyError as error:
        raise ValueError(f"Môn học chưa có tên cột tiếng Việt: {subject}") from error
    return f"diem_{translated_subject}_ca_nam_lop_{grade}"
