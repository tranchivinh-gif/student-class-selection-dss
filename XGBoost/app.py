"""Flask form for entering student grades and predicting elective subjects."""

from __future__ import annotations

import re
import sys
from typing import Any

import pandas as pd
from flask import Flask, render_template_string, request

from predict import load_model, predict_frame


web_app = Flask(__name__)
GRADE_COLUMN_PATTERN = re.compile(r"(?P<subject>.+)_lop(?P<grade>[6-9])$")
SUBJECT_NAMES = {
    "toan": "Toán",
    "ngu_van": "Ngữ văn",
    "ngoai_ngu": "Ngoại ngữ",
    "khoa_hoc_tu_nhien": "Khoa học tự nhiên",
    "lich_su_dia_ly": "Lịch sử và Địa lý",
    "giao_duc_cong_dan": "Giáo dục công dân",
    "tin_hoc": "Tin học",
    "cong_nghe": "Công nghệ",
    "giao_duc_the_chat": "Giáo dục thể chất",
    "am_nhac": "Âm nhạc",
    "mi_thuat": "Mỹ thuật",
    "noi_dung_giao_duc_dia_phuong": "Nội dung giáo dục địa phương",
    "hoat_dong_trai_nghiem_huong_nghiep": "Hoạt động trải nghiệm, hướng nghiệp",
}
ELECTIVE_NAMES = {
    "Vật lý": "Vật lý",
    "Hóa học": "Hóa học",
    "Sinh học": "Sinh học",
    "Công nghệ": "Công nghệ",
    "Tin học": "Tin học",
    "Địa lý": "Địa lý",
    "Giáo dục kinh tế và pháp luật": "Giáo dục kinh tế và pháp luật",
    "Âm nhạc": "Âm nhạc",
    "Mỹ thuật": "Mỹ thuật",
}

PAGE = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dự đoán môn tự chọn</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; color: #18212f;
            background: #f4f7fb; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 2rem 1rem 4rem; }
    main { max-width: 1050px; margin: 0 auto; }
    h1 { margin-bottom: .4rem; }
    .intro, .notice { color: #46556a; line-height: 1.6; }
    .notice { background: #eaf3fb; border-left: 4px solid #2680c2;
              border-radius: 6px; padding: .8rem 1rem; }
    .section { background: #fff; border: 1px solid #dce3ec; border-radius: 12px;
               padding: 1.25rem; margin: 1rem 0; }
    .subject-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(265px, 1fr));
                    gap: 1rem; }
    .subject { border: 1px solid #e1e7ef; border-radius: 10px; padding: 1rem; }
    .subject h3 { margin: 0 0 .8rem; font-size: 1rem; }
    .grade-row { display: grid; grid-template-columns: 1fr 110px; align-items: center;
                 gap: .75rem; margin: .55rem 0; }
    input[type=number] { width: 100%; padding: .55rem; border: 1px solid #bdc8d6;
                         border-radius: 6px; font: inherit; }
    input:focus { outline: 2px solid #73b5e7; border-color: #2680c2; }
    .actions { display: flex; flex-wrap: wrap; align-items: center; gap: .75rem; }
    button { border: 0; border-radius: 7px; padding: .75rem 1rem; font: inherit;
             font-weight: 650; cursor: pointer; }
    .primary { color: white; background: #1769aa; }
    .secondary { color: #26364a; background: #e8edf3; }
    .error { color: #9c2020; background: #fff0f0; border-radius: 7px; padding: .8rem; }
    table { border-collapse: collapse; width: 100%; }
    th, td { text-align: left; padding: .65rem; border-bottom: 1px solid #e3e9f0; }
    th { color: #46556a; font-size: .9rem; }
    .yes { color: #14733e; font-weight: 700; }
    .no { color: #586578; font-weight: 650; }
    .probability { color: #526176; }
    @media (max-width: 520px) {
      body { padding: 1rem .65rem 2rem; }
      .section { padding: .9rem; }
      .grade-row { grid-template-columns: 1fr 88px; }
      th, td { padding: .5rem .3rem; font-size: .88rem; }
    }
  </style>
</head>
<body>
<main>
  <h1>Đề xuất 4 môn học tự chọn</h1>
  <p class="intro">Nhập điểm các môn từ lớp 6 đến lớp 9. Điểm phải nằm trong
  khoảng 0–10.</p>
  <p class="notice"><strong>Lưu ý:</strong> điểm mẫu được điền sẵn chỉ để minh họa,
  không phải dữ liệu của một học sinh thật. Hãy nhập điểm thực tế cho tất cả
  các lớp, kể cả Ngữ văn lớp 8–9.
  Mô hình và kết quả dự đoán chỉ mang tính minh họa; mô hình không đánh giá
  năng lực hay môn học phù hợp.</p>

  {% if error %}<p class="error" role="alert">{{ error }}</p>{% endif %}

  <form action="/predict" method="post">
    <section class="section">
      <label for="student_name"><strong>Tên hoặc mã học sinh (không bắt buộc)</strong></label>
      <input id="student_name" name="student_name" type="text" maxlength="80"
             value="{{ student_name }}" placeholder="Ví dụ: Học sinh A"
             style="display:block;max-width:420px;width:100%;margin-top:.5rem;padding:.65rem;border:1px solid #bdc8d6;border-radius:6px;font:inherit">
    </section>

    <section class="section">
      <h2>Điểm các môn</h2>
      <div class="subject-grid">
      {% for subject in subjects %}
        <div class="subject">
          <h3>{{ subject.name }}</h3>
          {% for field in subject.fields %}
          <label class="grade-row" for="{{ field.column }}">
            <span>Lớp {{ field.grade }}</span>
            <input id="{{ field.column }}" name="{{ field.column }}" type="number"
                   min="0" max="10" step="0.1" required
                   value="{{ field.value }}" inputmode="decimal"
                   aria-label="Điểm {{ subject.name }} lớp {{ field.grade }}">
          </label>
          {% endfor %}
        </div>
      {% endfor %}
      </div>
    </section>

    <div class="actions">
      <button class="primary" type="submit">Đề xuất 4 môn tự chọn</button>
      <button class="secondary" type="reset">Khôi phục điểm mẫu</button>
    </div>
  </form>

  {% if results %}
  <section class="section" aria-live="polite">
    <h2>4 môn được đề xuất{% if student_name %}: {{ student_name }}{% endif %}</h2>
    <p class="intro">Danh sách gồm 4 môn có xác suất dự đoán lựa chọn cao nhất.
    Hệ thống luôn đưa ra 4 môn, kể cả khi xác suất thấp. Xác suất không phải
    mức độ phù hợp học tập.</p>
    <table>
      <thead><tr><th>Môn tự chọn</th><th>Xác suất dự đoán Có</th></tr></thead>
      <tbody>
      {% for result in results %}
        <tr>
          <td>{{ result.name }}</td>
          <td class="probability">{{ '%.1f'|format(result.probability * 100) }}%</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}
</main>
</body>
</html>"""


def _sample_score(index: int) -> float:
    """Return a deterministic illustrative score between 2.0 and 9.0."""
    return ((index * 17 + 23) % 71 + 20) / 10


def _display_subject_name(subject: str) -> str:
    if subject in SUBJECT_NAMES:
        return SUBJECT_NAMES[subject]
    return subject.replace("_", " ").capitalize()


def _build_subjects(
    feature_columns: list[str],
    values: dict[str, str],
) -> list[dict[str, Any]]:
    subjects: dict[str, dict[str, Any]] = {}
    for column in feature_columns:
        match = GRADE_COLUMN_PATTERN.fullmatch(column)
        if match is None:
            raise ValueError(f"Tên cột điểm không đúng định dạng: {column}")
        subject_key = match.group("subject")
        subject = subjects.setdefault(
            subject_key,
            {"name": _display_subject_name(subject_key), "fields": []},
        )
        subject["fields"].append(
            {
                "column": column,
                "grade": match.group("grade"),
                "value": values[column],
            }
        )
    return list(subjects.values())


def _sample_values(feature_columns: list[str]) -> dict[str, str]:
    return {
        column: f"{_sample_score(index):.1f}"
        for index, column in enumerate(feature_columns)
    }


def _render_form(
    feature_columns: list[str],
    values: dict[str, str],
    *,
    error: str | None = None,
    student_name: str = "",
    results: list[dict[str, Any]] | None = None,
    status: int = 200,
):
    return (
        render_template_string(
            PAGE,
            subjects=_build_subjects(feature_columns, values),
            error=error,
            student_name=student_name,
            results=results or [],
        ),
        status,
    )


@web_app.get("/")
def index():
    try:
        artifact = load_model()
    except (FileNotFoundError, ValueError) as error:
        return (
            f"Mô hình chưa sẵn sàng: {error}. Hãy chạy python .\\XGBoost\\train.py.",
            503,
        )
    features = artifact["feature_columns"]
    return _render_form(features, _sample_values(features))


@web_app.post("/predict")
def predict_grades():
    try:
        artifact = load_model()
    except (FileNotFoundError, ValueError) as error:
        return (
            f"Mô hình chưa sẵn sàng: {error}. Hãy chạy python .\\XGBoost\\train.py.",
            503,
        )

    feature_columns: list[str] = artifact["feature_columns"]
    values = {column: request.form.get(column, "").strip() for column in feature_columns}
    student_name = request.form.get("student_name", "").strip()
    if any(not value for value in values.values()):
        return _render_form(
            feature_columns,
            values,
            error="Vui lòng nhập đủ điểm cho tất cả các môn hiển thị.",
            student_name=student_name,
            status=400,
        )

    try:
        input_frame = pd.DataFrame(
            [{column: float(value) for column, value in values.items()}]
        )
        predictions = predict_frame(input_frame, artifact).iloc[0]
    except (TypeError, ValueError) as error:
        return _render_form(
            feature_columns,
            values,
            error=f"Điểm không hợp lệ. Hãy nhập số từ 0 đến 10. Chi tiết: {error}",
            student_name=student_name,
            status=400,
        )

    results = [
        {
            "name": ELECTIVE_NAMES.get(label, label),
            "probability": float(predictions[f"probability::{label}"]),
        }
        for label in artifact["target_columns"]
        if int(predictions[f"predicted::{label}"]) == 1
    ]
    results.sort(key=lambda item: item["probability"], reverse=True)
    return _render_form(
        feature_columns,
        values,
        student_name=student_name,
        results=results,
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    web_app.run(host="127.0.0.1", port=5000, debug=False)
