"""Giao diện Streamlit để thử dự đoán với mô hình đã lưu."""

from __future__ import annotations

import re

import joblib
import pandas as pd
import streamlit as st

from common import (
    MODEL_PATH,
    STEP_03_DIR,
    build_ranked_results,
    compare_official_combinations,
    load_metadata,
    probability_frame,
    save_prediction_outputs,
)


st.set_page_config(page_title="Tư vấn tổ hợp môn", page_icon="📘", layout="wide")
st.title("Hệ thống tham khảo xu hướng lựa chọn môn")
st.caption(
    "Mô hình Logistic Regression học từ nhãn yes/no trong dữ liệu. "
    "Kết quả không phải đánh giá năng lực, mức độ phù hợp hay xác suất đỗ."
)

if not MODEL_PATH.is_file():
    st.error(
        "Chưa có mô hình. Hãy chạy các bước 1–4 trong Terminal trước khi mở giao diện."
    )
    st.stop()

metadata = load_metadata()
feature_columns = metadata["feature_columns"]
label_columns = metadata["label_columns"]
pipeline = joblib.load(MODEL_PATH)
test_features_path = STEP_03_DIR / "X_test.csv"
if not test_features_path.is_file():
    st.error("Không tìm thấy dữ liệu test. Hãy chạy lại bước 3.")
    st.stop()
test_features = pd.read_csv(test_features_path, encoding="utf-8-sig")
if test_features.columns.tolist() != feature_columns or len(test_features) < 10:
    st.error(
        "Dữ liệu test không khớp mô hình hoặc có ít hơn 10 dòng. "
        "Hãy chạy lại bước 3 và bước 4."
    )
    st.stop()
subject_display_names = {
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
features_by_subject: dict[str, list[str]] = {}
for column in feature_columns:
    match = re.fullmatch(r"(.+)_lop([6-9])", column, re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Tên cột điểm {column!r} không còn theo định dạng đã nhận diện."
        )
    features_by_subject.setdefault(match.group(1), []).append(column)

st.info(
    f"Mô hình có {len(feature_columns)} đặc trưng điểm và dự đoán "
    f"{len(label_columns)} nhãn lựa chọn. Hãy nhập đủ điểm trong khoảng 0–10."
)

sample_options = ["Tự nhập điểm"] + [
    f"Mẫu kiểm tra {index + 1:02d}" for index in range(10)
]
selected_sample = st.selectbox(
    "Chọn cách thử",
    sample_options,
    help=(
        "10 mẫu lấy từ tập kiểm tra 20%, không dùng để huấn luyện. "
        "Chọn một mẫu để tự điền điểm, hoặc chọn Tự nhập điểm."
    ),
)
sample_index = (
    sample_options.index(selected_sample) - 1 if selected_sample != "Tự nhập điểm" else None
)
if sample_index is not None:
    st.caption(
        "Điểm được điền từ một học sinh trong tập kiểm tra để thử giao diện; "
        "đây không phải hồ sơ mới."
    )

with st.form("student_scores"):
    entered: dict[str, float] = {}
    for subject, columns in features_by_subject.items():
        with st.expander(
            subject_display_names.get(subject, subject.replace("_", " ")),
            expanded=False,
        ):
            grid = st.columns(min(4, len(columns)))
            for index, column in enumerate(columns):
                grade = re.search(r"_lop([6-9])$", column, re.IGNORECASE)
                grade_label = f"Lớp {grade.group(1)}" if grade else column
                default_value = (
                    float(test_features.iloc[sample_index][column])
                    if sample_index is not None
                    else 0.0
                )
                value = grid[index % len(grid)].number_input(
                    grade_label,
                    min_value=0.0,
                    max_value=10.0,
                    value=default_value,
                    step=0.5,
                    key=f"score_{sample_index}_{column}",
                )
                entered[column] = float(value)

    submitted = st.form_submit_button("Dự đoán xu hướng lựa chọn")

if submitted:
    missing = [column for column in feature_columns if column not in entered]
    if missing:
        st.error(
            f"Còn thiếu hoặc sai định dạng {len(missing)} điểm. "
            f"Các cột cần kiểm tra: {', '.join(missing)}"
        )
    else:
        student = pd.DataFrame([entered], columns=feature_columns)
        probabilities, predictions = probability_frame(
            pipeline, student, label_columns
        )
        ranked = build_ranked_results(probabilities.iloc[0], predictions.iloc[0])
        all_path, top4_path, combination_path = save_prediction_outputs(
            ranked, output_prefix="giao_dien"
        )

        st.subheader("Bảng xếp hạng các môn")
        display = ranked.copy()
        display["diem_xac_suat_uoc_luong"] = display[
            "diem_xac_suat_uoc_luong"
        ].map(lambda value: f"{value:.2%}")
        st.dataframe(display, hide_index=True, width="stretch")

        st.subheader("Top 4 theo điểm xác suất mô hình ước lượng")
        st.dataframe(display.head(4), hide_index=True, width="stretch")

        combinations = compare_official_combinations(
            ranked.head(4)["ten_mon"].tolist()
        )
        st.subheader("Đối chiếu tổ hợp môn chính thức")
        if combinations.empty:
            st.warning(
                "Chưa cấu hình 5 tổ hợp chính thức. Không tự tạo tổ hợp hoặc "
                "đề xuất nguyện vọng. Hãy cung cấp danh sách để bổ sung "
                "src/subject_combinations.py."
            )
        else:
            st.dataframe(combinations, hide_index=True, width="stretch")
            if not combinations["khop_hoan_toan"].any():
                st.warning(
                    "Top 4 không khớp hoàn toàn tổ hợp chính thức nào. "
                    "Tổ hợp đồng hạng đầu chỉ gần nhất theo số môn trùng, "
                    "không phải kết luận phù hợp nhất."
                )

        st.caption(
            "Điểm xác suất này chưa được hiệu chỉnh; không diễn giải thành xác suất "
            "lựa chọn thực tế chính xác, điểm số, xác suất đỗ hoặc mức độ phù hợp."
        )
        st.success(
            "Đã lưu CSV trong outputs/step_06/: bảng xác suất, Top 4 và đối chiếu tổ hợp."
        )
        st.download_button(
            "Tải bảng xác suất CSV",
            data=all_path.read_bytes(),
            file_name=all_path.name,
            mime="text/csv",
        )
        st.download_button(
            "Tải bảng Top 4 CSV",
            data=top4_path.read_bytes(),
            file_name=top4_path.name,
            mime="text/csv",
        )
