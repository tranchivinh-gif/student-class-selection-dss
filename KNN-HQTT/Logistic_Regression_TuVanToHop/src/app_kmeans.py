"""Giao diện thử phân cụm hồ sơ điểm bằng K-Means."""

from __future__ import annotations

import json
import re

import joblib
import pandas as pd
import streamlit as st

from common import (
    KMEANS_METADATA_PATH,
    KMEANS_MODEL_PATH,
    KMEANS_OUTPUT_DIR,
    STEP_03_DIR,
)


st.set_page_config(
    page_title="Thử nghiệm K-Means",
    page_icon="📊",
    layout="wide",
)
st.title("Thử nghiệm K-Means: phân nhóm hồ sơ điểm")
st.warning(
    "K-Means chỉ nhóm các hồ sơ có điểm tương tự. Nó không dự đoán môn lựa chọn, "
    "không đánh giá năng lực, mức độ phù hợp hoặc khả năng đỗ."
)

required_files = [
    KMEANS_MODEL_PATH,
    KMEANS_METADATA_PATH,
    STEP_03_DIR / "X_test.csv",
]
missing_files = [path.name for path in required_files if not path.is_file()]
if missing_files:
    st.error(
        "Chưa có mô hình hoặc tập test K-Means. Hãy chạy bước 8 và bước 9 trong "
        f"Terminal trước. Thiếu: {', '.join(missing_files)}"
    )
    st.stop()

with KMEANS_METADATA_PATH.open(encoding="utf-8") as file:
    metadata = json.load(file)
feature_columns = metadata["feature_columns"]
test_features = pd.read_csv(STEP_03_DIR / "X_test.csv", encoding="utf-8-sig")
if test_features.columns.tolist() != feature_columns or len(test_features) < 10:
    st.error(
        "Tập test không khớp mô hình hoặc có ít hơn 10 dòng. "
        "Hãy chạy lại bước 3 rồi huấn luyện lại K-Means."
    )
    st.stop()

pipeline = joblib.load(KMEANS_MODEL_PATH)
clusterer = pipeline.named_steps["kmeans"]
scaled_test = pipeline[:-1].transform(test_features)
test_cluster_ids = pipeline.predict(test_features)
cluster_sizes = pd.Series(test_cluster_ids).value_counts().to_dict()
cluster_centers_scaled = clusterer.cluster_centers_
cluster_centers_scores = pipeline.named_steps["scaler"].inverse_transform(
    cluster_centers_scaled
)

st.info(
    f"Mô hình đã chọn k={metadata['n_clusters']} cụm từ tập train. "
    f"Giao diện đang dùng {len(feature_columns)} cột điểm và {len(test_features)} "
    "học sinh ở tập test."
)
st.caption(
    "K-Means được fit trên 80% train. 20% test chỉ được gán vào tâm cụm đã học; "
    "không dùng test để chọn k hoặc huấn luyện."
)

sample_options = ["Tự nhập điểm"] + [
    f"Mẫu kiểm tra {index + 1:02d}" for index in range(10)
]
selected_sample = st.selectbox(
    "Chọn cách thử",
    sample_options,
    help="10 mẫu đầu lấy từ tập kiểm tra 20%, không dùng để huấn luyện.",
)
sample_index = (
    sample_options.index(selected_sample) - 1
    if selected_sample != "Tự nhập điểm"
    else None
)
if sample_index is not None:
    st.caption(
        f"Mẫu này thuộc cụm {int(test_cluster_ids[sample_index]) + 1}; "
        "các số cụm chỉ là nhãn định danh."
    )

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
        raise ValueError(f"Tên cột đặc trưng không hợp lệ: {column!r}")
    features_by_subject.setdefault(match.group(1), []).append(column)

with st.form("kmeans_student_scores"):
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
                entered[column] = float(
                    grid[index % len(grid)].number_input(
                        grade_label,
                        min_value=0.0,
                        max_value=10.0,
                        value=default_value,
                        step=0.5,
                        key=f"kmeans_score_{sample_index}_{column}",
                    )
                )
    submitted = st.form_submit_button("Phân nhóm hồ sơ điểm")

if submitted:
    missing = [column for column in feature_columns if column not in entered]
    if missing:
        st.error(f"Thiếu điểm ở các cột: {', '.join(missing)}")
    else:
        student = pd.DataFrame([entered], columns=feature_columns)
        transformed = pipeline[:-1].transform(student)
        cluster_id = int(pipeline.predict(student)[0])
        distances = pipeline.named_steps["kmeans"].transform(transformed)[0]
        distance = float(distances[cluster_id])
        cluster_size = int(cluster_sizes.get(cluster_id, 0))
        comparison = pd.DataFrame(
            {
                "cum": [f"Cụm {index + 1}" for index in range(metadata["n_clusters"])],
                "khoang_cach_den_tam": distances,
                "so_hoc_sinh_test_duoc_gan": [
                    int(cluster_sizes.get(index, 0))
                    for index in range(metadata["n_clusters"])
                ],
            }
        ).sort_values("khoang_cach_den_tam", ascending=True)

        st.subheader(f"Kết quả: hồ sơ được xếp vào Cụm {cluster_id + 1}")
        st.write(
            f"Khoảng cách đến tâm cụm đã gán: **{distance:.4f}** "
            "(trong không gian điểm đã chuẩn hóa)."
        )
        st.caption(
            f"Số học sinh test được gán vào cụm này: {cluster_size}. "
            "Đây không phải độ tin cậy hay xác suất thuộc cụm."
        )
        st.subheader("Khoảng cách đến các tâm cụm")
        st.dataframe(comparison, hide_index=True, width="stretch")

        centers = pd.DataFrame(
            cluster_centers_scores,
            columns=feature_columns,
            index=[f"Cụm {index + 1}" for index in range(metadata["n_clusters"])],
        )
        st.subheader("Điểm trung tâm đại diện của cụm được gán")
        st.dataframe(
            centers.iloc[[cluster_id]].T.rename(
                columns={f"Cụm {cluster_id + 1}": "Điểm trung tâm"}
            ),
            width="stretch",
        )
        st.caption(
            "Điểm trung tâm là trung bình của các hồ sơ train trong cụm sau khi "
            "đổi về thang điểm gốc; không phải ngưỡng hay điểm chuẩn."
        )

        KMEANS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output = pd.DataFrame(
            [
                {
                    "loai_ho_so": (
                        "mau_test"
                        if sample_index is not None
                        else "nhap_thu"
                    ),
                    "so_thu_tu_mau_test": sample_index,
                    "cum_kmeans": cluster_id + 1,
                    "khoang_cach_den_tam_chuan_hoa": distance,
                    "so_hoc_sinh_test_trong_cum": cluster_size,
                }
            ]
        )
        output_path = KMEANS_OUTPUT_DIR / "ket_qua_giao_dien.csv"
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        st.success(f"Đã lưu kết quả vào {output_path.relative_to(KMEANS_OUTPUT_DIR.parents[1])}")
