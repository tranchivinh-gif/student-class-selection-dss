"""Streamlit demo for the locked CatBoost subject-ranking model.

Run from the repository root:
    streamlit run src/recommendation/demo_app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.recommendation import SubjectRanker  # noqa: E402


MODEL_PATH = ROOT_DIR / "artifacts/subject_rankers/catboost_subject_ranker.joblib"
SAMPLE_PATH = ROOT_DIR / "data/data-splits/X_development.csv"

SUBJECT_INPUTS = [
    ("Ngữ văn", "van"),
    ("Toán", "toan"),
    ("Ngoại ngữ 1", "ngoai_ngu_1"),
    ("GDCD", "gdcd"),
    ("Lịch sử và Địa lí", "lich_su_dia_li"),
    ("Khoa học tự nhiên", "khtn"),
    ("Công nghệ", "cong_nghe"),
    ("Tin học", "tin_hoc"),
]

PERIODS = [
    ("HKI", "hki"),
    ("HKII", "hkii"),
    ("Cả năm", "ca_nam"),
]


st.set_page_config(
    page_title="Gợi ý môn học",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f6f8fc; }
      .block-container { max-width: 1240px; padding-top: 2rem; }
      .hero {
        padding: 1.7rem 2rem; border-radius: 22px;
        background: linear-gradient(120deg, #123c69 0%, #176b87 58%, #45a29e 100%);
        color: white; box-shadow: 0 14px 35px rgba(18,60,105,.18);
        margin-bottom: 1.2rem;
      }
      .hero h1 { margin: 0 0 .4rem; font-size: 2.15rem; }
      .hero p { margin: 0; opacity: .9; font-size: 1.02rem; }
      .section-title { color: #123c69; margin: 1.2rem 0 .7rem; }
      .rec-card {
        background: white; border: 1px solid #e5eaf2; border-radius: 16px;
        padding: .9rem 1rem; margin-bottom: .65rem;
        box-shadow: 0 5px 18px rgba(23,43,77,.06);
      }
      .rec-card.primary { border-left: 6px solid #176b87; }
      .rec-card.additional { border-left: 6px solid #e6a23c; }
      .rec-head { display:flex; justify-content:space-between; gap:1rem; align-items:center; }
      .rec-name { font-weight: 700; color: #172b4d; }
      .rank-badge {
        display:inline-block; min-width:2rem; text-align:center; margin-right:.55rem;
        background:#e8f3f6; color:#176b87; border-radius:999px; padding:.2rem .45rem;
      }
      .score { color:#176b87; font-weight:700; white-space:nowrap; }
      .bar { height:7px; background:#edf1f7; border-radius:999px; margin-top:.65rem; overflow:hidden; }
      .bar > div { height:100%; border-radius:999px; background:linear-gradient(90deg,#176b87,#45a29e); }
      .note {
        background:#fff8e8; border:1px solid #f0d998; border-radius:12px;
        padding:.8rem 1rem; color:#694f12;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_ranker() -> SubjectRanker:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình: {MODEL_PATH}")
    return SubjectRanker.load(MODEL_PATH)


@st.cache_data
def load_sample() -> pd.Series:
    return pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig").iloc[0]


def empty_grade_table(default: float = 7.0) -> pd.DataFrame:
    columns = [f"Lớp {grade} · {period_name}" for grade in range(6, 10) for period_name, _ in PERIODS]
    return pd.DataFrame(default, index=[name for name, _ in SUBJECT_INPUTS], columns=columns)


def sample_grade_table(sample: pd.Series) -> pd.DataFrame:
    table = empty_grade_table()
    for subject_name, subject_slug in SUBJECT_INPUTS:
        for grade in range(6, 10):
            for period_name, period_slug in PERIODS:
                table.loc[subject_name, f"Lớp {grade} · {period_name}"] = sample[
                    f"L{grade}_{subject_slug}_{period_slug}"
                ]
    return table


def table_to_features(table: pd.DataFrame) -> pd.DataFrame:
    values: dict[str, float] = {}
    for subject_name, subject_slug in SUBJECT_INPUTS:
        for grade in range(6, 10):
            for period_name, period_slug in PERIODS:
                values[f"L{grade}_{subject_slug}_{period_slug}"] = float(
                    table.loc[subject_name, f"Lớp {grade} · {period_name}"]
                )
    return pd.DataFrame([values])


def recommendation_card(row: pd.Series, primary: bool) -> str:
    score = max(0.0, min(1.0, float(row["recommendation_score"])))
    kind = "primary" if primary else "additional"
    return f"""
      <div class="rec-card {kind}">
        <div class="rec-head">
          <div class="rec-name"><span class="rank-badge">{int(row['rank'])}</span>{html.escape(str(row['subject']))}</div>
          <div class="score">{score * 100:.1f} điểm</div>
        </div>
        <div class="bar"><div style="width:{score * 100:.1f}%"></div></div>
      </div>
    """


st.markdown(
    """
    <div class="hero">
      <h1>Khám phá môn học phù hợp</h1>
      <p>Nhập kết quả học tập lớp 6–9 để nhận danh sách môn tự chọn được xếp hạng bởi mô hình CatBoost.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Thông tin mô hình")
    st.metric("NDCG@4 · Independent Test", "0,7205")
    st.metric("Recall@6 · Independent Test", "91,25%")
    st.caption("Mô hình cuối: CatBoost · 96 đặc trưng điểm · 9 môn tự chọn")
    st.divider()
    use_sample = st.toggle("Dùng một học sinh mẫu", value=False)
    st.caption("Bật tùy chọn này để xem nhanh cách giao diện hoạt động.")

st.subheader("1. Nhập bảng điểm", anchor=False)
st.caption("Có thể dán nhiều ô trực tiếp từ Excel. Tất cả điểm phải nằm trong khoảng 0–10.")

initial_table = sample_grade_table(load_sample()) if use_sample else empty_grade_table()
grade_table = st.data_editor(
    initial_table,
    use_container_width=True,
    height=345,
    num_rows="fixed",
    column_config={
        column: st.column_config.NumberColumn(column, min_value=0.0, max_value=10.0, step=0.1, format="%.1f")
        for column in initial_table.columns
    },
    key=f"grade_editor_{use_sample}",
)

predict = st.button("Xếp hạng môn học", type="primary", use_container_width=True)

if predict:
    numeric = grade_table.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        st.error("Bảng điểm còn ô trống hoặc chứa giá trị không hợp lệ.")
    elif ((numeric < 0) | (numeric > 10)).any().any():
        st.error("Điểm phải nằm trong khoảng từ 0 đến 10.")
    else:
        try:
            ranker = load_ranker()
            recommendations = ranker.recommend_grouped(
                table_to_features(numeric), primary_k=4, additional_k=2
            )
        except Exception as exc:
            st.error(f"Không thể tạo gợi ý: {exc}")
        else:
            st.subheader("2. Kết quả gợi ý", anchor=False)
            primary = recommendations[recommendations["rank"] <= 4]
            additional = recommendations[recommendations["rank"] > 4]
            left, right = st.columns([1.45, 1], gap="large")
            with left:
                st.markdown("### Đề xuất phù hợp nhất")
                for _, row in primary.iterrows():
                    st.markdown(recommendation_card(row, True), unsafe_allow_html=True)
            with right:
                st.markdown("### Có thể cân nhắc thêm")
                for _, row in additional.iterrows():
                    st.markdown(recommendation_card(row, False), unsafe_allow_html=True)
                st.markdown(
                    '<div class="note">Điểm gợi ý dùng để so sánh thứ hạng giữa các môn, không phải kết luận nghề nghiệp hay xác suất chắc chắn.</div>',
                    unsafe_allow_html=True,
                )

            export = recommendations[
                ["rank", "recommendation_group", "subject", "recommendation_score"]
            ].copy()
            st.download_button(
                "Tải kết quả CSV",
                data=export.to_csv(index=False).encode("utf-8-sig"),
                file_name="goi_y_mon_hoc.csv",
                mime="text/csv",
                use_container_width=True,
            )

st.divider()
st.caption(
    "Công cụ hỗ trợ tham khảo dựa trên điểm số và lựa chọn lịch sử. "
    "Học sinh nên kết hợp sở thích, định hướng nghề nghiệp và tư vấn từ gia đình, giáo viên."
)
