"""Web interface for the artifact produced by ``src_v2.train``."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_v2.config import DEFAULT_OUTPUT, INPUT_SUBJECTS, PERIODS  # noqa: E402
from src_v2.model import SubjectRankingModel  # noqa: E402

MODEL_PATH = DEFAULT_OUTPUT / "champion.joblib"

st.set_page_config(page_title="Gợi ý môn học", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.stApp{background:#f5f7fb}.block-container{max-width:1200px;padding-top:2rem}
.hero{background:linear-gradient(120deg,#123c69,#176b87,#45a29e);color:white;padding:1.7rem 2rem;border-radius:22px;margin-bottom:1rem}
.hero h1{margin:0}.card{background:white;padding:1rem;border-radius:15px;border:1px solid #e3e8f0;margin:.55rem 0;box-shadow:0 5px 18px #17304a10}
.primary{border-left:6px solid #176b87}.extra{border-left:6px solid #e6a23c}.score{float:right;color:#176b87;font-weight:700}
</style>
<div class="hero"><h1>Gợi ý môn học phù hợp</h1><p>Nhập điểm lớp 6–9 để nhận Top-4 đề xuất chính và 2 môn cân nhắc thêm.</p></div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model() -> SubjectRankingModel:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Chưa có {MODEL_PATH}. Hãy chạy python -m src_v2.train --data <file.csv> trước."
        )
    return SubjectRankingModel.load(MODEL_PATH)


def make_table() -> pd.DataFrame:
    columns = [f"Lớp {g} · {name}" for g in range(6, 10) for name, _ in PERIODS]
    return pd.DataFrame(7.0, index=[name for name, _ in INPUT_SUBJECTS], columns=columns)


def to_features(table: pd.DataFrame) -> pd.DataFrame:
    row = {}
    for display, slug in INPUT_SUBJECTS:
        for grade in range(6, 10):
            for period_name, period_slug in PERIODS:
                row[f"L{grade}_{slug}_{period_slug}"] = float(
                    table.loc[display, f"Lớp {grade} · {period_name}"]
                )
    return pd.DataFrame([row])


st.subheader("Bảng điểm học sinh")
table = make_table()
edited = st.data_editor(
    table, width="stretch", height=345, num_rows="fixed",
    column_config={c: st.column_config.NumberColumn(c, min_value=0.0, max_value=10.0, step=.1, format="%.1f") for c in table.columns},
)

if st.button("Xếp hạng môn học", type="primary", width="stretch"):
    numeric = edited.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or ((numeric < 0) | (numeric > 10)).any().any():
        st.error("Tất cả điểm phải hợp lệ và nằm trong khoảng 0–10.")
    else:
        try:
            results = load_model().recommend(to_features(numeric))
        except Exception as exc:
            st.error(str(exc))
        else:
            left, right = st.columns([1.45, 1], gap="large")
            with left:
                st.markdown("### Đề xuất phù hợp nhất")
                for _, row in results[results["rank"] <= 4].iterrows():
                    st.markdown(f'<div class="card primary"><b>{int(row["rank"])}. {row["subject"]}</b><span class="score">{row["recommendation_score"]*100:.1f} điểm</span></div>', unsafe_allow_html=True)
            with right:
                st.markdown("### Có thể cân nhắc thêm")
                for _, row in results[results["rank"] > 4].iterrows():
                    st.markdown(f'<div class="card extra"><b>{int(row["rank"])}. {row["subject"]}</b><span class="score">{row["recommendation_score"]*100:.1f} điểm</span></div>', unsafe_allow_html=True)
            st.download_button(
                "Tải kết quả CSV", results.to_csv(index=False).encode("utf-8-sig"),
                "goi_y_mon_hoc.csv", "text/csv", width="stretch",
            )

st.caption("Điểm gợi ý dùng để xếp hạng, không thay thế sở thích và tư vấn định hướng nghề nghiệp.")
