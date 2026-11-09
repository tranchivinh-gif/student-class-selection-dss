"""Clear, compact Streamlit UI for the real-data combination model."""

from __future__ import annotations

import sys
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_real.config import DEPLOYED_MODEL_PATHS, FEATURES, MODEL_PATH, SUBJECT_SLUGS  # noqa: E402
from src_real.model import RealCombinationModel  # noqa: E402

SUBJECT_NAMES = [
    "Ngữ văn", "Toán", "Ngoại ngữ 1", "GDCD", "Lịch sử và Địa lí",
    "Khoa học tự nhiên", "Công nghệ", "Tin học",
]
PERIODS = [("HKI", "hki"), ("HKII", "hkii"), ("Cả năm", "ca_nam")]
deployment_manifest = ROOT / "artifacts" / "src_real" / "deployment_manifest.json"
deployed_model_paths = {"champion": ROOT / MODEL_PATH}
champion_algorithm = "champion"
selected_model_name = "Model đang triển khai"
model_source = "Không có thông tin nguồn dữ liệu"
model_is_simulated = False
deployed_metrics: dict[str, float] = {}
if deployment_manifest.exists():
    try:
        deployed = json.loads(deployment_manifest.read_text(encoding="utf-8"))
        run_name = Path(deployed.get("source_run", "")).name
        source_data = str(deployed.get("source_data", ""))
        model_source = Path(source_data).name or model_source
        model_is_simulated = "mo_phong" in source_data.lower()
        champion_algorithm = deployed.get("selected_model", "champion")
        deployed_model_paths.update({
            name: ROOT / Path(path)
            for name, path in deployed.get("deployed_models", {}).items()
        })
        deployed_metrics = (
            deployed.get("selection_report", {}).get("independent_test_metrics", {})
        )
        if run_name:
            selected_model_name = run_name
    except (OSError, ValueError, TypeError):
        pass

st.set_page_config(page_title="Tư vấn tổ hợp môn", page_icon="🎓", layout="wide")
st.markdown("""
<style>
:root{--navy:#14324a;--blue:#176b87;--teal:#169c89;--gold:#e7a93b;--ink:#1d2939}
.stApp{background:linear-gradient(180deg,#edf5f7 0,#f8fafc 330px)}
.block-container{max-width:1120px;padding-top:1.4rem;padding-bottom:3rem}
.hero{background:linear-gradient(125deg,#14324a 0%,#176b87 58%,#169c89 100%);color:white;padding:1.65rem 2rem;border-radius:20px;box-shadow:0 16px 40px #14324a24}
.hero h1{font-size:2rem;margin:0 0 .35rem}.hero p{margin:0;opacity:.92;font-size:1.03rem}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin:1rem 0 1.3rem}
.step{background:#ffffffc9;border:1px solid #dbe7eb;border-radius:13px;padding:.78rem 1rem;color:#344054}.step b{color:var(--blue);margin-right:.35rem}
.section-note{background:#e9f5f4;border-left:4px solid var(--teal);padding:.8rem 1rem;border-radius:8px;color:#344054;margin:.35rem 0 1rem}
.winner{background:linear-gradient(135deg,#fff,#eefaf7);border:2px solid #77cabb;border-radius:18px;padding:1.25rem 1.35rem;box-shadow:0 10px 28px #169c891c;margin:.4rem 0 1rem}
.winner-label{font-size:.78rem;font-weight:800;letter-spacing:.08em;color:#087b6b;text-transform:uppercase}
.winner-title{font-size:1.3rem;font-weight:800;color:var(--ink);margin:.4rem 0 .55rem;line-height:1.45}
.winner-score{display:inline-block;background:#dff5ef;color:#087b6b;border-radius:999px;padding:.35rem .7rem;font-weight:800}
.option{background:white;border:1px solid #e2e8f0;border-left:5px solid var(--gold);border-radius:13px;padding:.9rem 1rem;margin:.6rem 0;color:var(--ink)}
.option-rank{font-size:.76rem;font-weight:800;color:#9a6511;text-transform:uppercase}.option-title{font-weight:700;margin:.2rem 0}.option-score{color:#667085;font-size:.92rem}
.explain{background:#fff8e8;border:1px solid #f3d99a;border-radius:12px;padding:.8rem 1rem;color:#694b13}
[data-testid="stMetric"]{background:white;border:1px solid #e2e8f0;padding:.75rem 1rem;border-radius:13px}
[data-testid="stDataFrame"]{border:1px solid #d9e3e8;border-radius:12px;overflow:hidden}
.stTabs [data-baseweb="tab-list"]{gap:.45rem}.stTabs [data-baseweb="tab"]{background:white;border-radius:10px;padding:.5rem 1rem}
.stTabs [aria-selected="true"]{background:#176b87!important;color:white!important}
@media(max-width:700px){.steps{grid-template-columns:1fr}.hero{padding:1.25rem}.hero h1{font-size:1.55rem}}
</style>
<div class="hero"><h1>🎓 Tư vấn tổ hợp môn</h1><p>Nhập điểm lớp 6–9 để xem tổ hợp phù hợp nhất và hai phương án có thể cân nhắc.</p></div>
<div class="steps"><div class="step"><b>1.</b> Chọn từng khối lớp</div><div class="step"><b>2.</b> Nhập điểm ba học kỳ</div><div class="step"><b>3.</b> Xem kết quả Top-3</div></div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## Cấu hình dự đoán")
available = [name for name in ("baseline", "xgboost", "catboost") if name in deployed_model_paths and deployed_model_paths[name].exists()]
display_names = {"baseline": "Baseline", "xgboost": "XGBoost", "catboost": "CatBoost"}
default_index = available.index(champion_algorithm) if champion_algorithm in available else 0
selected_algorithm = st.sidebar.selectbox(
    "Chọn thuật toán", available or ["champion"], index=default_index,
    format_func=lambda name: f"{display_names.get(name, name)}{' · champion' if name == champion_algorithm else ''}",
)
selected_model_path = deployed_model_paths.get(selected_algorithm, ROOT / MODEL_PATH)
st.sidebar.markdown("**Lần huấn luyện đang sử dụng**")
st.sidebar.code(selected_model_name, language=None)
st.sidebar.caption(f"Nguồn: {model_source}")
selected_metrics = deployed_metrics.get(selected_algorithm, deployed_metrics) if isinstance(deployed_metrics, dict) else {}
if selected_metrics.get("exact_match") is not None:
    st.sidebar.caption(
        f"Independent test — Top-1: {selected_metrics['exact_match']:.1%}"
    )
if model_is_simulated:
    st.sidebar.warning(
        "Model hiện tại học từ dữ liệu mô phỏng. Metric cao không đồng nghĩa với "
        "hiệu quả tương đương trên học sinh thật."
    )
else:
    st.sidebar.info("Model hiện tại học từ dữ liệu thật.")


@st.cache_resource
def load_model(model_path: str) -> RealCombinationModel:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Chưa có artifact: {path}")
    return RealCombinationModel.load(path)


def grade_table(periods: list[tuple[str, str]], default: float = 7.0) -> pd.DataFrame:
    return pd.DataFrame(default, index=SUBJECT_NAMES, columns=[name for name, _ in periods])


def tables_to_features(
    tables: dict[int, pd.DataFrame], periods: list[tuple[str, str]]
) -> pd.DataFrame:
    row: dict[str, float] = {}
    for grade, table in tables.items():
        for display, slug in zip(SUBJECT_NAMES, SUBJECT_SLUGS):
            for period_name, period_slug in periods:
                row[f"L{grade}_{slug}_{period_slug}"] = float(table.loc[display, period_name])
    return pd.DataFrame([row])


def confidence_message(top_probability: float, gap: float) -> str:
    if top_probability >= .70 and gap >= .25:
        return "Mô hình khá tự tin: phương án đầu nổi trội rõ so với phương án thứ hai."
    if top_probability >= .50 and gap >= .12:
        return "Mô hình nghiêng về phương án đầu, nhưng vẫn nên xem phương án thứ hai."
    return "Các phương án khá sát nhau. Nên kết hợp thêm sở thích và định hướng nghề nghiệp."


def show_result(data: pd.DataFrame) -> None:
    model = load_model(str(selected_model_path))
    recommendations = model.recommend(data, 3).reset_index(drop=True)
    subject_scores = model.subject_probabilities(data).iloc[0].sort_values(ascending=False)
    first, second = recommendations.iloc[0], recommendations.iloc[1]
    gap = float(first["probability"] - second["probability"])

    st.markdown("---")
    st.markdown("## Kết quả tư vấn")
    feature_mode = "chỉ điểm cả năm" if len(model._features()) == 32 else "điểm HKI, HKII và cả năm"
    st.caption(f"Mô hình đang dùng: {display_names.get(selected_algorithm, selected_algorithm)} · {selected_model_name} · {feature_mode}")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Độ phù hợp Top-1", f'{first["probability"]:.1%}')
    metric2.metric("Chênh lệch Top-1/Top-2", f"{gap:.1%}")
    metric3.metric("Số phương án", "3 tổ hợp")
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.markdown(
            f'<div class="winner"><div class="winner-label">Đề xuất phù hợp nhất</div><div class="winner-title">{first["combination"]}</div><span class="winner-score">Độ phù hợp {first["probability"]:.1%}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### Hai phương án có thể cân nhắc")
        for _, row in recommendations.iloc[1:].iterrows():
            st.markdown(
                f'<div class="option"><div class="option-rank">Phương án {int(row["rank"])}</div><div class="option-title">{row["combination"]}</div><div class="option-score">Độ phù hợp: {row["probability"]:.1%}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="explain">💡 {confidence_message(float(first["probability"]), gap)}</div>', unsafe_allow_html=True)
    with right:
        st.markdown("#### Mức phù hợp theo từng môn")
        st.caption("Được tổng hợp từ xác suất của bốn tổ hợp.")
        for subject, score in subject_scores.items():
            # Tree-model probabilities can exceed 1 by a few floating-point
            # units after aggregation (for example 1.000000002). Streamlit's
            # progress widget requires the closed interval [0, 1].
            safe_score = max(0.0, min(1.0, float(score)))
            name_col, score_col = st.columns([3.5, 1])
            name_col.write(subject)
            score_col.markdown(f"**{safe_score:.0%}**")
            st.progress(safe_score)
        st.markdown("#### Giải thích dự đoán")
        if model.algorithm == "baseline":
            st.info("Baseline không dùng điểm cá nhân; kết quả dựa trên tần suất tổ hợp trong dữ liệu huấn luyện nên không có SHAP.")
        else:
            with st.expander("Xem các điểm ảnh hưởng mạnh nhất (SHAP)"):
                try:
                    explanation = model.feature_contributions(data, top_n=8)
                    chart = explanation.set_index("feature")["contribution"].sort_values()
                    st.bar_chart(chart, horizontal=True)
                    st.caption("Giá trị dương làm tăng, giá trị âm làm giảm điểm của tổ hợp đứng đầu. SHAP không chứng minh quan hệ nhân quả.")
                except Exception as exc:
                    st.warning(f"Chưa thể tính SHAP cho dự đoán này: {exc}")
    st.download_button(
        "⬇️ Tải kết quả tư vấn", recommendations.to_csv(index=False).encode("utf-8-sig"),
        "goi_y_to_hop.csv", "text/csv", width="stretch",
    )


manual_tab, upload_tab = st.tabs(["✍️ Nhập điểm một học sinh", "📄 Dự đoán bằng CSV"])
with manual_tab:
    manual_model = load_model(str(selected_model_path))
    annual_only = all(column.endswith("_ca_nam") for column in manual_model._features())
    active_periods = [("Cả năm", "ca_nam")] if annual_only else PERIODS
    input_description = (
        "Model đang chọn chỉ sử dụng điểm cả năm; HKI và HKII không cần nhập."
        if annual_only else
        "Model đang chọn sử dụng HKI, HKII và Cả năm."
    )
    st.markdown(
        f'<div class="section-note"><b>Cách nhập:</b> Chọn lần lượt Lớp 6, 7, 8, 9. {input_description}</div>',
        unsafe_allow_html=True,
    )
    grade_tabs = st.tabs([f"Lớp {grade}" for grade in range(6, 10)])
    edited_tables: dict[int, pd.DataFrame] = {}
    for grade, tab in zip(range(6, 10), grade_tabs):
        with tab:
            st.caption(f"Điểm của 8 môn trong năm lớp {grade}")
            table = grade_table(active_periods)
            edited_tables[grade] = st.data_editor(
                table, key=f"grade_editor_{selected_algorithm}_{'annual' if annual_only else 'all'}_{grade}", width="stretch", height=325, num_rows="fixed",
                column_config={name: st.column_config.NumberColumn(name, min_value=0.0, max_value=10.0, step=.1, format="%.1f") for name, _ in active_periods},
            )
    if st.button("✨ Xem tổ hợp phù hợp", type="primary", width="stretch"):
        numeric_tables = {grade: table.apply(pd.to_numeric, errors="coerce") for grade, table in edited_tables.items()}
        invalid = any(table.isna().any().any() or ((table < 0) | (table > 10)).any().any() for table in numeric_tables.values())
        if invalid:
            st.error("Vui lòng nhập đủ điểm từ 0 đến 10 ở cả bốn khối lớp.")
        else:
            try:
                show_result(tables_to_features(numeric_tables, active_periods))
            except Exception as exc:
                st.error(f"Không thể dự đoán: {exc}")

with upload_tab:
    st.markdown('<div class="section-note"><b>Dự đoán hàng loạt:</b> CSV cần đủ 96 cột điểm. Cột mã học sinh và các cột nhãn là tùy chọn.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Chọn file CSV", type=["csv"])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded, encoding="utf-8-sig")
            batch_model = load_model(str(selected_model_path))
            required_features = batch_model._features()
            missing = [column for column in required_features if column not in frame.columns]
            if missing:
                raise ValueError(f"Thiếu {len(missing)} cột điểm: {missing[:8]}")
            numeric = frame[required_features].apply(pd.to_numeric, errors="raise")
            if numeric.isna().any().any() or ((numeric < 0) | (numeric > 10)).any().any():
                raise ValueError("Điểm phải đầy đủ và nằm trong khoảng 0–10.")
            result = batch_model.recommend(frame, 3)
            display = result[["ma_hoc_sinh", "rank", "combination", "probability"]].copy()
            display.columns = ["Mã học sinh", "Hạng", "Tổ hợp", "Độ phù hợp"]
            display["Độ phù hợp"] = display["Độ phù hợp"].map(lambda value: f"{value:.1%}")
            st.success(f"Đã dự đoán thành công {len(frame)} học sinh.")
            st.dataframe(display, hide_index=True, width="stretch", height=420)
            st.download_button("⬇️ Tải toàn bộ kết quả", result.to_csv(index=False).encode("utf-8-sig"), "du_doan_to_hop.csv", "text/csv", width="stretch")
        except Exception as exc:
            st.error(f"Không thể đọc hoặc dự đoán file: {exc}")

st.markdown("---")
st.caption("Lưu ý: kết quả dùng để hỗ trợ tham khảo. Mô hình hiện chỉ biết bốn tổ hợp trong dữ liệu huấn luyện và không thay thế tư vấn hướng nghiệp.")
