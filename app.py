"""Local FastAPI web interface for the simulated CatBoost exam-score models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import pandas as pd
import joblib
from catboost import CatBoostClassifier, CatBoostRegressor
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
METADATA_PATH = MODEL_DIR / "metadata.json"
SVR_METADATA_PATH = MODEL_DIR / "svr_metadata.json"
ELECTIVE_METADATA_PATH = MODEL_DIR / "elective_metadata.json"
if not METADATA_PATH.exists():
    raise RuntimeError("Chưa có mô hình. Hãy chạy: python train_models.py")
if not SVR_METADATA_PATH.exists():
    raise RuntimeError("Chưa có mô hình SVR. Hãy chạy: python train_svr_models.py")

METADATA = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
SVR_METADATA = json.loads(SVR_METADATA_PATH.read_text(encoding="utf-8"))
ELECTIVE_METADATA = json.loads(ELECTIVE_METADATA_PATH.read_text(encoding="utf-8")) if ELECTIVE_METADATA_PATH.exists() else None
MODELS: dict[str, CatBoostRegressor] = {}
SVR_MODELS = {}
for subject_key, subject in METADATA["subjects"].items():
    model_path = MODEL_DIR / subject["model_file"]
    if not model_path.exists():
        raise RuntimeError(f"Thiếu tệp mô hình: {model_path}")
    model = CatBoostRegressor()
    model.load_model(str(model_path), format="cbm")
    MODELS[subject_key] = model
    svr_subject = SVR_METADATA["subjects"][subject_key]
    svr_model_path = MODEL_DIR / svr_subject["model_file"]
    if not svr_model_path.exists():
        raise RuntimeError(f"Thiếu tệp mô hình: {svr_model_path}")
    SVR_MODELS[subject_key] = joblib.load(svr_model_path)

ELECTIVE_MODELS: dict[str, dict[str, object]] = {"catboost": {}, "svc": {}}
if ELECTIVE_METADATA:
    for algorithm, details in ELECTIVE_METADATA["algorithms"].items():
        for index, (target, info) in enumerate(details["targets"].items(), start=1):
            model_path = MODEL_DIR / info["artifact"]
            if algorithm == "catboost":
                model = CatBoostClassifier()
                model.load_model(str(model_path), format="cbm")
            else:
                model = joblib.load(model_path)
            ELECTIVE_MODELS[algorithm][target] = model

app = FastAPI(title="Dự báo điểm thi mô phỏng", version="1.0.0")


class PredictionRequest(BaseModel):
    subject: Literal["toan", "ngu_van", "ngoai_ngu"]
    algorithm: Literal["catboost", "svr"]
    features: dict[str, float]


class ElectivePredictionRequest(BaseModel):
    algorithm: Literal["catboost", "svc"]
    features: dict[str, float]


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE


@app.get("/api/metadata")
def get_metadata() -> dict:
    return {
        "subjects": {
            key: {
                "label": value["label"],
                "mock_score": value["mock_score"],
            }
            for key, value in METADATA["subjects"].items()
        },
        "algorithms": {
            "catboost": {
                "label": "CatBoost",
                "train_rows": METADATA["split_counts"]["train"],
                "subjects": {
                    key: {
                        "test_metrics": value["test_metrics"],
                        "baseline_test_metrics": value["baseline_test_metrics"],
                    }
                    for key, value in METADATA["subjects"].items()
                },
            },
            "svr": {
                "label": "SVR (RBF)",
                "train_rows": SVR_METADATA["train_rows"],
                "limitations": SVR_METADATA["limitations"],
                "subjects": {
                    key: {
                        "test_metrics": value["test_metrics"],
                        "baseline_test_metrics": value["baseline_test_metrics"],
                    }
                    for key, value in SVR_METADATA["subjects"].items()
                },
            },
        },
        "features_common": METADATA["features_common"],
        "feature_groups": METADATA["feature_groups"],
        "defaults": METADATA["defaults"],
        "demo_profiles": [
            {"id": item["id"], "label": item["label"]}
            for item in METADATA["demo_profiles"]
        ],
        "limitations": METADATA["limitations"],
    }


@app.get("/api/sample/{sample_id}")
def get_sample(sample_id: int) -> dict:
    sample = next((item for item in METADATA["demo_profiles"] if item["id"] == sample_id), None)
    if sample is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy hồ sơ mẫu.")
    return sample


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    subject = METADATA["subjects"][request.subject]
    expected_features = subject["features"]
    missing = [name for name in expected_features if name not in request.features]
    extra = [name for name in request.features if name not in expected_features]
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": extra},
        )

    score_features: dict[str, float] = {}
    for name in expected_features:
        value = request.features[name]
        if not math.isfinite(value):
            raise HTTPException(status_code=422, detail=f"Giá trị {name} phải là số hữu hạn.")
        if name == "gio_tu_hoc_tuan":
            valid = 0 <= value <= 30
        elif name == "ti_le_chuyen_can":
            valid = 0 <= value <= 1
        else:
            valid = 0 <= value <= 10
        if not valid:
            raise HTTPException(status_code=422, detail=f"Giá trị {name} nằm ngoài khoảng cho phép.")
        score_features[name] = value

    ordered = pd.DataFrame([[score_features[name] for name in expected_features]], columns=expected_features)
    selected_model = MODELS[request.subject] if request.algorithm == "catboost" else SVR_MODELS[request.subject]
    prediction = float(selected_model.predict(ordered)[0])
    return {
        "subject": request.subject,
        "subject_label": subject["label"],
        "algorithm": request.algorithm,
        "prediction": round(prediction, 2),
        "scale": "điểm mô phỏng (0–10)",
        "note": METADATA["limitations"],
    }


@app.get("/api/electives/metadata")
def get_elective_metadata() -> dict:
    if ELECTIVE_METADATA is None:
        raise HTTPException(status_code=503, detail="Chua huan luyen mo hinh mon tu chon.")
    return ELECTIVE_METADATA


@app.get("/api/electives/sample/{sample_id}")
def get_elective_sample(sample_id: int) -> dict:
    if ELECTIVE_METADATA is None:
        raise HTTPException(status_code=503, detail="Chua huan luyen mo hinh mon tu chon.")
    sample = next((item for item in ELECTIVE_METADATA["demo_profiles"] if item["id"] == sample_id), None)
    if sample is None:
        raise HTTPException(status_code=404, detail="Khong tim thay ho so mau.")
    return sample


@app.post("/api/electives/predict")
def predict_electives(request: ElectivePredictionRequest) -> dict:
    if ELECTIVE_METADATA is None:
        raise HTTPException(status_code=503, detail="Chua huan luyen mo hinh mon tu chon.")
    expected = ELECTIVE_METADATA["features"]
    missing = [name for name in expected if name not in request.features]
    extra = [name for name in request.features if name not in expected]
    if missing or extra:
        raise HTTPException(status_code=422, detail={"missing_features": missing, "unexpected_features": extra})
    values = []
    for name in expected:
        value = request.features[name]
        if not math.isfinite(value) or not 0 <= value <= 10:
            raise HTTPException(status_code=422, detail=f"Gia tri {name} phai nam trong khoang 0-10.")
        values.append(value)
    ordered = pd.DataFrame([values], columns=expected)
    probabilities = []
    for target in ELECTIVE_METADATA["algorithms"][request.algorithm]["targets"]:
        model = ELECTIVE_MODELS[request.algorithm][target]
        classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
        positive_index = int(list(classes).index(1))
        probability = float(model.predict_proba(ordered)[0, positive_index])
        probabilities.append({"key": target, "label": ELECTIVE_METADATA["algorithms"][request.algorithm]["targets"][target]["label"], "probability": probability})
    ranked = sorted(probabilities, key=lambda item: item["probability"], reverse=True)
    return {"algorithm": request.algorithm, "top_four": ranked[:4],
            "note": ELECTIVE_METADATA["label_semantics"]}


@app.get("/electives", response_class=HTMLResponse)
def elective_home() -> str:
    return ELECTIVE_PAGE


PAGE = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dự báo điểm thi mô phỏng</title>
  <style>
    :root { color-scheme: light; --ink:#162c28; --muted:#61736f; --line:#dce6e2; --green:#176b53; --mint:#e9f4ef; --paper:#f5f8f6; }
    * { box-sizing:border-box }
    body { margin:0; background:var(--paper); color:var(--ink); font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif }
    header { background:#123c32; color:white; padding:30px max(22px,calc((100% - 1100px)/2)) }
    header h1 { margin:0 0 5px; font-size:clamp(24px,4vw,34px) }
    header p { margin:0; color:#d3e7df }
    main { max-width:1100px; padding:24px 18px 60px; margin:auto }
    .notice { background:#fff5d9; border:1px solid #eadcae; border-radius:10px; padding:13px 16px; margin:0 0 20px }
    .layout { display:grid; grid-template-columns:minmax(0,1fr) 310px; gap:20px; align-items:start }
    .card { background:white; border:1px solid var(--line); border-radius:14px; padding:20px; box-shadow:0 4px 18px #15382b0a }
    h2 { margin:0 0 16px; font-size:20px }
    .toolbar { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; align-items:end; margin-bottom:16px }
    label { display:block; font-weight:600; font-size:14px; margin-bottom:5px }
    select,input { font:inherit; width:100%; border:1px solid #c7d5d0; border-radius:8px; padding:9px 10px; color:var(--ink); background:white }
    input:focus,select:focus { outline:2px solid #78b69d; border-color:var(--green) }
    .btn { border:0; border-radius:8px; padding:10px 15px; background:var(--green); color:white; font:600 15px/1.2 system-ui,sans-serif; cursor:pointer }
    .btn.secondary { background:#e9f2ee; color:var(--green); border:1px solid #c9ded4 }
    .btn:disabled { opacity:.6; cursor:wait }
    .hint { color:var(--muted); font-size:13px; margin:5px 0 16px }
    details { border-top:1px solid var(--line); padding:12px 0 }
    summary { cursor:pointer; font-weight:700; color:#285448 }
    .grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px 12px; padding-top:12px }
    .grid label { font-weight:500; font-size:13px; color:#405751 }
    .grid input { padding:7px 8px; font-size:14px }
    .grid.single { grid-template-columns:repeat(3,minmax(0,1fr)) }
    .actions { display:flex; gap:10px; align-items:center; margin-top:18px; flex-wrap:wrap }
    #status { color:#9a3d25; font-size:14px }
    .result { display:none; background:var(--mint); border:1px solid #c8e0d4; border-radius:12px; padding:18px; margin-top:18px }
    .result.show { display:block }
    .score { font-size:36px; font-weight:750; color:#135d47; line-height:1.2; margin:5px 0 }
    .result p { margin:7px 0 }
    .side p { color:var(--muted); margin:0 0 12px; font-size:14px }
    .metric { border-top:1px solid var(--line); padding:10px 0; font-size:14px }
    .metric strong { display:block; color:var(--green) }
    .small { color:var(--muted); font-size:12px }
    @media(max-width:820px) { .layout { grid-template-columns:1fr } .grid { grid-template-columns:repeat(2,minmax(0,1fr)) } }
    @media(max-width:480px) { .grid,.grid.single { grid-template-columns:1fr 1fr } .toolbar { grid-template-columns:1fr } }
  </style>
</head>
<body>
  <header>
    <h1>Thử nghiệm dự báo điểm thi</h1>
    <p>CatBoost hoặc SVR · chọn môn · nhập hồ sơ để xem dự đoán</p>
    <p><a href="/electives" style="color:#d3e7df">Mở trang gợi ý 4 môn tự chọn</a></p>
  </header>
  <main>
    <div class="notice"><strong>Đây là mô hình thử nghiệm.</strong> Dữ liệu và điểm thi dùng để huấn luyện đều là mô phỏng; kết quả không phải tư vấn tuyển sinh hay dự đoán điểm thi thật.</div>
    <div class="layout">
      <section class="card">
        <h2>Nhập thông tin học tập</h2>
        <div class="toolbar">
          <div><label for="subject">Môn cần dự đoán</label><select id="subject"></select></div>
          <div><label for="algorithm">Thuật toán</label><select id="algorithm"></select></div>
          <div><label for="sample">Nạp hồ sơ mẫu</label><select id="sample"><option value="">Tự nhập</option></select></div>
        </div>
        <p class="hint">Các ô được điền sẵn bằng trung vị của tập train để tiện thử. Hãy sửa thành thông tin cần dự đoán. Tỷ lệ chuyên cần nhập theo phần trăm.</p>
        <div id="fields"></div>
        <div class="actions">
          <button class="btn" id="predict">Dự đoán</button>
          <button class="btn secondary" id="reset">Khôi phục giá trị mặc định</button>
          <span id="status"></span>
        </div>
        <div class="result" id="result">
          <div>Dự đoán điểm <strong id="result-subject"></strong></div>
          <div class="score" id="result-score"></div>
          <p id="sample-comparison"></p>
          <p class="small">Đây là đầu ra mô hình trên dữ liệu mô phỏng, không phải điểm thi thật.</p>
        </div>
      </section>
      <aside class="card side">
        <h2>Kết quả kiểm thử</h2>
        <p>MAE càng thấp càng tốt. So sánh trên tập test mô phỏng gồm 10.000 hồ sơ.</p>
        <div id="metrics"></div>
        <p class="small" style="margin-top:14px">Mô hình và số liệu đánh giá được lưu trong thư mục models/.</p>
      </aside>
    </div>
  </main>
  <script>
    let meta;
    let currentSample = null;
    const subject = document.getElementById('subject');
    const algorithm = document.getElementById('algorithm');
    const sampleSelect = document.getElementById('sample');
    const fieldsRoot = document.getElementById('fields');

    async function init() {
      meta = await (await fetch('/api/metadata')).json();
      for (const [key, item] of Object.entries(meta.subjects)) subject.add(new Option(item.label, key));
      for (const [key, item] of Object.entries(meta.algorithms)) algorithm.add(new Option(item.label, key));
      for (const item of meta.demo_profiles) sampleSelect.add(new Option(item.label, item.id));
      subject.addEventListener('change', renderFields);
      algorithm.addEventListener('change', renderMetrics);
      sampleSelect.addEventListener('change', loadSample);
      document.getElementById('reset').addEventListener('click', () => {
        currentSample = null; sampleSelect.value = ''; renderFields();
      });
      document.getElementById('predict').addEventListener('click', predict);
      renderFields();
    }

    function makeInput(name, value, min, max, step='0.1', label=null) {
      const wrap = document.createElement('div');
      const title = document.createElement('label');
      title.htmlFor = 'f-' + name;
      title.textContent = label || name;
      const input = document.createElement('input');
      input.type = 'number'; input.id = 'f-' + name; input.dataset.feature = name;
      input.min = min; input.max = max; input.step = step; input.required = true;
      input.value = value;
      wrap.append(title, input);
      return wrap;
    }

    function renderFields() {
      fieldsRoot.replaceChildren();
      for (const group of meta.feature_groups) {
        const details = document.createElement('details');
        details.open = ['toan','ngu_van','ngoai_ngu'].includes(group.key);
        const summary = document.createElement('summary'); summary.textContent = group.label; details.append(summary);
        const grid = document.createElement('div'); grid.className = 'grid';
        for (const name of group.features) {
          const grade = name.match(/_lop(\d+)$/)?.[1];
          const label = group.label + (grade ? ' lớp ' + grade : '');
          grid.append(makeInput(name, meta.defaults[name], 0, 10, '0.1', label));
        }
        details.append(grid); fieldsRoot.append(details);
      }
      const habits = document.createElement('details'); habits.open = true;
      const habitTitle = document.createElement('summary'); habitTitle.textContent = 'Thói quen học tập'; habits.append(habitTitle);
      const habitGrid = document.createElement('div'); habitGrid.className = 'grid single';
      habitGrid.append(makeInput('gio_tu_hoc_tuan', meta.defaults.gio_tu_hoc_tuan, 0, 30, '0.5', 'Giờ tự học mỗi tuần'));
      habitGrid.append(makeInput('ti_le_chuyen_can', (meta.defaults.ti_le_chuyen_can*100).toFixed(1), 0, 100, '0.1', 'Chuyên cần (%)'));
      const spec = meta.subjects[subject.value];
      const mockValue = currentSample ? currentSample.mock_scores[subject.value] : 5;
      habitGrid.append(makeInput(spec.mock_score, mockValue, 0, 10, '0.1', 'Điểm thi thử ' + spec.label));
      habits.append(habitGrid); fieldsRoot.append(habits);

      if (currentSample) {
        for (const [key, value] of Object.entries(currentSample.features)) {
          const input = fieldsRoot.querySelector('[data-feature="' + key + '"]');
          if (input) input.value = value;
        }
        const attendance = fieldsRoot.querySelector('[data-feature="ti_le_chuyen_can"]');
        if (attendance) attendance.value = (currentSample.features.ti_le_chuyen_can*100).toFixed(1);
        const mock = fieldsRoot.querySelector('[data-feature="' + spec.mock_score + '"]');
        if (mock) mock.value = currentSample.mock_scores[subject.value];
      }
      renderMetrics();
      document.getElementById('result').classList.remove('show');
    }

    async function loadSample() {
      const id = sampleSelect.value;
      currentSample = id ? await (await fetch('/api/sample/' + id)).json() : null;
      renderFields();
    }

    function renderMetrics() {
      const algorithmInfo = meta.algorithms[algorithm.value];
      const item = algorithmInfo.subjects[subject.value];
      const target = document.getElementById('metrics');
      target.replaceChildren();
      for (const [label, metrics] of [[algorithmInfo.label, item.test_metrics], ['Mốc: điểm lớp 9', item.baseline_test_metrics]]) {
        const box = document.createElement('div'); box.className = 'metric';
        const title = document.createElement('strong'); title.textContent = label;
        const line = document.createElement('span');
        line.textContent = 'MAE ' + metrics.mae.toFixed(3) + ' · RMSE ' + metrics.rmse.toFixed(3) + ' · R² ' + metrics.r2.toFixed(3);
        box.append(title, line); target.append(box);
      }
      const note = document.createElement('p');
      note.className = 'small';
      note.textContent = 'Train: ' + algorithmInfo.train_rows.toLocaleString('vi-VN') + ' hồ sơ.';
      target.append(note);
    }

    async function predict() {
      const status = document.getElementById('status');
      const button = document.getElementById('predict');
      const inputs = [...fieldsRoot.querySelectorAll('[data-feature]')];
      if (inputs.some(input => !input.reportValidity())) return;
      const features = {};
      for (const input of inputs) {
        const value = Number(input.value);
        features[input.dataset.feature] = input.dataset.feature === 'ti_le_chuyen_can' ? value / 100 : value;
      }
      button.disabled = true; status.textContent = 'Đang dự đoán…';
      try {
        const response = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({subject:subject.value, algorithm:algorithm.value, features})});
        const data = await response.json();
        if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Dữ liệu nhập chưa hợp lệ.');
        document.getElementById('result-subject').textContent = data.subject_label + ' · ' + meta.algorithms[algorithm.value].label;
        document.getElementById('result-score').textContent = data.prediction + ' / 10';
        document.getElementById('sample-comparison').textContent = currentSample
          ? 'Điểm mô phỏng trong hồ sơ mẫu: ' + currentSample.targets[subject.value].toFixed(2) + ' / 10.'
          : 'Bạn có thể nạp một hồ sơ mẫu để đối chiếu với nhãn mô phỏng.';
        document.getElementById('result').classList.add('show');
        status.textContent = '';
      } catch (error) { status.textContent = error.message; }
      finally { button.disabled = false; }
    }
    init().catch(error => { document.getElementById('status').textContent = 'Không tải được cấu hình: ' + error.message; });
  </script>
</body>
</html>"""


ELECTIVE_PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gợi ý môn tự chọn</title><style>
body{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f4f8f6;color:#18332d}header{background:#123c32;color:white;padding:24px max(18px,calc((100% - 1000px)/2))}main{max-width:1000px;margin:auto;padding:22px 16px 50px}.box{background:white;border:1px solid #dce6e2;border-radius:12px;padding:18px;margin:14px 0}.note{background:#fff5d9;padding:12px;border-radius:8px}.toolbar{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:15px}.toolbar>div{min-width:180px;flex:1}label{display:block;font-weight:600;margin:5px 0}select,input{font:inherit;padding:8px;border:1px solid #c7d5d0;border-radius:7px;width:100%;box-sizing:border-box}.fields{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.fields label{font-size:13px;font-weight:500}.btn{background:#176b53;color:white;border:0;border-radius:8px;padding:10px 16px;font:inherit;cursor:pointer}.result{display:none}.result.show{display:block}.row{margin:12px 0}.bar{background:#e8efec;height:10px;border-radius:6px;overflow:hidden}.bar i{display:block;background:#278463;height:100%}.row strong{display:flex;justify-content:space-between}.top{font-size:20px;color:#176b53}.small{font-size:13px;color:#536761}@media(max-width:700px){.fields{grid-template-columns:repeat(2,1fr)}}
</style></head><body><header><h1>Thử gợi ý 4 môn tự chọn</h1><a href="/" style="color:#d3e7df">← Quay lại dự đoán điểm</a></header><main>
<p class="note"><b>Lưu ý:</b> Nhãn yes/no hiện có biểu thị lựa chọn/lĩnh vực quan tâm trong dữ liệu mô phỏng; chưa xác nhận là môn học sinh thực sự đăng ký. Kết quả là xếp hạng xác suất theo các nhãn này, không phải tư vấn tuyển sinh.</p>
<section class="box"><div class="toolbar"><div><label for="algorithm">Mô hình phân loại</label><select id="algorithm"><option value="catboost">CatBoostClassifier</option><option value="svc">SVC (RBF)</option></select></div><div><label for="sample">Hồ sơ kiểm thử</label><select id="sample"><option value="">Nhập hồ sơ</option></select></div></div><p class="small">Nhập điểm học tập từ 0 đến 10. Giá trị ban đầu là trung vị của tập train.</p><div id="fields"></div><p><button class="btn" id="run">Xếp hạng 4 môn</button> <span id="status"></span></p></section>
<section class="box result" id="result"><h2>Bốn môn có xác suất quan tâm cao nhất</h2><div id="top"></div><p class="small" id="note"></p></section>
<section class="box"><h3>Kết quả đánh giá trên tập test</h3><div id="metrics"></div><p class="small">Top-4 mean match là tỷ lệ trung bình các môn trong 4 gợi ý cũng có nhãn yes trong dữ liệu test. Không phải độ chính xác trúng tuyển hay đăng ký thực tế.</p></section></main>
<script>
let meta;const fields=document.getElementById('fields');
async function init(){const res=await fetch('/api/electives/metadata');meta=await res.json();const labels={toan:'Toán',ngu_van:'Ngữ văn',ngoai_ngu:'Ngoại ngữ',khoa_hoc_tu_nhien:'Khoa học tự nhiên',lich_su_dia_ly:'Lịch sử và Địa lý',giao_duc_cong_dan:'Giáo dục công dân',tin_hoc:'Tin học',cong_nghe:'Công nghệ',giao_duc_the_chat:'Giáo dục thể chất',am_nhac:'Âm nhạc',mi_thuat:'Mỹ thuật',noi_dung_giao_duc_dia_phuong:'Giáo dục địa phương',hoat_dong_trai_nghiem_huong_nghiep:'Trải nghiệm, hướng nghiệp'};for(const g of meta.feature_groups){const box=document.createElement('div');box.className='box';const title=document.createElement('h3');title.textContent=labels[g.key]||g.label;box.append(title);const grid=document.createElement('div');grid.className='fields';for(const name of g.features){const match=name.match(/^(.*)_lop(\d+)$/);const wrap=document.createElement('label');wrap.textContent=match?(labels[match[1]]||match[1])+' lớp '+match[2]:name;const input=document.createElement('input');input.type='number';input.min='0';input.max='10';input.step='0.1';input.value=meta.defaults[name];input.dataset.key=name;wrap.append(input);grid.append(wrap)}box.append(grid);fields.append(box)}for(const p of meta.demo_profiles){const o=document.createElement('option');o.value=p.id;o.textContent=p.label;document.getElementById('sample').append(o)}renderMetrics()}
function renderMetrics(){const a=document.getElementById('algorithm').value;const m=meta.algorithms[a];const box=document.getElementById('metrics');box.textContent=m.label+' · train '+meta.train_rows.toLocaleString('vi-VN')+' hồ sơ · test '+meta.test_rows.toLocaleString('vi-VN')+' hồ sơ · top-4 mean match '+(100*m.top_four_mean_interest_match).toFixed(1)+'%'}
document.getElementById('algorithm').addEventListener('change',renderMetrics);
document.getElementById('sample').addEventListener('change',async function(){if(!this.value)return;const s=await(await fetch('/api/electives/sample/'+this.value)).json();for(const input of fields.querySelectorAll('[data-key]'))input.value=s.features[input.dataset.key]});
document.getElementById('run').addEventListener('click',async function(){const status=document.getElementById('status');status.textContent='Đang dự đoán…';const features={};for(const input of fields.querySelectorAll('[data-key]'))features[input.dataset.key]=Number(input.value);try{const res=await fetch('/api/electives/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({algorithm:document.getElementById('algorithm').value,features})});const data=await res.json();if(!res.ok)throw Error(JSON.stringify(data.detail));const top=document.getElementById('top');top.replaceChildren();for(let i=0;i<data.top_four.length;i++){const x=data.top_four[i];const row=document.createElement('div');row.className='row';const title=document.createElement('strong');title.textContent=(i+1)+'. '+x.label+' — '+(100*x.probability).toFixed(1)+'%';const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('i');fill.style.width=(100*x.probability)+'%';bar.append(fill);row.append(title,bar);top.append(row)}document.getElementById('note').textContent=data.note;document.getElementById('result').classList.add('show');status.textContent=''}catch(e){status.textContent='Lỗi: '+e.message}});
init().catch(e=>document.getElementById('status').textContent='Không tải được dữ liệu: '+e.message);
</script></body></html>"""
