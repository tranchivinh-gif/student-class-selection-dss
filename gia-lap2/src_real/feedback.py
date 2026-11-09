"""Collect confirmed Top-2 feedback and retrain without touching locked test data."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from .config import (
    ALLOWED_COMBINATIONS, FEATURES, ID_COLUMN, MODEL_PATH, OUTPUT_DIR, RANDOM_SEED,
)
from .data import combination_name, load_features
from .model import RealCombinationModel
from .train import evaluate

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

CONFIRMED_COLUMN = "confirmed_combination_key"
HISTORY_PATH = OUTPUT_DIR / "feedback_history.csv"


def init_template(output: Path) -> None:
    columns = [ID_COLUMN, *FEATURES, "top1_key", "top1_probability",
               "top2_key", "top2_probability", CONFIRMED_COLUMN]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Đã tạo mẫu phản hồi: {output}")


def prepare(input_path: Path, output: Path, model_path: Path = MODEL_PATH) -> None:
    data = load_features(input_path).copy()
    if ID_COLUMN not in data.columns:
        data[ID_COLUMN] = [f"NEW_{i + 1:05d}" for i in range(len(data))]
    if data[ID_COLUMN].astype(str).duplicated().any():
        raise ValueError("Mã học sinh trong dữ liệu mới bị trùng.")
    model = RealCombinationModel.load(model_path)
    recommendation = model.recommend(data, 2)
    wide = recommendation.pivot(index=ID_COLUMN, columns="rank")
    prepared = data[[ID_COLUMN, *FEATURES]].copy()
    prepared[ID_COLUMN] = prepared[ID_COLUMN].astype(str)
    indexed = wide.reindex(prepared[ID_COLUMN])
    prepared["top1_key"] = indexed[("combination_key", 1)].to_numpy()
    prepared["top1_name"] = indexed[("combination", 1)].to_numpy()
    prepared["top1_probability"] = indexed[("probability", 1)].to_numpy()
    prepared["top2_key"] = indexed[("combination_key", 2)].to_numpy()
    prepared["top2_name"] = indexed[("combination", 2)].to_numpy()
    prepared["top2_probability"] = indexed[("probability", 2)].to_numpy()
    prepared[CONFIRMED_COLUMN] = ""
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Đã tạo Top-2 cho {len(prepared)} học sinh tại {output}")
    print(f"Điền cột {CONFIRMED_COLUMN} sau khi có lựa chọn thực tế.")


def _load_confirmed(path: Path, output_dir: Path) -> pd.DataFrame:
    feedback = pd.read_csv(path, encoding="utf-8-sig", dtype={ID_COLUMN: str, CONFIRMED_COLUMN: str})
    required = [ID_COLUMN, *FEATURES, "top1_key", "top2_key", CONFIRMED_COLUMN]
    missing = [column for column in required if column not in feedback.columns]
    if missing:
        raise ValueError(f"File phản hồi thiếu cột: {missing[:8]}")
    feedback[CONFIRMED_COLUMN] = feedback[CONFIRMED_COLUMN].fillna("").str.strip().str.zfill(9)
    feedback = feedback[feedback[CONFIRMED_COLUMN].ne("000000000")].copy()
    if feedback.empty:
        raise ValueError("Chưa có đáp án xác nhận nào để học.")
    invalid = sorted(set(feedback[CONFIRMED_COLUMN]) - set(ALLOWED_COMBINATIONS))
    if invalid:
        raise ValueError(f"Đáp án không thuộc bốn tổ hợp được hỗ trợ: {invalid}")
    if feedback[ID_COLUMN].duplicated().any():
        raise ValueError("Một học sinh xuất hiện nhiều lần trong file phản hồi.")
    numeric = feedback[FEATURES].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 10)).any():
        raise ValueError("Điểm phản hồi phải đầy đủ và nằm trong khoảng 0–10.")
    feedback[FEATURES] = numeric

    development = pd.read_csv(output_dir / "development.csv", encoding="utf-8-sig", dtype={ID_COLUMN: str})
    locked = pd.read_csv(output_dir / "locked_independent_test.csv", encoding="utf-8-sig", dtype={ID_COLUMN: str})
    ids = set(feedback[ID_COLUMN])
    test_overlap = ids & set(locked[ID_COLUMN])
    if test_overlap:
        raise ValueError(f"Cấm học từ tập test khóa; phát hiện {len(test_overlap)} mã trùng.")
    development_overlap = ids & set(development[ID_COLUMN])
    if development_overlap:
        raise ValueError(f"Phản hồi phải là học sinh mới; có {len(development_overlap)} mã đã ở development.")
    feedback["was_in_top2"] = (
        feedback[CONFIRMED_COLUMN].eq(feedback["top1_key"].astype(str).str.zfill(9))
        | feedback[CONFIRMED_COLUMN].eq(feedback["top2_key"].astype(str).str.zfill(9))
    )
    feedback["confirmed_at"] = datetime.now().astimezone().isoformat()
    return feedback


def learn(feedback_path: Path, output_dir: Path = OUTPUT_DIR,
          repeats: int = 3, promote: bool = False) -> Path:
    new_feedback = _load_confirmed(feedback_path, output_dir)
    history_path = output_dir / "feedback_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path, encoding="utf-8-sig", dtype={ID_COLUMN: str, CONFIRMED_COLUMN: str})
        duplicate = set(new_feedback[ID_COLUMN]) & set(history[ID_COLUMN])
        if duplicate:
            raise ValueError(f"Có {len(duplicate)} học sinh đã được học từ phản hồi trước đó.")
        history = pd.concat([history, new_feedback], ignore_index=True)
    else:
        history = new_feedback

    development = pd.read_csv(
        output_dir / "development.csv", encoding="utf-8-sig", dtype={ID_COLUMN: str}
    )
    original = development[[ID_COLUMN, *FEATURES, "combination_key"]].copy()
    added = history[[ID_COLUMN, *FEATURES, CONFIRMED_COLUMN]].rename(
        columns={CONFIRMED_COLUMN: "combination_key"}
    )
    training = pd.concat([original, added], ignore_index=True)
    y = training["combination_key"].astype(str).str.zfill(9)

    splitter = RepeatedStratifiedKFold(5, n_repeats=repeats, random_state=RANDOM_SEED)
    metrics = []
    for run_index, (train_idx, validation_idx) in enumerate(splitter.split(training, y), 1):
        model = RealCombinationModel(RANDOM_SEED + run_index).fit(
            training.iloc[train_idx][FEATURES], y.iloc[train_idx]
        )
        result, _ = evaluate(model, training.iloc[validation_idx][FEATURES], y.iloc[validation_idx])
        metrics.append({"run": run_index, **result})
    cv = pd.DataFrame(metrics)
    final = RealCombinationModel().fit(training[FEATURES], y)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_path = output_dir / f"candidate_feedback_{timestamp}.joblib"
    final.save(candidate_path)
    cv_path = output_dir / f"feedback_cv_{timestamp}.csv"
    cv.to_csv(cv_path, index=False, encoding="utf-8-sig")
    history.to_csv(history_path, index=False, encoding="utf-8-sig")

    if promote:
        promote_candidate(candidate_path, output_dir)
    report = {
        "feedback_file": str(feedback_path), "new_feedback_rows": len(new_feedback),
        "total_feedback_rows": len(history), "training_rows": len(training),
        "top2_coverage_in_new_feedback": float(new_feedback["was_in_top2"].mean()),
        "cv_mean": cv.drop(columns="run").mean().to_dict(),
        "candidate": str(candidate_path), "promoted": promote,
        "locked_test_used": False,
    }
    report_path = output_dir / f"feedback_report_{timestamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã học {len(new_feedback)} phản hồi mới; tổng dữ liệu train={len(training)}")
    print(f"Top-2 đã chứa đáp án thật: {new_feedback['was_in_top2'].mean():.2%}")
    print(f"CV Exact Match mới: {report['cv_mean']['exact_match']:.4f}")
    print(f"Candidate: {candidate_path}")
    print("Đã thay champion." if promote else "Chưa thay champion; thêm --promote để áp dụng model mới.")
    return candidate_path


def promote_candidate(candidate_path: Path, output_dir: Path = OUTPUT_DIR) -> None:
    if not candidate_path.exists():
        raise FileNotFoundError(f"Không tìm thấy candidate: {candidate_path}")
    if not candidate_path.name.startswith("candidate_feedback_"):
        raise ValueError("Chỉ cho phép promote artifact candidate_feedback_*.joblib")
    # Loading also verifies that this is the expected artifact type.
    RealCombinationModel.load(candidate_path)
    champion_path = output_dir / "champion.joblib"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if champion_path.exists():
        shutil.copy2(champion_path, output_dir / f"champion_before_feedback_{timestamp}.joblib")
    shutil.copy2(candidate_path, champion_path)
    print(f"Đã áp dụng candidate: {candidate_path}")
    print(f"Champion mới: {champion_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-2 feedback workflow for src_real")
    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser("init", help="Tạo file mẫu phản hồi trống")
    init_parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "feedback_template.csv")
    prepare_parser = sub.add_parser("prepare", help="Tạo Top-2 cho học sinh mới")
    prepare_parser.add_argument("--input", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "feedback_pending.csv")
    prepare_parser.add_argument("--model", type=Path, default=MODEL_PATH)
    learn_parser = sub.add_parser("learn", help="Học lại từ đáp án đã xác nhận")
    learn_parser.add_argument("--feedback", type=Path, required=True)
    learn_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    learn_parser.add_argument("--repeats", type=int, default=3)
    learn_parser.add_argument("--promote", action="store_true")
    promote_parser = sub.add_parser("promote", help="Áp dụng một candidate đã kiểm tra")
    promote_parser.add_argument("--candidate", type=Path, required=True)
    promote_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.command == "init":
        init_template(args.output)
    elif args.command == "prepare":
        prepare(args.input, args.output, args.model)
    elif args.command == "promote":
        promote_candidate(args.candidate, args.output_dir)
    else:
        learn(args.feedback, args.output_dir, args.repeats, args.promote)


if __name__ == "__main__":
    main()

