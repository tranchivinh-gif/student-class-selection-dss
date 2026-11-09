from __future__ import annotations

import argparse, json, shutil, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from .config import DATA_PATH, DEPLOYED_MODEL_PATHS, FEATURE_SETS, MODEL_PATH, OUTPUT_DIR, RANDOM_SEED
from .data import load_training_data
from .model import RealCombinationModel

ALGORITHMS = ("baseline", "xgboost", "catboost")
DISPLAY_NAMES = {"baseline": "Baseline", "xgboost": "XGBoost", "catboost": "CatBoost"}
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"): stream.reconfigure(encoding="utf-8")


def evaluate(model: RealCombinationModel, x: pd.DataFrame, truth: pd.Series) -> tuple[dict, np.ndarray]:
    probability, classes = model.predict_proba(x), np.asarray(model.classes_)
    predicted, actual = classes[probability.argmax(axis=1)], truth.astype(str).to_numpy()
    order = np.argsort(-probability, axis=1)
    metrics = {
        "exact_match": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "top2_accuracy": float(np.mean([t in c for t, c in zip(actual, classes[order[:, :2]])])),
        "top3_accuracy": float(np.mean([t in c for t, c in zip(actual, classes[order[:, :3]])])),
    }
    return metrics, predicted


def _composite(row: pd.Series) -> float:
    return float(.5 * row["exact_match"] + .3 * row["macro_f1"] + .2 * row["balanced_accuracy"])


def promote_models(run_dir: Path, data_path: Path, report: dict) -> None:
    champion = report["selected_model"]
    sources = {name: run_dir / f"{name}.joblib" for name in ALGORITHMS}
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing: raise FileNotFoundError(f"Thiếu artifact: {missing}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir, timestamp = OUTPUT_DIR / "backups", datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, destination in DEPLOYED_MODEL_PATHS.items():
        if destination.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup_dir / f"{name}_{timestamp}.joblib")
        shutil.copy2(sources[name], destination)
    if MODEL_PATH.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MODEL_PATH, backup_dir / f"champion_{timestamp}.joblib")
    shutil.copy2(sources[champion], MODEL_PATH)
    manifest = {
        "active_model": str(MODEL_PATH), "selected_model": champion,
        "deployed_models": {name: str(path) for name, path in DEPLOYED_MODEL_PATHS.items()},
        "source_run": str(run_dir), "source_data": str(data_path),
        "promoted_at": datetime.now().astimezone().isoformat(), "selection_report": report,
    }
    (OUTPUT_DIR / "deployment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def promote_model(run_dir: Path, data_path: Path, metrics: dict | None = None) -> None:
    report_path = run_dir / "selection_report.json"
    if not report_path.exists():
        raise FileNotFoundError("Run phải có selection_report.json và ba artifact.")
    promote_models(run_dir, data_path, json.loads(report_path.read_text(encoding="utf-8")))


def run(data_path: Path, output: Path, folds: int = 5, repeats: int = 5,
        test_size: float = .2, promote: bool = False, feature_set: str = "all") -> None:
    if not data_path.exists(): raise FileNotFoundError(f"Không tìm thấy dữ liệu: {data_path}")
    if folds < 2 or repeats < 1: raise ValueError("folds >= 2 và repeats >= 1.")
    output.mkdir(parents=True, exist_ok=True)
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature-set phải thuộc {list(FEATURE_SETS)}")
    feature_columns = list(FEATURE_SETS[feature_set])
    data = load_training_data(data_path, feature_columns)
    development, independent_test = train_test_split(
        data, test_size=test_size, random_state=RANDOM_SEED, shuffle=True,
        stratify=data["combination_key"],
    )
    development, independent_test = development.reset_index(drop=True), independent_test.reset_index(drop=True)
    development.to_csv(output / "development.csv", index=False, encoding="utf-8-sig")
    independent_test.to_csv(output / "locked_independent_test.csv", index=False, encoding="utf-8-sig")
    y = development["combination_key"]
    split_indices = list(RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=repeats, random_state=RANDOM_SEED
    ).split(development, y))
    records, predictions = [], []
    for algorithm in ALGORITHMS:
        for run_index, (train_idx, validation_idx) in enumerate(split_indices, 1):
            repeat, fold = (run_index - 1) // folds + 1, (run_index - 1) % folds + 1
            model = RealCombinationModel(RANDOM_SEED + run_index, algorithm, feature_columns).fit(development.iloc[train_idx][feature_columns], y.iloc[train_idx])
            fold_metrics, predicted = evaluate(model, development.iloc[validation_idx][feature_columns], y.iloc[validation_idx])
            records.append({"model": algorithm, "repeat": repeat, "fold": fold, **fold_metrics})
            truth = y.iloc[validation_idx].to_numpy()
            for position, row_index in enumerate(validation_idx):
                predictions.append({"model": algorithm, "repeat": repeat, "fold": fold,
                    "ma_hoc_sinh": development.iloc[row_index]["ma_hoc_sinh"],
                    "actual": truth[position], "predicted": predicted[position],
                    "correct": truth[position] == predicted[position]})
            print(f"{algorithm} repeat {repeat}/{repeats}, fold {fold}/{folds}: exact={fold_metrics['exact_match']:.4f}")
    metrics = pd.DataFrame(records)
    metrics.to_csv(output / "repeated_cv_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(output / "repeated_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metric_cols = ["exact_match", "balanced_accuracy", "macro_f1", "top2_accuracy", "top3_accuracy"]
    metrics.groupby("model")[metric_cols].agg(["mean", "std", "min", "max"]).to_csv(output / "model_comparison_detailed.csv", encoding="utf-8-sig")
    means, stds = metrics.groupby("model")[metric_cols].mean(), metrics.groupby("model")[metric_cols].std()
    means["composite_score"] = means.apply(_composite, axis=1)
    ranking = means.sort_values("composite_score", ascending=False).reset_index()
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    for metric in ("exact_match", "balanced_accuracy", "macro_f1"):
        ranking[f"{metric}_std"] = ranking["model"].map(stds[metric])
    ranking.to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")
    champion = str(ranking.iloc[0]["model"])

    test_results = {}
    for algorithm in ALGORITHMS:
        final = RealCombinationModel(RANDOM_SEED, algorithm, feature_columns).fit(development[feature_columns], y)
        test_metrics, test_prediction = evaluate(final, independent_test[feature_columns], independent_test["combination_key"])
        test_results[algorithm] = test_metrics
        final.save(output / f"{algorithm}.joblib")
        pd.DataFrame({"ma_hoc_sinh": independent_test["ma_hoc_sinh"],
            "actual": independent_test["combination_key"], "predicted": test_prediction,
            "correct": independent_test["combination_key"].to_numpy() == test_prediction,
        }).to_csv(output / f"independent_test_predictions_{algorithm}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame.from_dict(test_results, orient="index").rename_axis("model").reset_index().to_csv(output / "independent_test_model_comparison.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(output / f"{champion}.joblib", output / "champion.joblib")
    baseline_score, champion_score = float(means.loc["baseline", "composite_score"]), float(means.loc[champion, "composite_score"])
    report = {"selected_model": champion, "display_name": DISPLAY_NAMES[champion],
        "selection_rule": "0.5*exact_match + 0.3*macro_f1 + 0.2*balanced_accuracy on repeated CV",
        "selection_used_independent_test": False, "composite_score": champion_score,
        "improvement_over_baseline": champion_score - baseline_score,
        "reason": [f"{DISPLAY_NAMES[champion]} có composite score CV cao nhất ({champion_score:.4f}).",
            f"Chênh lệch so với baseline: {champion_score - baseline_score:+.4f}.",
            "Independent test không được dùng để chọn model."],
        "cv_ranking": ranking.to_dict(orient="records"), "independent_test_metrics": test_results}
    (output / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {"run_name": output.name, "created_at": datetime.now().astimezone().isoformat(),
        "data": str(data_path), "rows": len(data), "features": len(feature_columns),
        "feature_set": feature_set, "feature_columns": feature_columns,
        "development_rows": len(development), "independent_test_rows": len(independent_test),
        "test_size": test_size, "classes": list(final.classes_), "folds": folds, "repeats": repeats,
        "algorithms": list(ALGORITHMS), "selected_model": champion, "promoted_to_application": promote}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if promote: promote_models(output, data_path, report)
    (output / "MODEL_LOCATION.txt").write_text(f"CHAMPION: {champion}\n{(output / 'champion.joblib').resolve()}\nTRIỂN KHAI: {'CÓ' if promote else 'CHƯA'}\n", encoding="utf-8")
    print(f"Champion theo CV: {champion}\nBáo cáo: {output / 'selection_report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH); parser.add_argument("--run-name")
    parser.add_argument("--output", type=Path); parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5); parser.add_argument("--test-size", type=float, default=.2)
    parser.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="all",
                        help="all: 96 cột; annual: chỉ 32 cột điểm cả năm")
    parser.add_argument("--promote", action="store_true", help="Deploy cả ba model và đặt champion làm mặc định")
    args = parser.parse_args(); run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if any(char in run_name for char in '<>:"/\\|?*') or run_name in {".", ".."}: parser.error("run-name không hợp lệ.")
    output = args.output or OUTPUT_DIR / "runs" / run_name
    if output.exists() and any(output.iterdir()): parser.error(f"Thư mục kết quả đã có dữ liệu: {output}")
    run(args.data, output, args.folds, args.repeats, args.test_size, args.promote, args.feature_set)


if __name__ == "__main__": main()
