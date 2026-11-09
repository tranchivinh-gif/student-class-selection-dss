"""Offline simulation of learning from confirmed Top-2 feedback.

This experiment uses development data only. It never reads the locked independent
test and never replaces the production champion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, OUTPUT_DIR, RANDOM_SEED
from .model import RealCombinationModel
from .train import evaluate


def run(output_dir: Path = OUTPUT_DIR, repeats: int = 10) -> pd.DataFrame:
    development = pd.read_csv(
        output_dir / "development.csv", encoding="utf-8-sig",
        dtype={"ma_hoc_sinh": str, "combination_key": str},
    )
    development["combination_key"] = development["combination_key"].str.zfill(9)
    records = []
    for repeat in range(1, repeats + 1):
        seed = RANDOM_SEED + repeat
        pool, evaluation = train_test_split(
            development, test_size=.20, random_state=seed,
            stratify=development["combination_key"],
        )
        base, feedback = train_test_split(
            pool, test_size=.25, random_state=seed,
            stratify=pool["combination_key"],
        )
        base, feedback, evaluation = (
            frame.reset_index(drop=True) for frame in (base, feedback, evaluation)
        )

        before = RealCombinationModel(seed).fit(base[FEATURES], base["combination_key"])
        before_metrics, _ = evaluate(
            before, evaluation[FEATURES], evaluation["combination_key"]
        )
        feedback_probability = before.predict_proba(feedback[FEATURES])
        classes = np.asarray(before.classes_)
        feedback_top2 = classes[np.argsort(-feedback_probability, axis=1)[:, :2]]
        truth = feedback["combination_key"].to_numpy()
        in_top2 = np.asarray([actual in candidates for actual, candidates in zip(truth, feedback_top2)])

        top2_confirmed = feedback.loc[in_top2]
        top2_training = pd.concat([base, top2_confirmed], ignore_index=True)
        top2_model = RealCombinationModel(seed + 1000).fit(
            top2_training[FEATURES], top2_training["combination_key"]
        )
        top2_metrics, _ = evaluate(
            top2_model, evaluation[FEATURES], evaluation["combination_key"]
        )

        all_training = pd.concat([base, feedback], ignore_index=True)
        all_model = RealCombinationModel(seed + 2000).fit(
            all_training[FEATURES], all_training["combination_key"]
        )
        all_metrics, _ = evaluate(
            all_model, evaluation[FEATURES], evaluation["combination_key"]
        )

        record = {
            "repeat": repeat, "base_rows": len(base), "feedback_rows": len(feedback),
            "evaluation_rows": len(evaluation), "confirmed_in_top2_rows": int(in_top2.sum()),
            "feedback_top2_coverage": float(in_top2.mean()),
        }
        for prefix, metrics in (
            ("before", before_metrics), ("learn_top2_only", top2_metrics),
            ("learn_all_confirmed", all_metrics),
        ):
            record.update({f"{prefix}_{name}": value for name, value in metrics.items()})
        records.append(record)
        print(
            f"repeat {repeat}/{repeats}: exact "
            f"{before_metrics['exact_match']:.3f} -> {top2_metrics['exact_match']:.3f} "
            f"(top2-only), {all_metrics['exact_match']:.3f} (all)"
        )

    result = pd.DataFrame(records)
    result["delta_top2_only_exact"] = (
        result["learn_top2_only_exact_match"] - result["before_exact_match"]
    )
    result["delta_all_exact"] = (
        result["learn_all_confirmed_exact_match"] - result["before_exact_match"]
    )
    result.to_csv(output_dir / "feedback_experiment_runs.csv", index=False, encoding="utf-8-sig")
    numeric = result.drop(columns="repeat")
    summary = numeric.agg(["mean", "std", "min", "max"])
    summary.to_csv(output_dir / "feedback_experiment_summary.csv", encoding="utf-8-sig")
    report = {
        "scope": "development only; locked independent test untouched",
        "production_model_changed": False,
        "repeats": repeats,
        "mean_before_exact": float(result["before_exact_match"].mean()),
        "mean_after_top2_only_exact": float(result["learn_top2_only_exact_match"].mean()),
        "mean_after_all_confirmed_exact": float(result["learn_all_confirmed_exact_match"].mean()),
        "mean_delta_top2_only_exact": float(result["delta_top2_only_exact"].mean()),
        "mean_delta_all_exact": float(result["delta_all_exact"].mean()),
        "mean_feedback_top2_coverage": float(result["feedback_top2_coverage"].mean()),
    }
    (output_dir / "feedback_experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    run(args.output_dir, args.repeats)


if __name__ == "__main__":
    main()

