from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "experiments" / "figures"


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS / "regression_metrics.csv")
    metrics = metrics[(metrics.status == "ok") & metrics.mae.notna()]
    for target in sorted(metrics.target.unique()):
        for feature_set in sorted(metrics.feature_set.unique()):
            part = metrics[(metrics.target == target) & (metrics.feature_set == feature_set)]
            summary = part.groupby(["model", "train_groups"]).mae.agg(["mean", "std"]).reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            for model, group in summary.groupby("model"):
                ax.plot(group.train_groups, group["mean"], marker="o", label=model)
                ax.fill_between(group.train_groups,
                    group["mean"] - group["std"].fillna(0),
                    group["mean"] + group["std"].fillna(0), alpha=.15)
            ax.set(xscale="log", xlabel="Số nhóm hồ sơ trong train", ylabel="MAE test",
                   title=f"Learning curve — {target} — {feature_set}")
            ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout()
            fig.savefig(FIGURES / f"learning_curve_{target}_{feature_set}.png", dpi=150)
            plt.close(fig)
    timing = pd.read_csv(RESULTS / "regression_timing.csv")
    bins = timing[(timing.status == "residual_bin") & (timing.train_groups == 6000) &
                  (timing.feature_set == "full_with_mock_score")]
    for target in sorted(bins.target.unique()):
        part = bins[bins.target == target]
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, group in part.groupby("model"):
            group = group.sort_values("grade9_mean")
            ax.plot(group.grade9_mean, group.residual_mae, marker="o", label=model)
        ax.set(xlabel="Điểm lớp 9 trung bình trong quintile", ylabel="MAE test trong quintile",
               title=f"Sai số theo điểm lớp 9 — {target}")
        ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(FIGURES / f"residual_{target}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
