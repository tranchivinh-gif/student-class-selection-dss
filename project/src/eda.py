"""Tạo bảng tóm tắt và biểu đồ thăm dò cho dữ liệu pilot tổng hợp."""

from __future__ import annotations

import argparse
from itertools import combinations
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from schema import CLASS_LABEL_COLUMN, STUDENT_ID_COLUMN, score_column


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "generator_config"
DEFAULT_SEED = 20261108


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Cấu hình bộ sinh phải là một ánh xạ YAML: {path}")
    return config


def run_eda(
    data_path: Path,
    config: dict[str, Any],
    seed: int,
    output_directory: Path,
    mean_sampling_probabilities: dict[str, float],
    correlation_validation: dict[str, Any],
    interest_career_summary: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    score_config = config["scores"]
    feature_columns = [
        score_column(subject, grade)
        for subject in score_config["subjects"]
        for grade in score_config["grades"]
    ]
    expected_columns = [STUDENT_ID_COLUMN, *feature_columns, CLASS_LABEL_COLUMN]
    data = pd.read_csv(data_path)
    if list(data.columns) != expected_columns:
        raise ValueError(
            f"{data_path}: cột phải đúng thứ tự {expected_columns}, "
            f"thực tế là {list(data.columns)}"
        )
    if data.empty:
        raise ValueError(f"{data_path}: dữ liệu đang rỗng")
    if data[STUDENT_ID_COLUMN].isna().any() or data[STUDENT_ID_COLUMN].duplicated().any():
        raise ValueError(f"{data_path}: ma_hoc_sinh phải có giá trị và không trùng")

    scores = data[feature_columns].apply(pd.to_numeric, errors="raise")
    score_values = scores.to_numpy(dtype=float)
    lower, upper = score_config["score_bounds"]
    decimal_places = score_config["decimal_places"]
    if not np.all(np.isfinite(score_values)):
        raise ValueError(f"{data_path}: điểm có giá trị không hữu hạn")
    if np.any(score_values < lower) or np.any(score_values > upper):
        raise ValueError(f"{data_path}: scores fall outside [{lower}, {upper}]")
    class_names = list(config["labels"]["classes"])
    if data[CLASS_LABEL_COLUMN].isna().any() or not set(data[CLASS_LABEL_COLUMN]) <= set(class_names):
        raise ValueError(f"{data_path}: nhan_lop có giá trị thiếu hoặc không xác định")

    output_directory.mkdir(parents=True, exist_ok=True)
    summary = scores.describe(percentiles=[0.25, 0.5, 0.75]).T
    summary = summary.rename(
        columns={
            "count": "so_luong",
            "mean": "trung_binh",
            "std": "do_lech_chuan",
            "min": "nho_nhat",
            "25%": "phan_vi_25",
            "50%": "trung_vi",
            "75%": "phan_vi_75",
            "max": "lon_nhat",
        }
    )
    scale = 10 ** score_config["decimal_places"]
    summary["so_gia_tri_khac_nhau"] = scores.nunique()
    summary["so_o_bang_can_duoi"] = scores.eq(lower).sum()
    summary["so_o_bang_can_tren"] = scores.eq(upper).sum()
    summary["phan_tram_cat"] = (
        (scores.eq(lower) | scores.eq(upper)).mean() * 100
    ).round(4)
    summary["so_vi_pham_do_chinh_xac"] = (
        ~np.isclose(scores.to_numpy() * scale, np.round(scores.to_numpy() * scale))
    ).sum(axis=0)
    summary_path = output_directory / "score_summary.csv"
    summary.to_csv(summary_path, encoding="utf-8")

    counts = data[CLASS_LABEL_COLUMN].value_counts().reindex(class_names, fill_value=0)
    subject_columns = {
        subject: [score_column(subject, grade) for grade in score_config["grades"]]
        for subject in score_config["subjects"]
    }
    subject_summary = pd.DataFrame(
        [
            {
                "mon": subject,
                "mean": float(scores[columns].to_numpy().mean()),
                "standard_deviation": float(scores[columns].to_numpy().std(ddof=1)),
            }
            for subject, columns in subject_columns.items()
        ]
    )
    subject_summary.to_csv(
        output_directory / "score_distribution_by_subject.csv",
        index=False,
        encoding="utf-8",
    )
    grade_summary = pd.DataFrame(
        [
            {
                "khoi_lop": grade,
                "mean": float(
                    scores[
                        [score_column(subject, grade) for subject in score_config["subjects"]]
                    ].to_numpy().mean()
                ),
                "standard_deviation": float(
                    scores[
                        [score_column(subject, grade) for subject in score_config["subjects"]]
                    ].to_numpy().std(ddof=1)
                ),
            }
            for grade in score_config["grades"]
        ]
    )
    grade_summary.to_csv(
        output_directory / "score_distribution_by_grade.csv",
        index=False,
        encoding="utf-8",
    )
    rounded_scores = scores.round(decimal_places)
    total_score_cells = rounded_scores.size
    unique_rounded_values = int(np.unique(rounded_scores.to_numpy()).size)
    duplicate_after_rounding_percent = float(
        (total_score_cells - unique_rounded_values) / total_score_cells * 100
    )
    clipping_lower_percent = float((scores == lower).to_numpy().mean() * 100)
    clipping_upper_percent = float((scores == upper).to_numpy().mean() * 100)
    subject_means = pd.DataFrame(
        {
            subject: scores[columns].mean(axis=1)
            for subject, columns in subject_columns.items()
        }
    )
    subject_correlation = subject_means.corr()
    subject_correlation.to_csv(
        output_directory / "subject_correlations.csv",
        encoding="utf-8",
    )
    within_subject_correlations = pd.DataFrame(
        [
            {
                "mon": subject,
                "khoi_lop_a": grade_a,
                "khoi_lop_b": grade_b,
                "correlation": float(
                    scores[
                        [score_column(subject, grade_a), score_column(subject, grade_b)]
                    ].corr().iloc[0, 1]
                ),
            }
            for subject, (grade_a, grade_b) in (
                (subject, pair)
                for subject in subject_columns
                for pair in combinations(score_config["grades"], 2)
            )
        ]
    )
    within_subject_correlations.to_csv(
        output_directory / "within_subject_year_correlations.csv",
        index=False,
        encoding="utf-8",
    )
    label_summary = pd.DataFrame(
        {
            CLASS_LABEL_COLUMN: class_names,
            "so_luong": [int(counts[name]) for name in class_names],
            "ty_le_phan_tram": [round(float(counts[name] / len(data) * 100), 2) for name in class_names],
        }
    )
    label_summary.to_csv(output_directory / "label_distribution.csv", index=False, encoding="utf-8")
    softmax_summary = pd.DataFrame(
        {
            CLASS_LABEL_COLUMN: class_names,
            "xac_suat_softmax_trung_binh": [
                mean_sampling_probabilities[name] for name in class_names
            ],
        }
    )
    softmax_summary.to_csv(
        output_directory / "mean_softmax_probabilities.csv",
        index=False,
        encoding="utf-8",
    )
    ability_names = config["scores"]["ability_names"]
    correlation_matrix = pd.DataFrame(
        correlation_validation["matrix"],
        index=ability_names,
        columns=ability_names,
    )
    correlation_matrix.to_csv(
        output_directory / "correlation_matrix_R_Z.csv",
        encoding="utf-8",
    )
    interest_summary = pd.DataFrame(
        [
            {"nhom": group, "chieu": dimension, **values}
            for group, dimensions in interest_career_summary.items()
            for dimension, values in dimensions.items()
        ]
    )
    interest_summary.to_csv(
        output_directory / "interest_career_dimension_summary.csv",
        index=False,
        encoding="utf-8",
    )
    correlation_matrix_markdown = [
        "| Năng lực | " + " | ".join(ability_names) + " |",
        "|---" + "|---:" * len(ability_names) + "|",
        *[
            f"| {name} | "
            + " | ".join(f"{value:.4f}" for value in correlation_matrix.loc[name])
            + " |"
            for name in ability_names
        ],
    ]

    correlation = scores.corr()
    correlation.to_csv(output_directory / "score_correlations.csv", encoding="utf-8")
    class_means = scores.assign(**{CLASS_LABEL_COLUMN: data[CLASS_LABEL_COLUMN]}).groupby(
        CLASS_LABEL_COLUMN, observed=False
    )[feature_columns].mean().reindex(class_names)
    class_means.to_csv(output_directory / "score_means_by_class.csv", encoding="utf-8")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(8, 4, figsize=(18, 24), constrained_layout=True)
    for axis, column in zip(axes.flat, feature_columns):
        sns.histplot(scores[column], bins=np.arange(lower, upper + 0.5, 0.5), ax=axis)
        axis.set_title(column, fontsize=9)
        axis.set_xlim(lower, upper)
        axis.set_xlabel("Điểm cả năm")
    fig.suptitle("Phân phối điểm cả năm của pilot (32 đặc trưng)", fontsize=16)
    fig.savefig(output_directory / "score_distributions.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(16, 13), constrained_layout=True)
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=True,
        yticklabels=True,
        ax=axis,
    )
    axis.set_title("Tương quan giữa 32 đặc trưng điểm cả năm")
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    axis.tick_params(axis="y", labelrotation=0, labelsize=7)
    fig.savefig(output_directory / "score_correlations.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    sns.barplot(data=label_summary, x=CLASS_LABEL_COLUMN, y="so_luong", color="#4472C4", ax=axis)
    axis.set_title("Phân bố nhãn lớp của pilot")
    axis.set_xlabel("Nhãn lớp")
    axis.set_ylabel("Số bản ghi")
    fig.savefig(output_directory / "label_distribution.png", dpi=160)
    plt.close(fig)

    clipping_percent = float(
        ((scores == lower) | (scores == upper)).to_numpy().mean() * 100
    )
    min_class_count = int(counts.min())
    max_class_count = int(counts.max())
    imbalance_ratio = (
        float(max_class_count / min_class_count) if min_class_count > 0 else None
    )
    largest_class = class_names[int(counts.to_numpy().argmax())]
    smallest_class = class_names[int(counts.to_numpy().argmin())]
    count_difference = max_class_count - min_class_count
    percentage_difference = count_difference / len(data) * 100
    class_percentage_std = float(
        np.std(counts.to_numpy(dtype=float) / len(data) * 100, ddof=0)
    )
    class_probabilities = counts.to_numpy(dtype=float) / len(data)
    entropy = float(-np.sum(class_probabilities * np.log(class_probabilities)))
    normalized_entropy = float(entropy / np.log(len(class_names)))
    imbalance_level = (
        "thấp"
        if imbalance_ratio is not None and imbalance_ratio <= 1.5
        else "vừa"
        if imbalance_ratio is not None and imbalance_ratio <= 2.0
        else "cao"
    )
    missing_features = int(scores.isna().sum().sum())
    report_lines = [
        f"# Phân tích dữ liệu thăm dò — pilot seed {seed}",
        "",
        "> Chỉ là pilot thăm dò dữ liệu tổng hợp. Các kết quả này không phải quan sát về học sinh thật,",
        "> không phải bằng chứng hiệu năng mô hình; thiết kế C1–C6 đã được xác nhận.",
        "",
        "## Phân bố nhãn C1–C6",
        "",
        "| Nhãn | Số mẫu | Tỷ lệ (%) |",
        "|---|---:|---:|",
        *[
            f"| {name} | {int(counts[name]):,} | {float(counts[name] / len(data) * 100):.2f} |"
            for name in class_names
        ],
        "",
        "## Xác suất Softmax trung bình",
        "",
        "| Nhãn | Xác suất Softmax trung bình |",
        "|---|---:|",
        *[
            f"| {name} | {mean_sampling_probabilities[name]:.4f} |"
            for name in class_names
        ],
        "",
        "Giá trị là trung bình xác suất Softmax trước bước lấy mẫu nhãn.",
        "",
        "## Bằng chứng kiểm tra ma trận tương quan R_Z",
        "",
        f"- Đối xứng: **{correlation_validation['symmetric']}**.",
        f"- Đường chéo bằng 1: **{correlation_validation['unit_diagonal']}**.",
        "- Tất cả trị riêng dương: "
        f"**{correlation_validation['all_eigenvalues_positive']}** "
        f"({', '.join(f'{value:.6f}' for value in correlation_validation['eigenvalues'])}).",
        f"- Cholesky decomposition: **{correlation_validation['cholesky_decomposition']}**.",
        "",
        "Ma trận R_Z chính xác được sử dụng:",
        "",
        *correlation_matrix_markdown,
        "",
        "## Phân phối các chiều S và D",
        "",
        "| Nhóm | Chiều | Mean | SD | Min | Q25 | Median | Q75 | Max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *[
            "| "
            + " | ".join(
                [
                    group,
                    dimension,
                    f"{values['mean']:.4f}",
                    f"{values['standard_deviation']:.4f}",
                    f"{values['minimum']:.4f}",
                    f"{values['q25']:.4f}",
                    f"{values['median']:.4f}",
                    f"{values['q75']:.4f}",
                    f"{values['maximum']:.4f}",
                ]
            )
            + " |"
            for group, dimensions in interest_career_summary.items()
            for dimension, values in dimensions.items()
        ],
        "",
        "S là các chiều sở thích môn học; D là các chiều định hướng nghề nghiệp "
        "RIASEC. Các giá trị được biến đổi vào [0;1] bằng CDF chuẩn chuẩn hóa.",
        "",
        f"- Chênh lệch {largest_class}–{smallest_class}: {count_difference:,} mẫu "
        f"({percentage_difference:.2f} điểm phần trăm).",
        f"- Tỷ số số mẫu {largest_class}/{smallest_class}: "
        + (f"{imbalance_ratio:.2f}" if imbalance_ratio is not None else "không xác định") + ".",
        f"- Độ lệch chuẩn tỷ lệ lớp: {class_percentage_std:.2f} điểm phần trăm.",
        f"- Đánh giá mất cân bằng: **{imbalance_level}** "
        "(quy ước: tỷ số lớp lớn nhất/nhỏ nhất ≤ 1,5 là thấp; ≤ 2,0 là vừa).",
        f"- Entropy phân phối nhãn: **{entropy:.4f} nat**.",
        f"- Entropy chuẩn hóa: **{normalized_entropy:.4f}** "
        f"(H / ln({len(class_names)})); 1,0 là phân phối đều hoàn toàn.",
        "- Công thức: H = −Σ pᵢ ln(pᵢ), với pᵢ là tỷ lệ của lớp i.",
        "",
        "## Kiểm tra dữ liệu",
        "",
        f"- Số bản ghi: {len(data):,}",
        f"- Đặc trưng điểm cả năm: {len(feature_columns)}",
        f"- Khoảng điểm quan sát: {score_values.min():.1f}–{score_values.max():.1f}",
        f"- Ô điểm bị thiếu: {missing_features}",
        f"- Ô điểm bằng 0 hoặc 10: {clipping_percent:.2f}%",
        f"- Nhãn không có bản ghi: {int((counts == 0).sum())}",
        f"- Tỷ lệ lớp lớn nhất/nhỏ nhất: "
        + (f"{imbalance_ratio:.2f}" if imbalance_ratio is not None else "không xác định"),
        "",
        "## Tệp",
        "",
        "- `score_summary.csv`: thống kê mô tả và giá trị biên theo đặc trưng.",
        "- `label_distribution.csv`: số lượng và tỷ lệ phần trăm theo lớp.",
        "- `mean_softmax_probabilities.csv`: xác suất Softmax trung bình theo lớp.",
        "- `correlation_matrix_R_Z.csv`: ma trận R_Z chính xác được sử dụng.",
        "- `interest_career_dimension_summary.csv`: bảng mô tả các chiều S và D.",
        "- `score_distribution_by_subject.csv`, `score_distribution_by_grade.csv`: mean và "
        "độ lệch chuẩn theo môn và khối.",
        "- `subject_correlations.csv`: tương quan giữa điểm trung bình các môn.",
        "- `within_subject_year_correlations.csv`: tương quan cùng môn qua các khối 6–9.",
        "- `score_correlations.csv`: tương quan từng cặp của các đặc trưng điểm.",
        "- `score_means_by_class.csv`: trung bình đặc trưng theo nhãn được sinh.",
        "- `score_distributions.png`, `score_correlations.png`, `label_distribution.png`: biểu đồ.",
        "",
        "## Kiểm tra phân phối điểm học tập",
        "",
        f"- Clipping tại {lower:g}: {clipping_lower_percent:.2f}%; "
        f"clipping tại {upper:g}: {clipping_upper_percent:.2f}%.",
        f"- Tỷ lệ giá trị trùng sau làm tròn {decimal_places} chữ số: "
        f"{duplicate_after_rounding_percent:.2f}% "
        f"(1 − số giá trị duy nhất / tổng số ô điểm; {unique_rounded_values}/"
        f"{total_score_cells}).",
        "- Tương quan liên môn được tính trên điểm trung bình bốn khối của từng môn; "
        "tương quan cùng môn được tính cho từng cặp khối trong 6–7, 6–8, 6–9, 7–8, 7–9, 8–9.",
        "- μ_s = 7,0 cho cả tám môn là giả định đơn giản hóa của benchmark, không phải "
        "khẳng định điểm trung bình các môn ngoài thực tế bằng nhau.",
        "",
        "## Giới hạn diễn giải",
        "",
        "Dùng các đầu ra này để kiểm tra bộ sinh hoạt động đúng đặc tả. Mọi thay đổi giả định "
        "của bộ sinh phải được ghi lại trước khi sinh các bộ dữ liệu lớn hơn. Pilot 1.000 bản ghi "
        "quá nhỏ để khẳng định tính hợp lệ thực tế "
        "hoặc sự cân bằng lớp cuối cùng.",
        "",
    ]
    report_path = output_directory / "eda_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "seed": seed,
        "records": len(data),
        "features": len(feature_columns),
        "missing_score_cells": missing_features,
        "boundary_score_percent": clipping_percent,
        "empty_classes": int((counts == 0).sum()),
        "class_imbalance_ratio": imbalance_ratio,
        "output_directory": str(output_directory),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Thăm dò một bộ dữ liệu pilot tổng hợp đã sinh.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    data_path = args.data or (
        PROJECT_ROOT / "data" / "pilot" / f"seed_{args.seed}" / f"pilot_seed_{args.seed}.csv"
    )
    output_directory = args.output or (
        PROJECT_ROOT / "results" / "pilot" / f"seed_{args.seed}"
    )
    manifest_path = args.manifest or (
        PROJECT_ROOT / "logs" / "pilot_runs" / f"seed_{args.seed}.json"
    )
    try:
        config = load_config(args.config)
        if args.seed not in config["dataset"]["seeds"]:
            raise ValueError(f"Seed {args.seed} không có trong cấu hình bộ sinh")
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = yaml.safe_load(file)
        if not isinstance(manifest, dict):
            raise ValueError(f"Manifest phải là một ánh xạ YAML/JSON: {manifest_path}")
        mean_sampling_probabilities = manifest.get("mean_sampling_probabilities")
        correlation_validation = manifest.get("correlation_matrix_validation")
        interest_career_summary = manifest.get("interest_career_summary")
        class_names = list(config["labels"]["classes"])
        if (
            not isinstance(mean_sampling_probabilities, dict)
            or set(mean_sampling_probabilities) != set(class_names)
            or any(
                not isinstance(mean_sampling_probabilities[name], (int, float))
                or not np.isfinite(mean_sampling_probabilities[name])
                for name in class_names
            )
        ):
            raise ValueError(
                f"{manifest_path}: thiếu hoặc sai mean_sampling_probabilities theo C1-C6"
            )
        required_correlation_fields = {
            "matrix",
            "symmetric",
            "unit_diagonal",
            "eigenvalues",
            "all_eigenvalues_positive",
            "cholesky_decomposition",
        }
        if (
            not isinstance(correlation_validation, dict)
            or set(correlation_validation) != required_correlation_fields
            or correlation_validation["cholesky_decomposition"] != "success"
            or not correlation_validation["symmetric"]
            or not correlation_validation["unit_diagonal"]
            or not correlation_validation["all_eigenvalues_positive"]
        ):
            raise ValueError(f"{manifest_path}: bằng chứng kiểm tra R_Z không hợp lệ")
        if (
            not isinstance(interest_career_summary, dict)
            or set(interest_career_summary) != {"S_subject_interest", "D_career_interest"}
        ):
            raise ValueError(f"{manifest_path}: thiếu bảng mô tả các chiều S và D")
        report = run_eda(
            data_path,
            config,
            args.seed,
            output_directory,
            {name: float(mean_sampling_probabilities[name]) for name in class_names},
            correlation_validation,
            interest_career_summary,
        )
        print(pd.Series(report).to_json(force_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        print(f"Phân tích thăm dò thất bại: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
