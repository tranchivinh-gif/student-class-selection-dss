"""Sinh các bộ dữ liệu học sinh tổng hợp độc lập theo thiết kế YAML đã duyệt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.special import ndtr

from schema import CLASS_LABEL_COLUMN, STUDENT_ID_COLUMN, score_column

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "generator_config"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Cấu hình bộ sinh phải là một ánh xạ YAML: {path}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    dataset = config.get("dataset", {})
    latent = config.get("latent_abilities", {})
    scores = config.get("scores", {})
    interests = config.get("interests_and_careers", {})
    labels = config.get("labels", {})

    seeds = dataset.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(
        not isinstance(seed, int) or seed < 0 for seed in seeds
    ):
        raise ValueError("dataset.seeds phải là danh sách số nguyên không âm và không rỗng")
    if dataset.get("output_format") != "csv":
        raise ValueError("dataset.output_format phải là csv")
    if len(set(seeds)) != len(seeds):
        raise ValueError("dataset.seeds không được trùng lặp")
    if not isinstance(dataset.get("records_per_seed"), int) or dataset["records_per_seed"] <= 0:
        raise ValueError("dataset.records_per_seed phải là số nguyên dương")

    latent_mean = np.asarray(latent.get("mean"), dtype=float)
    correlation = np.asarray(latent.get("correlation_matrix"), dtype=float)
    ability_count = latent.get("count")
    if (
        not isinstance(ability_count, int)
        or ability_count <= 0
        or latent_mean.shape != (ability_count,)
        or not np.all(np.isfinite(latent_mean))
        or correlation.shape != (ability_count, ability_count)
        or not np.all(np.isfinite(correlation))
        or not np.allclose(correlation, correlation.T)
        or not np.allclose(np.diag(correlation), 1.0)
    ):
        raise ValueError("Trung bình năng lực ẩn và ma trận tương quan có kích thước hoặc giá trị không hợp lệ")
    eigenvalues = np.linalg.eigvalsh(correlation)
    if not np.all(eigenvalues > 0):
        raise ValueError(
            "latent_abilities.correlation_matrix phải có tất cả trị riêng dương; "
            f"trị riêng là {eigenvalues.tolist()}"
        )
    try:
        np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as error:
        raise ValueError("latent_abilities.correlation_matrix phải xác định dương") from error

    score_subjects = scores.get("subjects", [])
    grades = scores.get("grades", [])
    score_weights = scores.get("ability_weights", {})
    if not score_subjects or not grades or set(score_weights) != set(score_subjects):
        raise ValueError("Phải cấu hình môn học, lớp và các hàng trọng số năng lực")
    for subject, row in score_weights.items():
        weights = np.asarray(row, dtype=float)
        if (
            weights.shape != (ability_count,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0)
        ):
            raise ValueError(f"scores.ability_weights.{subject} phải không âm và có tổng bằng 1")
    if scores.get("score_bounds", [0, 10])[0] >= scores.get("score_bounds", [0, 10])[1]:
        raise ValueError("scores.score_bounds phải có cận dưới nhỏ hơn cận trên")

    electives = interests.get("elective_subjects", [])
    dimensions = interests.get("riasec_dimensions", [])
    career_matrix = interests.get("subject_career_matrix", {})
    if not electives or not dimensions or set(career_matrix) != set(electives):
        raise ValueError("Môn quan tâm, chiều RIASEC và các hàng ma trận nghề nghiệp phải khớp nhau")
    for subject, row in career_matrix.items():
        values = np.asarray(row, dtype=float)
        if (
            values.shape != (len(dimensions),)
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(values > 1)
            or values.sum() <= 0
        ):
            raise ValueError(f"interests_and_careers.subject_career_matrix.{subject} không hợp lệ")

    classes = labels.get("classes", {})
    expected_classes = [f"C{index}" for index in range(1, 7)]
    if list(classes) != expected_classes:
        raise ValueError("labels.classes phải xác định đúng thứ tự C1 đến C6")
    label_weights = labels.get("subject_ability_weights", {})
    utility_weights = labels.get("utility_weights", {})
    temperature = labels.get("softmax_temperature")
    if not classes or set(label_weights) != set(electives):
        raise ValueError("Nhãn lớp và các hàng trọng số năng lực theo môn chưa đầy đủ")
    for class_name, subjects in classes.items():
        if len(subjects) != 4 or len(set(subjects)) != 4 or not set(subjects) <= set(electives):
            raise ValueError(f"labels.classes.{class_name} phải có đúng bốn môn khác nhau đã cấu hình")
    for subject, row in label_weights.items():
        weights = np.asarray(row, dtype=float)
        if (
            weights.shape != (ability_count,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0)
        ):
            raise ValueError(f"labels.subject_ability_weights.{subject} phải không âm và có tổng bằng 1")
    if (
        set(utility_weights) != {"ability_fit", "subject_interest", "career_fit"}
        or any(not np.isfinite(value) or value < 0 for value in utility_weights.values())
        or not np.isclose(sum(utility_weights.values()), 1.0)
    ):
        raise ValueError("labels.utility_weights phải không âm và có tổng bằng 1")
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError("labels.softmax_temperature phải là số dương")
    if interests.get("shared_variance_parameter", -1) < 0 or interests.get(
        "shared_variance_parameter", 2
    ) > 1:
        raise ValueError("shared_variance_parameter phải nằm trong [0, 1]")


def generate_dataset(
    config: dict[str, Any], seed: int, records: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sinh bản ghi cho một seed và trả về CSV cùng chẩn đoán theo lớp."""
    validate_config(config)
    if seed not in config["dataset"]["seeds"]:
        raise ValueError(f"Seed {seed} không có trong dataset.seeds")
    row_count = config["dataset"]["records_per_seed"] if records is None else records
    if not isinstance(row_count, int) or row_count <= 0:
        raise ValueError("records phải là số nguyên dương")

    rng = np.random.default_rng(seed)
    latent_config = config["latent_abilities"]
    correlation = np.asarray(latent_config["correlation_matrix"], dtype=float)
    abilities = rng.multivariate_normal(
        np.asarray(latent_config["mean"], dtype=float),
        correlation,
        size=row_count,
        method="cholesky",
        check_valid="raise",
    )

    score_config = config["scores"]
    grades = score_config["grades"]
    score_bounds = score_config["score_bounds"]
    score_weights = np.asarray(
        [score_config["ability_weights"][subject] for subject in score_config["subjects"]],
        dtype=float,
    )
    score_features: dict[str, np.ndarray] = {}
    ability_component = score_config["ability_scale"] * (abilities @ score_weights.T)
    for subject_index, subject in enumerate(score_config["subjects"]):
        subject_slope = rng.normal(
            0.0, score_config["student_subject_slope_sd"], size=row_count
        )
        yearly_noise = rng.normal(
            0.0,
            score_config["observation_noise_sd"],
            size=(row_count, len(grades)),
        )
        for grade_index, grade in enumerate(grades):
            raw_scores = (
                score_config["subject_baseline"]
                + ability_component[:, subject_index]
                + (score_config["mean_grade_trend"] + subject_slope) * (grade - 6)
                + yearly_noise[:, grade_index]
            )
            score_features[score_column(subject, grade)] = np.round(
                np.clip(raw_scores, score_bounds[0], score_bounds[1]),
                decimals=score_config["decimal_places"],
            )

    interest_config = config["interests_and_careers"]
    elective_subjects = interest_config["elective_subjects"]
    riasec_dimensions = interest_config["riasec_dimensions"]
    rho = interest_config["shared_variance_parameter"]
    career_matrix = np.asarray(
        [interest_config["subject_career_matrix"][subject] for subject in elective_subjects],
        dtype=float,
    )
    normalized_career_weights = career_matrix / np.linalg.norm(career_matrix, axis=1)[:, None]
    shared_interest = rng.standard_normal((row_count, len(riasec_dimensions)))
    private_subject_interest = rng.standard_normal((row_count, len(elective_subjects)))
    private_career_interest = rng.standard_normal((row_count, len(riasec_dimensions)))
    subject_latent_interest = (
        np.sqrt(rho) * (shared_interest @ normalized_career_weights.T)
        + np.sqrt(1.0 - rho) * private_subject_interest
    )
    career_latent_interest = (
        np.sqrt(rho) * shared_interest + np.sqrt(1.0 - rho) * private_career_interest
    )
    subject_interest = ndtr(subject_latent_interest)
    career_interest = ndtr(career_latent_interest)

    label_config = config["labels"]
    label_subjects = list(label_config["subject_ability_weights"])
    label_weights = np.asarray(
        [label_config["subject_ability_weights"][subject] for subject in label_subjects],
        dtype=float,
    )
    projected_variances = np.einsum("ij,jk,ik->i", label_weights, correlation, label_weights)
    if np.any(projected_variances <= 0):
        raise ValueError("Mỗi trọng số năng lực theo môn của nhãn phải có phương sai dương")
    normalized_ability = (abilities @ label_weights.T) / np.sqrt(projected_variances)[None, :]
    subject_ability_fit = ndtr(normalized_ability)

    career_denominators = career_matrix.sum(axis=1)
    subject_career_fit = (career_interest @ career_matrix.T) / career_denominators[None, :]
    interest_by_subject = {
        subject: subject_interest[:, index]
        for index, subject in enumerate(elective_subjects)
    }
    ability_by_subject = {
        subject: subject_ability_fit[:, index]
        for index, subject in enumerate(label_subjects)
    }
    career_by_subject = {
        subject: subject_career_fit[:, index]
        for index, subject in enumerate(elective_subjects)
    }
    class_names = list(label_config["classes"])
    utilities = np.empty((row_count, len(class_names)), dtype=float)
    utility_weights = label_config["utility_weights"]
    for class_index, class_name in enumerate(class_names):
        subjects = label_config["classes"][class_name]
        ability_fit = np.mean([ability_by_subject[subject] for subject in subjects], axis=0)
        interest_fit = np.mean([interest_by_subject[subject] for subject in subjects], axis=0)
        career_fit = np.mean([career_by_subject[subject] for subject in subjects], axis=0)
        utilities[:, class_index] = (
            utility_weights["ability_fit"] * ability_fit
            + utility_weights["subject_interest"] * interest_fit
            + utility_weights["career_fit"] * career_fit
        )

    scaled_utilities = utilities / label_config["softmax_temperature"]
    scaled_utilities -= scaled_utilities.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled_utilities)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    draws = rng.random(row_count)
    selected_indices = np.minimum(
        (draws[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1),
        len(class_names) - 1,
    )
    selected_labels = np.asarray(class_names, dtype=object)[selected_indices]

    data = pd.DataFrame(
        {
            STUDENT_ID_COLUMN: [f"S{seed}_{index:05d}" for index in range(1, row_count + 1)],
            **score_features,
            CLASS_LABEL_COLUMN: selected_labels,
        }
    )
    counts = data[CLASS_LABEL_COLUMN].value_counts().reindex(class_names, fill_value=0)
    correlation_eigenvalues = np.linalg.eigvalsh(correlation)
    def summarize_dimensions(values: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
        return {
            name: {
                "mean": float(values[:, index].mean()),
                "standard_deviation": float(values[:, index].std(ddof=1)),
                "minimum": float(values[:, index].min()),
                "q25": float(np.quantile(values[:, index], 0.25)),
                "median": float(np.median(values[:, index])),
                "q75": float(np.quantile(values[:, index], 0.75)),
                "maximum": float(values[:, index].max()),
            }
            for index, name in enumerate(names)
        }
    diagnostics = {
        "class_counts": {name: int(counts[name]) for name in class_names},
        "class_probabilities_mean": {
            name: float(probabilities[:, index].mean())
            for index, name in enumerate(class_names)
        },
        "correlation_matrix_validation": {
            "matrix": correlation.tolist(),
            "symmetric": bool(np.allclose(correlation, correlation.T)),
            "unit_diagonal": bool(np.allclose(np.diag(correlation), 1.0)),
            "eigenvalues": correlation_eigenvalues.tolist(),
            "all_eigenvalues_positive": bool(np.all(correlation_eigenvalues > 0)),
            "cholesky_decomposition": "success",
        },
        "interest_career_summary": {
            "S_subject_interest": summarize_dimensions(subject_interest, elective_subjects),
            "D_career_interest": summarize_dimensions(career_interest, riasec_dimensions),
        },
        "feature_count": len(score_features),
    }
    return data, diagnostics


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary_path, path)


def generate_seed_output(
    config: dict[str, Any],
    seed: int,
    output_root: Path,
    records: int | None = None,
    overwrite: bool = False,
    pilot: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Sinh và ghi nguyên tử một CSV cùng manifest tái lập."""
    if pilot:
        seed_directory = output_root / "data" / "pilot" / f"seed_{seed}"
        log_directory = output_root / "logs" / "pilot_runs"
        file_prefix = "pilot_"
    else:
        seed_directory = output_root / "data" / f"seed_{seed}"
        log_directory = output_root / "logs" / "runs"
        file_prefix = "dataset_"
    seed_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    data_path = seed_directory / f"{file_prefix}seed_{seed}.csv"
    manifest_path = log_directory / f"seed_{seed}.json"
    if not overwrite and (data_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"Đầu ra cho seed {seed} đã tồn tại. Dùng --force để thay thế: "
            f"{data_path}, {manifest_path}"
        )

    config_json = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    started_at = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    data, diagnostics = generate_dataset(config, seed, records)
    temporary_data_path = data_path.with_suffix(data_path.suffix + ".tmp")
    try:
        data.to_csv(temporary_data_path, index=False, encoding="utf-8")
        os.replace(temporary_data_path, data_path)
    finally:
        if temporary_data_path.exists():
            temporary_data_path.unlink()
    elapsed_seconds = time.perf_counter() - start_clock
    completed_at = datetime.now(timezone.utc)
    manifest = {
        "run_type": "exploratory_pilot" if pilot else "production",
        "approval_status": "pilot_exploratory" if pilot else "approved",
        "seed": seed,
        "records": len(data),
        "features": diagnostics["feature_count"],
        "target": CLASS_LABEL_COLUMN,
        "class_counts": diagnostics["class_counts"],
        "mean_sampling_probabilities": diagnostics["class_probabilities_mean"],
        "correlation_matrix_validation": diagnostics["correlation_matrix_validation"],
        "interest_career_summary": diagnostics["interest_career_summary"],
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "config": config,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "pyyaml": yaml.__version__,
        },
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "dataset_path": str(data_path.relative_to(output_root)),
        "dataset_sha256": _sha256_file(data_path),
    }
    _write_json_atomic(manifest_path, manifest)
    return data_path, manifest_path, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sinh các bộ dữ liệu học sinh tổng hợp riêng cho từng seed đã cấu hình."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, action="append", help="Seed cần chạy; lặp lại để chọn nhiều seed.")
    parser.add_argument("--records", type=int, help="Ghi đè số bản ghi (dùng cho kiểm thử nhanh).")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--force", action="store_true", help="Thay thế CSV và manifest hiện có.")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Sinh pilot thăm dò (mặc định 1.000 bản ghi) trong data/pilot; không coi là đã duyệt.",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    try:
        config_path = args.config.resolve()
        config = load_config(config_path)
        seeds = (
            [config["dataset"]["seeds"][0]]
            if args.pilot and args.seed is None
            else config["dataset"]["seeds"] if args.seed is None else args.seed
        )
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Các seed yêu cầu không được rỗng hoặc trùng lặp")
        unknown_seeds = sorted(set(seeds) - set(config["dataset"]["seeds"]))
        if unknown_seeds:
            raise ValueError(f"Các seed yêu cầu chưa được cấu hình: {unknown_seeds}")
        if args.pilot:
            pilot_records = 1000 if args.records is None else args.records
            if pilot_records <= 0 or pilot_records > 1000:
                raise ValueError("Pilot hỗ trợ từ 1 đến 1.000 bản ghi")
        elif args.records is not None and args.records <= 0:
            raise ValueError("--records phải là số nguyên dương")
        output_root = args.output_root.resolve()

        for seed in seeds:
            data_path, manifest_path, manifest = generate_seed_output(
                config,
                seed,
                output_root,
                records=pilot_records if args.pilot else args.records,
                overwrite=args.force,
                pilot=args.pilot,
            )
            print(
                f"seed={seed} bản ghi={manifest['records']} "
                f"lớp={manifest['class_counts']} "
                f"csv={data_path} nhật ký={manifest_path}"
            )
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Sinh dữ liệu thất bại: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
