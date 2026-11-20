"""Kiểm tra CSV dữ liệu đã sinh và manifest tái lập theo từng seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from schema import CLASS_LABEL_COLUMN, STUDENT_ID_COLUMN, score_column


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "generator_config"


def validate_dataset(
    data_path: Path,
    config: dict[str, Any],
    seed: int,
    expected_records: int | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Ném ValueError nếu schema, giá trị hoặc manifest tái lập vi phạm."""
    dataset_config = config["dataset"]
    scores = config["scores"]
    class_names = list(config["labels"]["classes"])
    expected_count = dataset_config["records_per_seed"] if expected_records is None else expected_records
    feature_columns = [
        score_column(subject, grade)
        for subject in scores["subjects"]
        for grade in scores["grades"]
    ]
    expected_columns = [STUDENT_ID_COLUMN, *feature_columns, CLASS_LABEL_COLUMN]
    data = pd.read_csv(data_path)
    if list(data.columns) != expected_columns:
        raise ValueError(
            f"{data_path}: cột phải đúng thứ tự {expected_columns}; "
            f"thực tế là {list(data.columns)}"
        )
    if len(data) != expected_count:
        raise ValueError(f"{data_path}: cần {expected_count} dòng nhưng có {len(data)}")
    if data[STUDENT_ID_COLUMN].isna().any() or data[STUDENT_ID_COLUMN].duplicated().any():
        raise ValueError(f"{data_path}: ma_hoc_sinh phải có giá trị và không trùng")
    if data[CLASS_LABEL_COLUMN].isna().any() or not set(data[CLASS_LABEL_COLUMN]) <= set(class_names):
        raise ValueError(f"{data_path}: nhan_lop có giá trị thiếu hoặc không xác định")

    scores_array = data[feature_columns].to_numpy(dtype=float)
    lower, upper = scores["score_bounds"]
    if not np.all(np.isfinite(scores_array)):
        raise ValueError(f"{data_path}: các cột điểm có giá trị không hữu hạn")
    if np.any(scores_array < lower) or np.any(scores_array > upper):
        raise ValueError(f"{data_path}: điểm nằm ngoài [{lower}, {upper}]")
    decimal_scale = 10 ** scores["decimal_places"]
    if not np.allclose(scores_array * decimal_scale, np.round(scores_array * decimal_scale)):
        raise ValueError(f"{data_path}: điểm không đúng độ chính xác đã cấu hình")

    if manifest_path is None:
        manifest_path = PROJECT_ROOT / "logs" / "runs" / f"seed_{seed}.json"
    if not manifest_path.is_file():
        raise ValueError(f"Thiếu manifest chạy cho seed {seed}: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("seed") != seed or manifest.get("records") != len(data):
        raise ValueError(f"{manifest_path}: seed hoặc số bản ghi không khớp CSV")
    expected_run_type = "exploratory_pilot" if "pilot" in data_path.parts else "production"
    if manifest.get("run_type") != expected_run_type:
        raise ValueError(f"{manifest_path}: loại chạy không khớp vị trí dữ liệu")
    if manifest.get("features") != len(feature_columns):
        raise ValueError(f"{manifest_path}: số đặc trưng không khớp CSV")
    config_json = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    expected_config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    if manifest.get("config_sha256") != expected_config_hash:
        raise ValueError(f"{manifest_path}: mã băm cấu hình không khớp cấu hình hiện tại")
    if manifest.get("dataset_sha256") != _sha256_file(data_path):
        raise ValueError(f"{manifest_path}: mã kiểm tra dữ liệu không khớp CSV")
    if manifest.get("class_counts") != {
        name: int(data[CLASS_LABEL_COLUMN].eq(name).sum()) for name in class_names
    }:
        raise ValueError(f"{manifest_path}: số lượng lớp không khớp CSV")

    counts = data[CLASS_LABEL_COLUMN].value_counts().reindex(class_names, fill_value=0)
    return {
        "seed": seed,
        "records": len(data),
        "features": len(feature_columns),
        "class_counts": {name: int(counts[name]) for name in class_names},
        "minimum_score": float(scores_array.min()),
        "maximum_score": float(scores_array.max()),
        "csv": str(data_path),
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Kiểm tra dữ liệu đã sinh theo các seed đã cấu hình.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, action="append", help="Seed cần kiểm tra; lặp lại để chọn nhiều seed.")
    parser.add_argument("--records", type=int, help="Số dòng mong đợi (cho dữ liệu kiểm thử nhanh).")
    parser.add_argument("--pilot", action="store_true", help="Kiểm tra pilot thăm dò riêng biệt.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    args = parser.parse_args()
    try:
        with args.config.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        if not isinstance(config, dict):
            raise ValueError("Cấu hình bộ sinh phải là một ánh xạ YAML")
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
        if args.records is not None and args.records <= 0:
            raise ValueError("--records phải là số nguyên dương")
        if args.pilot and args.records is None:
            args.records = 1000
        if args.pilot and args.records > 1000:
            raise ValueError("Kiểm tra pilot hỗ trợ tối đa 1.000 bản ghi")
        data_root = args.data_root or (
            PROJECT_ROOT / "data" / "pilot" if args.pilot else PROJECT_ROOT / "data"
        )
        log_root = args.log_root or (
            PROJECT_ROOT / "logs" / "pilot_runs" if args.pilot else PROJECT_ROOT / "logs" / "runs"
        )

        for seed in seeds:
            prefix = "pilot_" if args.pilot else "dataset_"
            data_path = data_root / f"seed_{seed}" / f"{prefix}seed_{seed}.csv"
            manifest_path = log_root / f"seed_{seed}.json"
            report = validate_dataset(
                data_path,
                config,
                seed,
                args.records,
                manifest_path=manifest_path,
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    except (OSError, ValueError, KeyError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(f"Kiểm tra thất bại: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
