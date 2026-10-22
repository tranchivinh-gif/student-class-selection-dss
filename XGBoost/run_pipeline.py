"""Run EDA, cleaning, training, evaluation and presentation in order."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analyze import analyze
from eda_report import create_eda_report
from evaluate import main as evaluate_main
from preprocessing import DEFAULT_SOURCE, PROJECT_DIR, prepare_data
from train import train_model


def resolve_source(source: Path) -> Path:
    """Accept paths from the current directory or the project's data folder."""
    source = Path(source)
    if source.is_file():
        return source.resolve()
    if not source.is_absolute() and source.parts and source.parts[0].casefold() == "data":
        candidate = PROJECT_DIR / source
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu nguồn: {source}. "
        f"Thử đường dẫn {PROJECT_DIR / 'data' / source.name}"
    )


def run_pipeline(source: Path = DEFAULT_SOURCE) -> None:
    source = resolve_source(source)
    print("[1/5] EDA", flush=True)
    create_eda_report(source)
    print("[2/5] Xử lý dữ liệu", flush=True)
    prepare_data(source)
    print("[3/5] Huấn luyện", flush=True)
    train_model()
    print("[4/5] Đánh giá", flush=True)
    evaluate_main()
    print("[5/5] Trình bày kết quả", flush=True)
    analyze()
    print("Hoàn tất. Xem các tệp trong XGBoost/reports/.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    run_pipeline(parser.parse_args().input)
