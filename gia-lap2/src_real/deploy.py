"""Promote a reviewed training run to the stable application model path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .model import RealCombinationModel
from .train import promote_model

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đưa model của một lần huấn luyện đã duyệt vào web/predict."
    )
    parser.add_argument("--run", type=Path, required=True, help="Thư mục của lần huấn luyện")
    args = parser.parse_args()
    run_dir = args.run
    metadata_path = run_dir / "metadata.json"
    model_paths = [run_dir / f"{name}.joblib" for name in ("baseline", "xgboost", "catboost")]
    if not metadata_path.exists() or not all(path.exists() for path in model_paths):
        parser.error("Thư mục run phải có metadata.json và đủ baseline/xgboost/catboost.joblib.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for model_path in model_paths:
        RealCombinationModel.load(model_path)
    promote_model(
        run_dir,
        Path(metadata.get("data", "không_xác_định")),
        metadata.get("independent_test_metrics", {}),
    )
    metadata["promoted_to_application"] = True
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã đưa model vào ứng dụng từ: {run_dir}")


if __name__ == "__main__":
    main()
