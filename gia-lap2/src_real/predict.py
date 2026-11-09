from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import MODEL_PATH
from .data import load_features
from .model import RealCombinationModel

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=Path("predictions_real.csv"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    model = RealCombinationModel.load(args.model)
    data = load_features(args.input, model._features())
    result = model.recommend(data, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Đã lưu {len(result)} dòng tại {args.output}")


if __name__ == "__main__":
    main()

