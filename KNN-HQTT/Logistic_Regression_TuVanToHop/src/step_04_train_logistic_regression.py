"""Huấn luyện mô hình Logistic Regression đa nhãn bằng Pipeline."""

from __future__ import annotations

import json
import sys

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import MODELS_DIR, MODEL_PATH, STEP_03_DIR, STEP_04_DIR


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    required_files = [
        STEP_03_DIR / "X_train.csv",
        STEP_03_DIR / "y_train.csv",
        STEP_03_DIR / "dataset_metadata.json",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Thiếu tập dữ liệu bước 3. Hãy chạy "
            "python src/step_03_prepare_dataset.py trước.\n"
            + "\n".join(missing)
        )

    X_train = pd.read_csv(required_files[0], encoding="utf-8-sig")
    y_train = pd.read_csv(required_files[1], encoding="utf-8-sig")
    with required_files[2].open(encoding="utf-8") as file:
        metadata = json.load(file)

    features = metadata["feature_columns"]
    labels = metadata["label_columns"]
    if X_train.columns.tolist() != features or y_train.columns.tolist() != labels:
        raise ValueError(
            "Các cột trong tập huấn luyện không khớp metadata bước 3. "
            "Hãy chạy lại bước 3."
        )
    if X_train.empty or y_train.empty or len(X_train) != len(y_train):
        raise ValueError("Tập huấn luyện rỗng hoặc X_train và y_train lệch số dòng.")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                MultiOutputClassifier(
                    LogisticRegression(
                        max_iter=2000,
                        random_state=metadata["random_state"],
                        class_weight=None,
                    )
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    STEP_04_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    metadata["model_type"] = "MultiOutputClassifier(LogisticRegression)"
    metadata["preprocessing"] = [
        "SimpleImputer(strategy='median')",
        "StandardScaler()",
    ]
    metadata["class_weight"] = None
    metadata["model_path"] = str(MODEL_PATH.relative_to(MODEL_PATH.parents[1]))
    metadata["training_samples"] = len(X_train)
    metadata["feature_count"] = len(features)
    metadata["label_count"] = len(labels)
    with (MODELS_DIR / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    model_report = {
        "loai_mo_hinh": metadata["model_type"],
        "so_mau_huan_luyen": len(X_train),
        "so_dac_trung": len(features),
        "so_nhan": len(labels),
        "ten_dac_trung": features,
        "ten_nhan": labels,
        "class_weight": None,
        "ly_do_class_weight": (
            "Không bật balanced vì tỷ lệ yes/no mỗi nhãn ở dữ liệu đã kiểm tra "
            "gần 50/50; đánh giá trên tập kiểm tra vẫn cần thiết."
        ),
        "ly_do_scaler": (
            "StandardScaler đưa các điểm số về thang đo tương đương để tối ưu "
            "Logistic Regression ổn định hơn."
        ),
        "ly_do_pipeline": (
            "Imputer và Scaler chỉ được fit trên tập huấn luyện; tập kiểm tra "
            "không tham gia học tham số tiền xử lý."
        ),
        "duong_dan_mo_hinh": str(MODEL_PATH.relative_to(MODEL_PATH.parents[1])),
    }
    with (STEP_04_DIR / "thong_tin_huan_luyen.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(model_report, file, ensure_ascii=False, indent=2)

    print("HUẤN LUYỆN HOÀN TẤT")
    print(f"Mô hình: {metadata['model_type']}")
    print(f"Mẫu huấn luyện: {len(X_train)}")
    print(f"Số đặc trưng: {len(features)}")
    print(f"Số nhãn: {len(labels)}")
    print(f"Pipeline lưu tại: {MODEL_PATH}")
    print(f"Metadata lưu tại: {MODELS_DIR / 'metadata.json'}")
    print("Lưu ý: đây là mô hình dự đoán xu hướng nhãn, không phải đo mức độ phù hợp.")


if __name__ == "__main__":
    main()
