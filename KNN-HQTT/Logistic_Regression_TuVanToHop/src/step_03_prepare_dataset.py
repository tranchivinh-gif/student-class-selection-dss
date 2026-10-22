"""Xác định đặc trưng/nhãn và chia dữ liệu thành tập huấn luyện, kiểm tra."""

from __future__ import annotations

import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from common import (
    ID_COLUMN_CANDIDATES,
    PROCESSED_CSV,
    STEP_03_DIR,
    identify_score_columns,
    identify_yes_no_columns,
)


TEST_SIZE = 0.20
RANDOM_STATE = 42


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not PROCESSED_CSV.is_file():
        raise FileNotFoundError(
            "Chưa có dữ liệu đã làm sạch. Hãy chạy "
            "python src/step_02_clean_data.py trước."
        )

    data = pd.read_csv(PROCESSED_CSV, encoding="utf-8-sig")
    data.columns = [str(column).strip() for column in data.columns]
    feature_columns = identify_score_columns(data)
    label_columns = identify_yes_no_columns(data)
    id_columns = [
        column
        for column in data.columns
        if str(column).strip().casefold() in ID_COLUMN_CANDIDATES
    ]

    if not feature_columns:
        raise ValueError("Không tìm thấy cột điểm để tạo X.")
    if not label_columns:
        raise ValueError(
            "Không tìm thấy nhãn yes/no hợp lệ. Không tạo nhãn giả để huấn luyện."
        )

    X = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    if X.isna().all(axis=0).any():
        invalid_columns = X.columns[X.isna().all(axis=0)].tolist()
        raise ValueError(f"Cột điểm không có giá trị số hợp lệ: {invalid_columns}")

    raw_y = data[label_columns].astype("string").apply(
        lambda column: column.str.strip().str.casefold()
    )
    invalid_labels = {
        column: sorted(raw_y[column].dropna().unique().tolist())
        for column in label_columns
        if not set(raw_y[column].dropna().unique()).issubset({"yes", "no"})
    }
    if invalid_labels:
        raise ValueError(f"Có giá trị nhãn không phải yes/no: {invalid_labels}")
    if raw_y.isna().any().any():
        missing = raw_y.isna().sum()
        raise ValueError(
            "Có nhãn lựa chọn bị thiếu. Không tự đoán nhãn để huấn luyện. "
            f"Số thiếu theo cột: {missing[missing > 0].to_dict()}"
        )

    y = raw_y.eq("yes").astype("int8")
    one_class_labels = [
        column for column in label_columns if y[column].nunique() < 2
    ]
    if one_class_labels:
        raise ValueError(
            "Các nhãn sau chỉ có một lớp, không huấn luyện phân loại được: "
            f"{one_class_labels}"
        )

    row_indices = pd.Series(data.index, name="row_index")
    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]

    STEP_03_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "X_train.csv": X_train,
        "X_test.csv": X_test,
        "y_train.csv": y_train,
        "y_test.csv": y_test,
    }
    for filename, frame in outputs.items():
        frame.to_csv(STEP_03_DIR / filename, index=False, encoding="utf-8-sig")

    distribution = pd.DataFrame(
        {
            "nhan": label_columns,
            "train_yes": [int(y_train[column].sum()) for column in label_columns],
            "train_no": [
                int((y_train[column] == 0).sum()) for column in label_columns
            ],
            "test_yes": [int(y_test[column].sum()) for column in label_columns],
            "test_no": [
                int((y_test[column] == 0).sum()) for column in label_columns
            ],
        }
    )
    distribution.to_csv(
        STEP_03_DIR / "phan_bo_nhan.csv", index=False, encoding="utf-8-sig"
    )

    metadata = {
        "feature_columns": feature_columns,
        "label_columns": label_columns,
        "id_columns_excluded": id_columns,
        "test_size": TEST_SIZE,
        "train_size": 1.0 - TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_rows": len(data),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "split_files": list(outputs),
        "label_mapping": {"no": 0, "yes": 1},
    }
    with (STEP_03_DIR / "dataset_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("ĐÃ CHUẨN BỊ X VÀ y")
    print(f"X: {len(feature_columns)} cột điểm: {feature_columns}")
    print(f"y: {len(label_columns)} nhãn yes/no: {label_columns}")
    print(f"Cột mã học sinh loại khỏi X: {id_columns or 'không tìm thấy'}")
    print(
        f"Chia tập: train={len(X_train)} ({len(X_train)/len(data):.0%}), "
        f"test={len(X_test)} ({len(X_test)/len(data):.0%}), "
        f"random_state={RANDOM_STATE}"
    )
    print(
        "Không dùng tập kiểm tra để học tham số tiền xử lý; Imputer/Scaler sẽ "
        "được fit bên trong Pipeline chỉ bằng X_train ở bước 4."
    )
    print(f"Đã lưu X_train, X_test, y_train, y_test và phân bố nhãn tại {STEP_03_DIR}")


if __name__ == "__main__":
    main()
