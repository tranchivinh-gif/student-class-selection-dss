from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "results"
DATA = ROOT / "CATBoost-SVR" / "dulieuhocsinh_50000_mo_phong.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def r2_table(df: pd.DataFrame, targets: list[str]) -> list[dict[str, object]]:
    train = df[df.tap_du_lieu == "train"]
    test = df[df.tap_du_lieu == "test"]
    rows = []
    for target in targets:
        subject = target.removeprefix("diem_thi_").removesuffix("_mo_phong")
        last = f"{subject}_lop9"
        mock = f"diem_thi_thu_{subject}"
        candidates = {
            "lop9": [last],
            "lop9_va_thi_thu": [last, mock],
            "day_du": [c for c in df.columns if c.endswith(tuple(f"_lop{i}" for i in range(6, 10)))
                       or c in ("gio_tu_hoc_tuan", "ti_le_chuyen_can", mock)],
        }
        for name, cols in candidates.items():
            model = LinearRegression().fit(train[cols], train[target])
            pred = model.predict(test[cols])
            rows.append({"target": target, "featureset": name, "n_features": len(cols),
                         "test_r2": float(r2_score(test[target], pred))})
    return rows


def near_duplicate_summary(df: pd.DataFrame, features: list[str]) -> list[dict[str, object]]:
    # Sample up to 300 test groups; all records within a sampled group are retained.
    test_groups = df.loc[df.tap_du_lieu == "test", "nhom_ho_so_goc"].drop_duplicates()
    sample_groups = test_groups.sample(n=min(300, len(test_groups)), random_state=42)
    test = df[df.nhom_ho_so_goc.isin(sample_groups)].copy()
    train = df[df.tap_du_lieu == "train"].copy()
    scaler = StandardScaler().fit(train[features])
    xt = scaler.transform(train[features])
    xv = scaler.transform(test[features])
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="euclidean", n_jobs=-1).fit(xt)
    d_cross = nn.kneighbors(xv, return_distance=True)[0][:, 0]
    # Within-group pair distances; sample a maximum of five points per group to bound work.
    d_within = []
    for _, group in test.groupby("nhom_ho_so_goc", sort=False):
        arr = scaler.transform(group[features].head(5))
        if len(arr) > 1:
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    d_within.append(float(np.linalg.norm(arr[i] - arr[j])))
    return [
        {"comparison": "test_to_train_nearest_other_group", "n_rows": len(test),
         "n_test_groups": test.nhom_ho_so_goc.nunique(), "mean": float(np.mean(d_cross)),
         "median": float(np.median(d_cross)), "p05": float(np.quantile(d_cross, .05)),
         "p95": float(np.quantile(d_cross, .95))},
        {"comparison": "within_test_group_pair_sample", "n_rows": len(d_within),
         "n_test_groups": test.nhom_ho_so_goc.nunique(), "mean": float(np.mean(d_within)),
         "median": float(np.median(d_within)), "p05": float(np.quantile(d_within, .05)),
         "p95": float(np.quantile(d_within, .95))},
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [
        ROOT / "CATBoost-SVR" / "dulieuhocsinh_10000.csv",
        ROOT / "XGBoost" / "data" / "hoc_sinh_moi.csv",
        ROOT / "KNN-HQTT" / "Logistic_Regression_TuVanToHop" / "data" / "raw" / "dulieuhocsinh_10000.csv",
    ]
    identity = []
    frames = {}
    for path in paths:
        df0 = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
        df0.columns = df0.columns.str.strip()
        # Read only headers for comparability; full data read below for exact match.
        frames[str(path.relative_to(ROOT))] = pd.read_csv(path, encoding="utf-8-sig")
        frames[str(path.relative_to(ROOT))].columns = frames[str(path.relative_to(ROOT))].columns.str.strip()
        df = frames[str(path.relative_to(ROOT))]
        identity.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                         "rows": int(len(df)), "columns": int(len(df.columns)),
                         "schema_sha256": hashlib.sha256("\n".join(df.columns).encode()).hexdigest(),
                         "label_columns": [c for c in df.columns if c not in ["ma_hoc_sinh"] and
                                           not c.endswith(tuple(f"_lop{i}" for i in range(6, 10)))]})
    pd.DataFrame(identity).to_csv(OUT / "audit_10k_files.csv", index=False, encoding="utf-8")
    (OUT / "audit_10k_identity.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.read_csv(DATA, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    feature_cols = [c for c in df.columns if c.endswith(tuple(f"_lop{i}" for i in range(6, 10)))
                    or c in ("gio_tu_hoc_tuan", "ti_le_chuyen_can", "diem_thi_thu_toan",
                             "diem_thi_thu_ngu_van", "diem_thi_thu_ngoai_ngu")]
    score_cols = [c for c in df.columns if c.endswith(tuple(f"_lop{i}" for i in range(6, 10)))
                  or c.startswith("diem_thi_")]
    group_sizes = df.groupby("nhom_ho_so_goc", sort=False).size().value_counts().sort_index()
    audit = {
        "path": str(DATA.relative_to(ROOT)), "sha256": sha256(DATA), "shape": list(df.shape),
        "dtypes": {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
        "missing_cells": int(df.isna().sum().sum()), "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df.ma_hoc_sinh.duplicated().sum()), "groups": int(df.nhom_ho_so_goc.nunique()),
        "group_size_distribution": {str(int(k)): int(v) for k, v in group_sizes.items()},
        "score_domain": {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                              "out_of_0_10": int((~df[c].between(0, 10)).sum())} for c in score_cols},
        "split_counts": {k: int(v) for k, v in df.tap_du_lieu.value_counts().items()},
        "split_group_counts": {k: int(df.loc[df.tap_du_lieu == k, "nhom_ho_so_goc"].nunique())
                               for k in df.tap_du_lieu.unique()},
    }
    (OUT / "audit_50k.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    audit["group_variation"] = []
    grouped = df.groupby("nhom_ho_so_goc", sort=False)
    for col in feature_cols:
        within = grouped[col].var(ddof=0).mean()
        between = grouped[col].mean().var(ddof=0)
        total = df[col].var(ddof=0)
        audit["group_variation"].append({"feature": col, "within_group_variance_mean": float(within),
                                         "between_group_variance": float(between),
                                         "within_fraction_total": float(within / total) if total else 0.0,
                                         "between_fraction_total": float(between / total) if total else 0.0})
    pd.DataFrame(audit.pop("group_variation")).to_csv(OUT / "audit_group_variation.csv", index=False, encoding="utf-8")
    (OUT / "audit_50k.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    targets = [c for c in df.columns if c.startswith("diem_thi_") and c.endswith("_mo_phong")]
    pd.DataFrame(r2_table(df, targets)).to_csv(OUT / "audit_target_linear_r2.csv", index=False, encoding="utf-8")
    corr_rows = []
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    for target in targets:
        for feature in numeric_features:
            corr_rows.append({"target": target, "feature": feature,
                "pearson": float(df[target].corr(df[feature], method="pearson")),
                "spearman": float(df[target].corr(df[feature], method="spearman"))})
    pd.DataFrame(corr_rows).to_csv(OUT / "audit_target_correlations.csv", index=False, encoding="utf-8")
    pd.DataFrame(near_duplicate_summary(df, feature_cols)).to_csv(OUT / "audit_nearest_distances.csv", index=False, encoding="utf-8")
    print("Audit completed", json.dumps({"shape": df.shape, "groups": audit["groups"], "targets": targets}))


if __name__ == "__main__":
    main()
