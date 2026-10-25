from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, jaccard_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "CATBoost-SVR" / "dulieuhocsinh_10000.csv"
RESULTS = ROOT / "experiments" / "results"
SEEDS = [42, 101, 202, 303, 404]
SUBJECTS = ["Vật lý", "Hóa học", "Sinh học", "Công nghệ", "Tin học", "Địa lý",
            "Giáo dục kinh tế và pháp luật", "Âm nhạc", "Mỹ thuật"]


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "precision_at_4": float(np.mean((y * pred).sum(axis=1) / 4)),
        "recall_at_4": float(np.mean((y * pred).sum(axis=1) / y.sum(axis=1))),
        "jaccard": float(jaccard_score(y, pred, average="samples", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    feature_cols = [c for c in df.columns if c.endswith(tuple(f"_lop{i}" for i in range(6, 10)))]
    X = df[feature_cols].astype(float)
    yframe = df[SUBJECTS].apply(lambda c: c.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}))
    if yframe.isna().any().any():
        raise ValueError("Invalid/missing elective labels")
    y = yframe.to_numpy(dtype=int)
    if not np.all(y.sum(axis=1) == 4):
        raise ValueError("This evaluation protocol requires exactly four observed Yes labels per row")
    rows, per_label_rows = [], []
    for seed in SEEDS:
        idx = np.arange(len(X))
        trv, te = train_test_split(idx, test_size=.20, random_state=seed, shuffle=True)
        tr, va = train_test_split(trv, test_size=.25, random_state=seed, shuffle=True)
        # Hyperparameters are fixed a priori to bound runtime; no test-based selection.
        models = {
            "logistic_regression": MultiOutputClassifier(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))),
            "catboost": None,
            "svc": MultiOutputClassifier(make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, random_state=seed))),
            "xgboost": MultiOutputClassifier(XGBClassifier(n_estimators=160, max_depth=2, learning_rate=.05,
                subsample=.8, colsample_bytree=.8, reg_lambda=2.0, objective="binary:logistic",
                eval_metric="logloss", tree_method="hist", n_jobs=2, random_state=seed, verbosity=0)),
        }
        for name, model in models.items():
            if name == "catboost":
                probs=[]
                for j in range(y.shape[1]):
                    clf=CatBoostClassifier(iterations=180, depth=6, learning_rate=.05, loss_function="Logloss",
                        eval_metric="Logloss", random_seed=seed, verbose=False, allow_writing_files=False)
                    clf.fit(X.iloc[tr],y[tr,j],eval_set=(X.iloc[va],y[va,j]),early_stopping_rounds=25,verbose=False)
                    probs.append(clf.predict_proba(X.iloc[te])[:,1])
                p=np.column_stack(probs)
            else:
                model.fit(X.iloc[tr],y[tr])
                p=np.column_stack([z[:,1] for z in model.predict_proba(X.iloc[te])])
            top=np.argpartition(-p,kth=3,axis=1)[:,:4]
            pred=np.zeros_like(y[te]); np.put_along_axis(pred,top,1,axis=1)
            row={"seed":seed,"model":name,"status":"ok",**metrics(y[te],pred),
                 "roc_auc_macro":float(roc_auc_score(y[te],p,average="macro")),"n_train":len(tr),"n_test":len(te)}
            rows.append(row)
            for j, label in enumerate(SUBJECTS):
                per_label_rows.append({"seed":seed,"model":name,"label":label,
                    "roc_auc":float(roc_auc_score(y[te,j],p[:,j]))})
        # Popularity baseline is computed on train only.
        top_pop=np.argsort(-y[tr].mean(axis=0),kind="stable")[:4]
        pop=np.zeros_like(y[te]); pop[:,top_pop]=1
        rows.append({"seed":seed,"model":"popularity_top4","status":"ok",**metrics(y[te],pop),
                     "roc_auc_macro":None,"n_train":len(tr),"n_test":len(te)})
        rng=np.random.default_rng(seed)
        rand=np.zeros_like(y[te])
        for i in range(len(rand)):
            rand[i,rng.choice(y.shape[1],size=4,replace=False)]=1
        rows.append({"seed":seed,"model":"random_top4","status":"ok",**metrics(y[te],rand),
                     "roc_auc_macro":None,"n_train":len(tr),"n_test":len(te)})
        pd.DataFrame(rows).to_csv(RESULTS/"elective_metrics_by_seed.csv",index=False,encoding="utf-8")
        pd.DataFrame(per_label_rows).to_csv(RESULTS/"elective_per_label_roc_auc.csv",index=False,encoding="utf-8")
        print(f"Completed seed {seed}; rows={len(rows)}",flush=True)
    summary=pd.DataFrame(rows).groupby("model").agg({
        "precision_at_4":["mean","std"],"recall_at_4":["mean","std"],"jaccard":["mean","std"],
        "macro_f1":["mean","std"],"roc_auc_macro":["mean","std"]})
    summary.to_csv(RESULTS/"elective_metrics_summary.csv",encoding="utf-8")
    (RESULTS/"elective_protocol.json").write_text(json.dumps({"dataset":str(DATA.relative_to(ROOT)),
        "feature_count":len(feature_cols),"label_count":len(SUBJECTS),"seeds":SEEDS,
        "split":"60/20/20 by independent source row","labels_each_row":4,
        "interpretation":"observed Yes/No preference/selection labels; not suitability",
        "threshold_comparison":"Top-4 metrics are not comparable to threshold-0.5 per-label F1/exact match"},
        ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
