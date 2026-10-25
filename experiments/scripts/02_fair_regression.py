from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
CONFIG_PATH = ROOT / "experiments" / "config" / "regression.yaml"
DATA_PATH = ROOT / "CATBoost-SVR" / "dulieuhocsinh_50000_mo_phong.csv"
MOCK = {"diem_thi_toan_mo_phong": "diem_thi_thu_toan",
        "diem_thi_ngu_van_mo_phong": "diem_thi_thu_ngu_van",
        "diem_thi_ngoai_ngu_mo_phong": "diem_thi_thu_ngoai_ngu"}
BASELINE = {"diem_thi_toan_mo_phong": "toan_lop9",
            "diem_thi_ngu_van_mo_phong": "ngu_van_lop9",
            "diem_thi_ngoai_ngu_mo_phong": "ngoai_ngu_lop9"}


def worker(kind: str, payload: dict, output_dir: str) -> None:
    out = Path(output_dir)
    try:
        xtr, ytr = payload["xtr"], payload["ytr"]
        xval, yval = payload["xval"], payload["yval"]
        xte = payload["xte"]
        start = time.perf_counter()
        if kind == "catboost":
            model = CatBoostRegressor(**payload["params"], verbose=False, allow_writing_files=False)
        else:
            model = make_pipeline(StandardScaler(), SVR(**payload["params"]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(xtr, ytr)
        fit_seconds = time.perf_counter() - start
        fit_status = "ok"
        if kind == "svr" and model.named_steps["svr"].fit_status_ != 0:
            fit_status = "non_converged"
        start = time.perf_counter()
        val_pred = np.asarray(model.predict(xval), dtype=float)
        test_pred = np.asarray(model.predict(xte), dtype=float)
        predict_seconds = time.perf_counter() - start
        np.savez_compressed(out / "pred.npz", val_pred=val_pred, test_pred=test_pred)
        (out / "meta.json").write_text(json.dumps({"fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds, "status": fit_status,
            "warnings": [str(w.message) for w in caught]}), encoding="utf-8")
    except BaseException:
        (out / "meta.json").write_text(json.dumps({"status": "error",
            "error": traceback.format_exc()}), encoding="utf-8")


def fit_capped(kind: str, payload: dict, timeout: int) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    with tempfile.TemporaryDirectory(prefix="fit_", dir=RESULTS) as temp:
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=worker, args=(kind, payload, temp))
        proc.start()
        proc.join(timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            return {"status": "timeout", "fit_seconds": float(timeout), "predict_seconds": None,
                    "error": f"Process exceeded {timeout}s fit-and-predict cap"}, None, None
        meta_path = Path(temp) / "meta.json"
        if not meta_path.exists():
            return {"status": "error", "fit_seconds": None, "predict_seconds": None,
                    "error": f"Worker exited with code {proc.exitcode} and no result"}, None, None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["status"] not in ("ok", "non_converged"):
            return meta, None, None
        with np.load(Path(temp) / "pred.npz") as p:
            return meta, p["val_pred"].copy(), p["test_pred"].copy()


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p))}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    group_ids = np.array(sorted(df.nhom_ho_so_goc.unique()))
    grade_features = [c for c in df.columns if c.endswith(tuple(f"_lop{i}" for i in range(6, 10)))]
    common = grade_features + ["gio_tu_hoc_tuan", "ti_le_chuyen_can"]
    rows, timings, pair_rows = [], [], []
    pair_error_cache = {}
    total = len(cfg["seeds"]) * len(cfg["train_group_counts"]) * len(cfg["feature_sets"]) * len(cfg["targets"]) * 2
    done = 0
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(group_ids)
        n_train = int(len(shuffled) * .60); n_val = int(len(shuffled) * .20)
        train_groups, val_groups, test_groups = shuffled[:n_train], shuffled[n_train:n_train+n_val], shuffled[n_train+n_val:]
        assert not (set(train_groups) & set(val_groups) or set(train_groups) & set(test_groups) or set(val_groups) & set(test_groups))
        train_all = df[df.nhom_ho_so_goc.isin(train_groups)]
        val = df[df.nhom_ho_so_goc.isin(val_groups)]
        test = df[df.nhom_ho_so_goc.isin(test_groups)]
        nested = rng.permutation(train_groups)
        for ng in cfg["train_group_counts"]:
            selected = nested[:ng]
            tr = train_all[train_all.nhom_ho_so_goc.isin(selected)]
            assert tr.nhom_ho_so_goc.nunique() == ng
            for fs in cfg["feature_sets"]:
                for target in cfg["targets"]:
                    features = common + ([MOCK[target]] if fs == "full_with_mock_score" else [])
                    xtr, ytr = tr[features].to_numpy(dtype=float), tr[target].to_numpy(dtype=float)
                    xv, yv = val[features].to_numpy(dtype=float), val[target].to_numpy(dtype=float)
                    xt, yt = test[features].to_numpy(dtype=float), test[target].to_numpy(dtype=float)
                    tg = test.nhom_ho_so_goc.to_numpy()
                    base_pred = test[BASELINE[target]].to_numpy(dtype=float)
                    for kind in ("catboost", "svr"):
                        params = cfg[kind].copy()
                        if kind == "catboost":
                            params["random_seed"] = seed
                        done += 1
                        print(f"[{done}/{total}] seed={seed} groups={ng} set={fs} target={target} model={kind}", flush=True)
                        meta, vp, tp = fit_capped(kind, {"xtr": xtr, "ytr": ytr, "xval": xv,
                            "yval": yv, "xte": xt, "params": params}, cfg["fit_timeout_seconds"])
                        row = {"seed": seed, "train_groups": ng, "train_rows": len(tr),
                               "validation_rows": len(val), "test_rows": len(test), "feature_set": fs,
                               "target": target, "model": kind, "status": meta["status"],
                               "mae": None, "rmse": None, "r2": None, "validation_mae": None,
                               "fit_seconds": meta.get("fit_seconds"), "predict_seconds": meta.get("predict_seconds"),
                               "feature_count": len(features), "error": meta.get("error")}
                        if tp is not None:
                            row.update(metrics(yt, tp)); row["validation_mae"] = float(mean_absolute_error(yv, vp))
                            grperr = pd.DataFrame({"group": tg, "err": np.abs(yt-tp)}).groupby("group").err.mean()
                            if meta["status"] == "ok":
                                pair_error_cache[(seed,ng,fs,target,kind)] = grperr
                            # residual summary by contemporaneous grade-9 score quintile.
                            grade = test[BASELINE[target]].to_numpy()
                            bins = pd.qcut(grade, q=5, duplicates="drop", labels=False)
                            for b in np.unique(bins):
                                mask = bins == b
                                timings.append({"seed":seed,"train_groups":ng,"feature_set":fs,"target":target,
                                    "model":kind,"status":"residual_bin","grade9_bin":int(b),
                                    "grade9_mean":float(grade[mask].mean()),"residual_mae":float(np.mean(np.abs(yt[mask]-tp[mask])))})
                        rows.append(row)
                        timings.append({"seed":seed,"train_groups":ng,"feature_set":fs,"target":target,
                            "model":kind,"status":meta["status"],"fit_seconds":row["fit_seconds"],
                            "predict_seconds":row["predict_seconds"],"warnings":json.dumps(meta.get("warnings",[])),
                            "error":meta.get("error")})
                    # Baseline and Ridge; alpha selection uses validation only.
                    for modelname in ("baseline_lop9", "ridge"):
                        start=time.perf_counter()
                        if modelname == "baseline_lop9":
                            pred=base_pred; best_alpha=None
                        else:
                            best_alpha=min(cfg["ridge_alpha_candidates"], key=lambda a: mean_absolute_error(
                                yv, make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(xtr,ytr).predict(xv)))
                            ridge=make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
                            ridge.fit(xtr,ytr)
                            pred=ridge.predict(xt)
                        elapsed=time.perf_counter()-start
                        row={"seed":seed,"train_groups":ng,"train_rows":len(tr),"validation_rows":len(val),
                             "test_rows":len(test),"feature_set":fs,"target":target,"model":modelname,
                             "status":"ok",**metrics(yt,pred),"validation_mae":None,"fit_seconds":elapsed,
                             "predict_seconds":None,"feature_count":len(features),"ridge_alpha":best_alpha,"error":None}
                        rows.append(row)
                        grperr=pd.DataFrame({"group":tg,"err":np.abs(yt-pred)}).groupby("group").err.mean()
                        pair_error_cache[(seed,ng,fs,target,modelname)]=grperr
                        timings.append({"seed":seed,"train_groups":ng,"feature_set":fs,"target":target,
                            "model":modelname,"status":"ok","fit_seconds":elapsed,"predict_seconds":None})
        # Checkpoint after each seed so an interrupted run keeps completed results.
        pd.DataFrame(rows).to_csv(RESULTS/"regression_metrics.csv",index=False,encoding="utf-8")
        pd.DataFrame(timings).to_csv(RESULTS/"regression_timing.csv",index=False,encoding="utf-8")
    rng=np.random.default_rng(cfg["bootstrap_seed"])
    for key in list(pair_error_cache):
        seed,ng,fs,target,kind=key
        if kind != "catboost": continue
        svr=pair_error_cache.get((seed,ng,fs,target,"svr"))
        if svr is None: continue
        cat=pair_error_cache[key]
        common_groups=cat.index.intersection(svr.index)
        a=cat.loc[common_groups].to_numpy(); b=svr.loc[common_groups].to_numpy()
        ix=rng.integers(0,len(a),size=(cfg["bootstrap_replicates"],len(a)))
        diff=(a[ix]-b[ix]).mean(axis=1)
        pair_rows.append({"seed":seed,"train_groups":ng,"feature_set":fs,"target":target,
            "difference_definition":"CatBoost MAE - SVR MAE; negative favors CatBoost",
            "mae_difference":float(a.mean()-b.mean()),"ci95_low":float(np.quantile(diff,.025)),
            "ci95_high":float(np.quantile(diff,.975)),"ci_contains_zero":bool(np.quantile(diff,.025)<=0<=np.quantile(diff,.975)),
            "bootstrap_replicates":cfg["bootstrap_replicates"],"test_groups":len(a)})
    pd.DataFrame(pair_rows).to_csv(RESULTS/"regression_pairwise_ci.csv",index=False,encoding="utf-8")
    print(f"Completed; metrics={len(rows)}, paired CIs={len(pair_rows)}")


if __name__ == "__main__":
    main()
