"""Step 4 - XGBoost on the per-hour tabular features (hospital A only).

Protocol (PROJECT_PLAN.md, Step 4): 40 random configurations x 3 seeds, selected on MEAN
val PR-AUC; the winner is re-run over 5 seeds. Early stopping watches val aucpr, the
metric that decides the project. No reweighting (scale_pos_weight = 1). The utility
threshold is tuned on val. Nothing here reads hospital B or the internal test split.

NaN is passed through untouched: XGBoost learns a default direction for missing values at
every split, which is the tree's native answer to "this lab was not drawn". That is the
clearest place where the tree and the logistic regression diverge in this project - the
linear model needs an imputed number plus a flag; the tree needs neither.

Usage:  python src/train_xgb.py [--n-iter 40]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
FEATS = ROOT / "data" / "processed" / "features_a.parquet"
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
ID_COLS = ["patient_id", "hour", "split", "SepsisLabel"]

SEEDS = [0, 1, 2, 3, 4]
SEARCH_SEEDS = [0, 1, 2]
SEARCH_SEED = 20260926
N_TREES_MAX = 3000


def sample_config(rng: np.random.Generator) -> dict:
    return {
        "max_depth": int(rng.integers(3, 10)),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "colsample_bytree": float(rng.uniform(0.3, 1.0)),
        # Guards against carving tiny pockets out of ~9,400 positive hours that come from
        # only 968 patients - positives are far less independent than their count suggests.
        "min_child_weight": float(np.exp(rng.uniform(np.log(1), np.log(200)))),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.5), np.log(50)))),
    }


def build(cfg: dict, seed: int, n_jobs: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(n_estimators=N_TREES_MAX, early_stopping_rounds=50, eval_metric="aucpr",
                             objective="binary:logistic", tree_method="hist", device="cpu",
                             scale_pos_weight=1.0, random_state=seed, n_jobs=n_jobs, verbosity=0, **cfg)


def load() -> dict:
    f = pd.read_parquet(FEATS)
    X = f.drop(columns=ID_COLS).astype("float32")
    out = {}
    for s in ["train", "val"]:
        m = (f["split"] == s).to_numpy()
        out[s] = {"X": X[m], "y": f.loc[m, "SepsisLabel"].astype(int).to_numpy(),
                  "pid": f.loc[m, "patient_id"].to_numpy(), "hour": f.loc[m, "hour"].to_numpy()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)

    D = load()
    tr, va = D["train"], D["val"]
    print(f"train {tr['X'].shape}  {tr['y'].mean():.2%} positive hours | val {va['X'].shape}")

    rng = np.random.default_rng(SEARCH_SEED)
    trials, t_all = [], time.perf_counter()
    for i in range(args.n_iter):
        cfg = sample_config(rng); t0 = time.perf_counter()
        scores, trees = [], []
        for s in SEARCH_SEEDS:
            m = build(cfg, s, args.n_jobs).fit(tr["X"], tr["y"], eval_set=[(va["X"], va["y"])], verbose=False)
            scores.append(ev.metrics(va["y"], m.predict_proba(va["X"])[:, 1])["pr_auc"])
            trees.append(int(m.best_iteration) + 1)
        trials.append({**cfg, "pr_auc_mean": float(np.mean(scores)), "pr_auc_sd": float(np.std(scores)),
                       "trees_per_seed": trees, "seconds": time.perf_counter() - t0})
        print(f"  [{i + 1:>2}/{args.n_iter}] PR-AUC {np.mean(scores):.4f} +-{np.std(scores):.4f}  "
              f"d={cfg['max_depth']} lr={cfg['learning_rate']:.3f} mcw={cfg['min_child_weight']:.1f} "
              f"sub={cfg['subsample']:.2f} col={cfg['colsample_bytree']:.2f} l2={cfg['reg_lambda']:.1f} "
              f"-> {trees} trees, {time.perf_counter() - t0:.0f}s", flush=True)

    best = max(trials, key=lambda r: r["pr_auc_mean"])
    best_cfg = {k: best[k] for k in sample_config(np.random.default_rng(0))}
    print(f"\nSearch took {(time.perf_counter() - t_all) / 60:.1f} min. Best config: {best_cfg}")

    runs, final, trees, val_preds = [], None, [], []
    for seed in SEEDS:
        m = build(best_cfg, seed, args.n_jobs).fit(tr["X"], tr["y"], eval_set=[(va["X"], va["y"])], verbose=False)
        p = m.predict_proba(va["X"])[:, 1]
        thr, _ = ev.best_utility_threshold(va["pid"], va["hour"], va["y"], p)
        runs.append(ev.metrics(va["y"], p, va["pid"], va["hour"], threshold=thr))
        trees.append(int(m.best_iteration) + 1); val_preds.append(p)
        final = final or m
        print(f"  seed {seed}: PR-AUC {runs[-1]['pr_auc']:.4f}  utility {runs[-1]['utility']:.4f}  ({trees[-1]} trees)", flush=True)

    summary = ev.seed_summary(runs)
    ev.print_metrics("xgboost", summary)
    imp = sorted(zip(tr["X"].columns, final.feature_importances_), key=lambda kv: -kv[1])[:20]
    print("\n  Top 20 features by gain (sanity check, not inference):")
    for n, g in imp:
        print(f"    {g:.4f}  {n}")

    final.save_model(MODELS / "xgb.json")
    np.save(REPORTS / "step4_xgb_val_preds.npy", np.array(val_preds, dtype="float32"))
    out = {"model": "xgboost", "selected_on": "mean val pr_auc over 3 seeds", "best_config": best_cfg,
           "search": {"n_iter": args.n_iter, "search_seed": SEARCH_SEED, "search_seeds": SEARCH_SEEDS},
           "trees_per_seed": trees, "seeds": SEEDS, "val": summary, "trials": trials,
           "feature_importance_gain": [{"feature": n, "gain": float(g)} for n, g in imp],
           "note": "Hospital A val only. Internal test and hospital B untouched until Step 6."}
    (REPORTS / "step4_xgb.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote reports/step4_xgb.json and models/xgb.json")


if __name__ == "__main__":
    main()
