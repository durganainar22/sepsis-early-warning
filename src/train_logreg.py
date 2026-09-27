"""Step 4 - L2 logistic regression on the per-hour tabular features (hospital A only).

The simplest credible model, given a fair shot: a dense C sweep, selected on val PR-AUC,
with a warning if the optimum lands on the edge of the grid.

WHERE THIS DIVERGES FROM THE TREE: a linear model cannot take NaN, so it needs a number
for every missing value - and the choice of number is a modelling decision:
  *_hours_since   never measured -> the cap (72 h). That is literally what it means.
  *_slope_*       fewer than 2 points -> 0: "no trend known", the neutral value for a slope.
  everything else train median. The *_measured_yet flags already tell the model the value
                  was imputed, so the median is a placeholder, not a claim.
Then standardize. All of it is fitted on TRAIN only and applied unchanged to val.

lbfgs is deterministic given the data, so seeds cannot change the result; the readmission
project measured that (sd exactly 0). One fit per C, no seed loop.

Usage:  python src/train_logreg.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

import evaluate as ev
from build_features import SINCE_CAP
from train_xgb import ID_COLS, FEATS, MODELS, REPORTS

C_GRID = np.logspace(-4, 1, 11)


class Prep:
    """Train-fitted imputation + standardization, in the order it is applied."""

    def fit(self, X: pd.DataFrame) -> "Prep":
        self.cols = list(X.columns)
        self.since = [c for c in self.cols if c.endswith("_hours_since")]
        self.slope = [c for c in self.cols if "_slope_" in c]
        self.binary = [c for c in self.cols if c.endswith("_measured_yet") or c.startswith("icu_") or c == "Gender"]
        self.median = X.median()
        Z = self._impute(X)
        scale = [c for c in self.cols if c not in self.binary]
        self.mu, self.sd = Z[scale].mean(), Z[scale].std().replace(0, 1.0)
        self.scale = scale
        return self

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = X.copy()
        Z[self.since] = Z[self.since].fillna(SINCE_CAP)
        Z[self.slope] = Z[self.slope].fillna(0.0)
        return Z.fillna(self.median)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._impute(X[self.cols])
        Z[self.scale] = (Z[self.scale] - self.mu) / self.sd
        assert not Z.isna().any().any(), "NaN survived preprocessing"
        return Z.to_numpy(dtype="float64")


def main() -> None:
    f = pd.read_parquet(FEATS)
    X = f.drop(columns=ID_COLS)
    tr, va = (f["split"] == "train").to_numpy(), (f["split"] == "val").to_numpy()
    y_tr, y_va = f.loc[tr, "SepsisLabel"].astype(int).to_numpy(), f.loc[va, "SepsisLabel"].astype(int).to_numpy()
    prep = Prep().fit(X[tr])
    Xtr, Xva = prep.transform(X[tr]), prep.transform(X[va])
    print(f"train {Xtr.shape}  val {Xva.shape}  | sweeping C over {len(C_GRID)} values")

    sweep, models = [], {}
    for C in C_GRID:
        t0 = time.perf_counter()
        m = LogisticRegression(l1_ratio=0.0, C=C, solver="lbfgs", max_iter=5000, class_weight=None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            m.fit(Xtr, y_tr)
        ok = not any(issubclass(x.category, ConvergenceWarning) for x in w)
        r = ev.metrics(y_va, m.predict_proba(Xva)[:, 1])
        sweep.append({"C": float(C), **r, "converged": ok}); models[C] = m
        print(f"  C={C:>9.4g}  PR-AUC {r['pr_auc']:.4f}  ROC-AUC {r['roc_auc']:.4f}  "
              f"{time.perf_counter() - t0:5.0f}s{'' if ok else '  [NOT CONVERGED]'}", flush=True)

    best = max(sweep, key=lambda r: r["pr_auc"]); C = best["C"]
    if C in (C_GRID[0], C_GRID[-1]):
        print(f"  WARNING: best C={C:g} is at the edge of the grid - widen it.")
    m = models[C]; p = m.predict_proba(Xva)[:, 1]
    pid, hour = f.loc[va, "patient_id"].to_numpy(), f.loc[va, "hour"].to_numpy()
    thr, _ = ev.best_utility_threshold(pid, hour, y_va, p)
    res = ev.metrics(y_va, p, pid, hour, threshold=thr)
    ev.print_metrics(f"logreg (L2, C={C:g})", res)

    coef = pd.Series(m.coef_.ravel(), index=prep.cols)
    top = coef.reindex(coef.abs().sort_values(ascending=False).index)[:20]
    print("\n  Top 20 coefficients (standardized, sanity check - not inference):")
    for n, v in top.items():
        print(f"    {v:+.4f}  {n}")

    joblib.dump({"model": m, "prep": prep, "C": C}, MODELS / "logreg.joblib")
    np.save(REPORTS / "step4_logreg_val_preds.npy", p.astype("float32")[None, :])
    out = {"model": "logreg_l2", "best_C": C, "val": res, "sweep": sweep,
           "top_coefficients": [{"feature": n, "coef": float(v)} for n, v in top.items()],
           "note": "Hospital A val only. lbfgs is deterministic: seed sd is 0 by construction."}
    (REPORTS / "step4_logreg.json").write_text(json.dumps(out, indent=2))
    print("\nWrote reports/step4_logreg.json and models/logreg.joblib")


if __name__ == "__main__":
    main()
