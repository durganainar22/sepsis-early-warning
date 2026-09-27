"""One scoring implementation for every model in the project.

If each training script computed its own metrics, the benchmark table would compare call
sites as much as models. Everything is scored here.

Metrics (PROJECT_PLAN.md, Step 2 decision 7):
  pr_auc         DECIDES the comparison. Threshold-free, and at ~2% positive hours it tracks
                 the minority class the project is about; ROC-AUC is reported alongside.
  utility        the challenge's normalised utility score: an alarm earns full credit 6 h
                 before onset, sliding to 0 at 12 h before and at 3 h after; a missed case
                 costs up to -2; each false-alarm hour costs -0.05. 1 = perfect, 0 = never
                 alarm.
                 Threshold-DEPENDENT, so the threshold is tuned on validation and carried
                 to test unchanged.
  brier          calibration check - no reweighting was chosen partly to keep it good.

Scoring is over SCORED hours only (row index >= 6, Step 2 decision 1). For the utility
this is exactly equivalent to the official score with no alarms allowed in the 6-hour
warm-up, applied to both the model and the "best possible" reference, so normalisation
stays fair. (Excluded patients were the only ones whose label switched on in the warm-up,
so no septic patient's reward window is cut short by more than the forced silence.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PRIMARY_METRIC = "pr_auc"

# Official parameters, from physionetchallenges/evaluation-2019 (BSD-2-Clause).
DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1.0, -2.0, -0.05, 0.0


def _row_utility(d: np.ndarray, septic: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Vectorized port of compute_prediction_utility(); d = t - t_sepsis (inf if never septic)."""
    m1, b1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY), -MAX_U_TP / (DT_OPTIMAL - DT_EARLY) * DT_EARLY
    m2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL); b2 = -m2 * DT_LATE
    m3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL); b3 = -m3 * DT_OPTIMAL
    u = np.zeros(len(d))
    live = d <= DT_LATE                       # after onset + 3 h nothing counts
    early = d <= DT_OPTIMAL
    tp, fn = septic & pred & live, septic & ~pred & live
    u[tp & early] = np.maximum(m1 * d[tp & early] + b1, U_FP)
    u[tp & ~early] = m2 * d[tp & ~early] + b2
    u[fn & ~early] = m3 * d[fn & ~early] + b3
    u[~septic & pred] = U_FP
    u[~septic & ~pred] = U_TN
    return u


def _onset_offset(patient_id: np.ndarray, hour: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"p": patient_id, "h": hour, "y": y})
    first = df[df.y == 1].groupby("p")["h"].min()
    t_sepsis = df["p"].map(first - DT_OPTIMAL)        # onset = first label hour + 6
    septic = t_sepsis.notna().to_numpy()
    # np.array, not .to_numpy(): pandas 3 copy-on-write hands back a read-only view.
    d = np.array(df["h"] - t_sepsis, dtype="float64")
    d[~septic] = np.inf
    return d, septic


def normalized_utility(patient_id, hour, y, pred) -> float:
    """(observed - inaction) / (best - inaction), summed over all patients - as officially."""
    d, septic = _onset_offset(np.asarray(patient_id), np.asarray(hour), np.asarray(y))
    pred = np.asarray(pred).astype(bool)
    best = septic & (d >= DT_EARLY) & (d <= DT_LATE)
    obs = _row_utility(d, septic, pred).sum()
    opt = _row_utility(d, septic, best).sum()
    none = _row_utility(d, septic, np.zeros_like(pred)).sum()
    return float((obs - none) / (opt - none))


def best_utility_threshold(patient_id, hour, y, p, n_grid: int = 200) -> tuple[float, float]:
    """Threshold maximizing utility - to be called on VALIDATION only, never test."""
    qs = np.unique(np.quantile(p, np.linspace(0.80, 0.9999, n_grid)))
    scores = [normalized_utility(patient_id, hour, y, p >= q) for q in qs]
    i = int(np.argmax(scores))
    return float(qs[i]), float(scores[i])


def metrics(y, p, patient_id=None, hour=None, threshold: float | None = None) -> dict:
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype="float64")
    out = {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
           "brier": float(brier_score_loss(y, p)), "base_rate": float(y.mean()), "n_hours": int(len(y))}
    if threshold is not None:
        out["threshold"] = float(threshold)
        out["utility"] = normalized_utility(patient_id, hour, y, p >= threshold)
        flag = p >= threshold
        out["alarm_rate"] = float(flag.mean())
        out["precision_at_threshold"] = float(y[flag].mean()) if flag.any() else float("nan")
        out["recall_at_threshold"] = float(flag[y == 1].mean())
    return out


def seed_summary(runs: list[dict]) -> dict:
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    out = {k: {"mean": float(np.mean([r[k] for r in runs])), "sd": float(np.std([r[k] for r in runs])),
               "min": float(np.min([r[k] for r in runs])), "max": float(np.max([r[k] for r in runs]))}
           for k in keys}
    out["n_seeds"] = len(runs)
    return out


def print_metrics(name: str, m: dict, split: str = "val") -> None:
    g = lambda k: m[k]["mean"] if isinstance(m.get(k), dict) else m.get(k)
    sd = lambda k: f" +- {m[k]['sd']:.4f}" if isinstance(m.get(k), dict) else ""
    print(f"\n  {name}  [{split}]")
    print(f"    PR-AUC   {g('pr_auc'):.4f}{sd('pr_auc')}   <- deciding metric (base rate {g('base_rate'):.2%})")
    print(f"    ROC-AUC  {g('roc_auc'):.4f}{sd('roc_auc')}    Brier {g('brier'):.4f}")
    if g("utility") is not None:
        print(f"    utility  {g('utility'):.4f}{sd('utility')} at threshold {g('threshold'):.4f} "
              f"(alarms on {g('alarm_rate'):.1%} of hours, precision {g('precision_at_threshold'):.1%}, "
              f"recall {g('recall_at_threshold'):.1%})")
