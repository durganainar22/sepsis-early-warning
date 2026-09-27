"""Step 3 - per-hour features for the TABULAR models (logistic regression, XGBoost).

A tabular model sees one row at a time and has no memory, so each hour's row must carry a
summary of the patient's past. The GRU in Step 5 does NOT use this file: it reads the raw
hourly sequence and learns its own summaries. That contrast - hand-built history versus
learned history - is the comparison this project exists to make.

CAUSALITY IS THE WHOLE GAME. Every feature at hour t uses rows <= t only (PROJECT_PLAN.md,
Step 2 decision 2). Septic records end 9 h after the label switches on, so anything that
peeks forward would leak the outcome. This is not left to careful coding: the script ends
with a truncation test - recompute features on records cut off at a random hour and check
they equal the full-record features at that hour. Any look-ahead makes them differ.

Features per hour (decisions in PROJECT_PLAN.md, Step 3):
  all 27 measurements   last value (carried forward), hours since last measured,
                        measured-yet flag
  labs (20)             number of measurements in the last 24 h   - testing intensity
  vitals (7)            mean / min / max / slope over the last 6 h and 24 h
  context               Age, Gender, ICULOS, HospAdmTime, ICU type (medical/surgical/unknown)

Missing stays missing (NaN). XGBoost handles NaN natively; the logistic regression step
fits its own imputation on train. Nothing here is fitted, so nothing here can leak across
splits - every value is a function of one patient's own past.

Output: data/processed/features_a.parquet  (scored hours only, with label and split)
        data/processed/features_b.parquet  (scored-eligible hours, NO label)

Usage:  python src/build_features.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
META = json.loads((PROC / "cohort.meta.json").read_text())
KEPT = META["kept"]
VITALS = [v for v in data.VITALS if v in KEPT]          # 7 (EtCO2 was dropped)
LABS = [v for v in KEPT if v not in VITALS]             # 20
WINDOWS = [6, 24]
SINCE_CAP = 72  # hours; "measured 3 days ago" and "longer" carry the same information here
TEST_PATIENTS = 300


def _pad(df: pd.DataFrame, cols: list[str], pad: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack patients with `pad` NaN rows between them.

    Trailing-window statistics over the padded array can then never reach back into the
    previous patient: whatever the window overlaps before a patient's first row is NaN, and
    every statistic below ignores NaN. One vectorized pass instead of 20k groupby-rollings.
    Returns the padded values and a mask selecting the real rows, in the original order.
    """
    n_pat = df["patient_id"].nunique()
    starts = np.flatnonzero(df["patient_id"].to_numpy()[1:] != df["patient_id"].to_numpy()[:-1]) + 1
    real_pos = np.arange(len(df)) + pad * (np.searchsorted(starts, np.arange(len(df)), side="right") + 1)
    out = np.full((len(df) + pad * (n_pat + 1), len(cols)), np.nan, dtype="float64")
    out[real_pos] = df[cols].to_numpy(dtype="float64")
    return out, real_pos


def _rolling(arr: np.ndarray, w: int, how: str) -> np.ndarray:
    return getattr(pd.DataFrame(arr).rolling(w, min_periods=1), how)().to_numpy()


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Features for every row of `df`. df must be sorted by patient, then hour."""
    g = df.groupby("patient_id", sort=False)
    out = {"patient_id": df["patient_id"].to_numpy(), "hour": df["hour"].to_numpy()}

    # --- last value, hours since measured, measured yet: all 27 ---------------------
    # Time is ICULOS, not the row index (Step 2 decision 3). Within a patient they only
    # differ by a constant, but using ICULOS keeps "hours since" meaning real hours.
    t = df["ICULOS"].astype("float64")
    for v in KEPT:
        seen = df[v].notna()
        out[f"{v}_last"] = g[v].ffill().to_numpy()
        t_last = t.where(seen).groupby(df["patient_id"], sort=False).ffill()
        out[f"{v}_hours_since"] = (t - t_last).clip(upper=SINCE_CAP).to_numpy()
        out[f"{v}_measured_yet"] = seen.groupby(df["patient_id"], sort=False).cummax().to_numpy().astype("int8")

    # --- trailing windows -------------------------------------------------------------
    pad = max(WINDOWS)
    lab_arr, pos = _pad(df, LABS, pad)
    counts = _rolling(~np.isnan(lab_arr) * 1.0, 24, "sum")[pos]
    for j, v in enumerate(LABS):
        out[f"{v}_count_24h"] = counts[:, j]

    vit, pos = _pad(df, VITALS, pad)
    tt, _ = _pad(df, ["ICULOS"], pad)
    tt = np.repeat(tt, len(VITALS), axis=1)
    m = ~np.isnan(vit)
    x, tm = np.nan_to_num(vit), np.where(m, np.nan_to_num(tt), 0.0)
    for w in WINDOWS:
        mean, mn, mx = (_rolling(vit, w, h)[pos] for h in ("mean", "min", "max"))
        # Least-squares slope from rolling sums over MEASURED points only:
        #   slope = (n*Stx - St*Sx) / (n*Stt - St^2)
        # Undefined (NaN) with fewer than 2 points, or when they share one time.
        n = _rolling(m * 1.0, w, "sum")[pos]
        St, Sx = _rolling(tm, w, "sum")[pos], _rolling(x * m, w, "sum")[pos]
        Stx, Stt = _rolling(tm * x, w, "sum")[pos], _rolling(tm * tm, w, "sum")[pos]
        den = n * Stt - St ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = np.where((n >= 2) & (den > 1e-9), (n * Stx - St * Sx) / den, np.nan)
        for j, v in enumerate(VITALS):
            out[f"{v}_mean_{w}h"], out[f"{v}_min_{w}h"] = mean[:, j], mn[:, j]
            out[f"{v}_max_{w}h"], out[f"{v}_slope_{w}h"] = mx[:, j], slope[:, j]

    # --- context ------------------------------------------------------------------------
    out["Age"] = df["Age"].to_numpy(dtype="float64")
    out["Gender"] = df["Gender"].astype("float64").to_numpy()
    out["ICULOS"] = t.to_numpy()
    out["HospAdmTime"] = df["HospAdmTime"].to_numpy(dtype="float64")
    out["icu_medical"] = (df["Unit1"] == 1).fillna(False).to_numpy().astype("int8")
    out["icu_surgical"] = (df["Unit2"] == 1).fillna(False).to_numpy().astype("int8")
    out["icu_unknown"] = df["Unit1"].isna().to_numpy().astype("int8")
    return pd.DataFrame(out)


def truncation_test(df: pd.DataFrame, feats: pd.DataFrame, rng: np.random.Generator) -> int:
    """Features at hour k must not change when every row after k is deleted."""
    pids = rng.choice(df["patient_id"].unique(), size=TEST_PATIENTS, replace=False)
    sub = df[df["patient_id"].isin(pids)]
    cut = sub.groupby("patient_id")["hour"].transform(lambda h: rng.integers(0, h.max() + 1))
    trunc = build(sub[sub["hour"] <= cut].reset_index(drop=True))
    last = trunc.groupby("patient_id").tail(1).set_index(["patient_id", "hour"])
    full = feats.set_index(["patient_id", "hour"]).loc[last.index]
    pd.testing.assert_frame_equal(last, full, check_dtype=False)
    return len(last)


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(20260926)
    for name, src, keep in [("a", "a_hourly.parquet", ["split", data.LABEL]), ("b", "b_inputs.parquet", [])]:
        df = pd.read_parquet(PROC / src).sort_values(["patient_id", "hour"]).reset_index(drop=True)
        feats = build(df)
        n = truncation_test(df, feats, rng)
        print(f"hospital {name.upper()}: truncation test passed on {n} patients cut at random hours")
        for c in keep + ["scored"]:
            feats[c] = df[c].to_numpy()
        feats = feats[feats["scored"]].drop(columns="scored").reset_index(drop=True)
        feats.to_parquet(PROC / f"features_{name}.parquet", index=False)
        n_feat = feats.shape[1] - 2 - len(keep)
        print(f"  {len(feats):,} scored hours x {n_feat} features -> features_{name}.parquet")
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
