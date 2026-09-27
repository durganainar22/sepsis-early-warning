"""Step 5a - hour-by-hour input sequences for the GRU.

The GRU reads each patient's stay one hour at a time, so it gets the RAW series, not
Step 3's hand-built history. Per hour, for each of the 27 measurements (PROJECT_PLAN.md,
Step 5 decisions):
    value   last known value, standardized with TRAIN mean/sd of measured values;
            0 before the first measurement (= the train mean, the neutral input)
    mask    1 if measured this hour
    delta   hours since last measured, log1p-scaled; "never" = log1p(72) (the Step 3 cap)
plus static/context channels: age, sex, ICU type (medical/surgical/unknown),
HospAdmTime (signed log), and log1p(ICULOS).

WHERE THIS DIVERGES FROM THE TREE: standardization matters here and did not for XGBoost.
A GRU shares one set of weights across every input channel and every hour; an input
ranging 0-300 (platelets) next to one ranging 7.0-7.6 (pH) would dominate the gradient.
A tree splits on rank and never cares.

Storage is RAGGED - one flat (total_hours, channels) array plus per-patient offsets - since
stays run 8 to 336 hours; padding every patient to 336 would be ~10x the memory. Batches
are padded on the fly in train_gru.py.

Output: data/processed/seq_a.npz  (x, y, scored, offsets, patient ids, split)
        data/processed/seq_b.npz  (x, scored, offsets, patient ids - NO label)
        data/processed/seq.meta.json

Usage:  python src/build_sequences.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_features import KEPT, PROC, SINCE_CAP
import data


def channels(df: pd.DataFrame, stats: dict) -> tuple[np.ndarray, list[str]]:
    g = df.groupby("patient_id", sort=False)
    t = df["ICULOS"].astype("float64")
    cols, names = [], []
    for v in KEPT:
        seen = df[v].notna()
        last = g[v].ffill()
        cols.append(((last - stats[v][0]) / stats[v][1]).fillna(0.0).to_numpy())
        cols.append(seen.to_numpy(dtype="float64"))
        t_last = t.where(seen).groupby(df["patient_id"], sort=False).ffill()
        cols.append(np.log1p((t - t_last).clip(upper=SINCE_CAP).fillna(SINCE_CAP)).to_numpy())
        names += [f"{v}_value", f"{v}_mask", f"{v}_delta"]
    age = (df["Age"].astype("float64") - stats["Age"][0]) / stats["Age"][1]
    # HospAdmTime is missing for 8 hours of one hospital-A patient (notebook 01 rounded this
    # to 0.0%). Train median, the same fill the logistic regression uses.
    adm = df["HospAdmTime"].astype("float64").fillna(stats["HospAdmTime_median"])
    cols += [age.to_numpy(), df["Gender"].astype("float64").to_numpy(),
             (df["Unit1"] == 1).fillna(False).to_numpy(dtype="float64"),
             (df["Unit2"] == 1).fillna(False).to_numpy(dtype="float64"),
             df["Unit1"].isna().to_numpy(dtype="float64"),
             (np.sign(adm) * np.log1p(adm.abs())).to_numpy(),
             np.log1p(t).to_numpy()]
    names += ["age", "gender", "icu_medical", "icu_surgical", "icu_unknown", "hosp_adm_slog", "log_iculos"]
    x = np.column_stack(cols).astype("float32")
    assert np.isfinite(x).all(), "non-finite input channel"
    return x, names


def offsets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pid = df["patient_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1]
    return np.r_[starts, len(df)].astype("int64"), pid[starts]


def main() -> None:
    A = pd.read_parquet(PROC / "a_hourly.parquet").sort_values(["patient_id", "hour"]).reset_index(drop=True)
    B = pd.read_parquet(PROC / "b_inputs.parquet").sort_values(["patient_id", "hour"]).reset_index(drop=True)

    tr = A[A["split"] == "train"]
    # Mean/sd over MEASURED train values - not over forward-filled ones, which would weight
    # a single lab draw by however many hours it happened to be carried forward.
    stats = {v: (float(tr[v].mean()), float(tr[v].std() or 1.0)) for v in KEPT + ["Age"]}
    stats["HospAdmTime_median"] = float(tr["HospAdmTime"].median())

    xa, names = channels(A, stats)
    off_a, pid_a = offsets(A)
    split = A.groupby("patient_id", sort=False)["split"].first().loc[pid_a].to_numpy()
    np.savez(PROC / "seq_a.npz", x=xa, y=A[data.LABEL].to_numpy(dtype="int8"),
             scored=A["scored"].to_numpy(), offsets=off_a, pid=pid_a, split=split)

    xb, _ = channels(B, stats)
    off_b, pid_b = offsets(B)
    np.savez(PROC / "seq_b.npz", x=xb, scored=B["scored"].to_numpy(), offsets=off_b, pid=pid_b)

    (PROC / "seq.meta.json").write_text(json.dumps({"channels": names, "train_stats": stats,
                                                    "since_cap": SINCE_CAP}, indent=2))
    for h, x, off in [("A", xa, off_a), ("B", xb, off_b)]:
        L = np.diff(off)
        print(f"hospital {h}: {len(L):,} patients, {len(x):,} hours x {x.shape[1]} channels "
              f"({x.nbytes / 1e6:.0f} MB), stay length {L.min()}-{L.max()} h")
    ch = pd.DataFrame(xa[np.isin(np.repeat(split, np.diff(off_a)), ["train"])], columns=names)
    vals = ch[[c for c in names if c.endswith("_value")]]
    print(f"train value channels: mean {vals.to_numpy().mean():+.3f}, sd {vals.to_numpy().std():.3f} "
          "(forward-filled, so not exactly 0/1 - expected)")


if __name__ == "__main__":
    main()
