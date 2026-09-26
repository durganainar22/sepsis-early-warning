"""Step 2 - cohort, cleaning, scored hours and the hospital A split.

Every decision here is recorded in PROJECT_PLAN.md ("Step 2 decisions") and traces to a
finding in notebooks/01_data_audit.ipynb. This script implements them; it decides nothing.

    hospital A  ->  data/processed/a_hourly.parquet   cleaned, cohort applied, with split
                                                     and a `scored` flag per row
    hospital B  ->  data/processed/b_inputs.parquet  cleaned the SAME way, NO label, and
                                                     NO cohort exclusion - that rule reads
                                                     labels, so B gets it in Step 6 only

Cleaning is identical for both hospitals and uses no labels, so applying it to B now
leaks nothing. The split and the cohort rule use labels, so they touch A only.

Usage:  python src/make_cohort.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
SEED = 20260926

# Decision 5: recorded in < 1% of hours at one of the two hospitals (notebook 01 §3).
DROPPED = ["EtCO2", "BaseExcess", "HCO3", "Chloride", "Bilirubin_direct", "TroponinI", "Fibrinogen"]
KEPT = [v for v in data.MEASUREMENTS if v not in DROPPED]

# Decision 6: generous physiological limits. A value outside is impossible or a unit error,
# and becomes missing - the model then treats it as "not measured", which is the honest
# reading. Deliberately loose: the goal is to remove nonsense, not to second-guess the
# chart. MAP/SBP/DBP disagreement is NOT here - see PROJECT_PLAN.md.
LIMITS = {
    "Temp": (25.0, 45.0),
    "FiO2": (0.0, 1.0),        # a fraction; 10 is almost certainly "10%" typed in
    "Potassium": (0.5, 12.0),
    "MAP": (0.0, 250.0),
    "HR": (0.0, 300.0),
    "SBP": (0.0, 300.0),
    "pH": (6.5, 8.0),
}

# Decision 1: predictions are scored from row index 6 onward (after 6 hours of data).
WARMUP_ROWS = 6
SPLIT_SHARES = {"train": 0.70, "val": 0.15, "test_internal": 0.15}


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Same cleaning for either hospital. Returns the frame and per-rule counts."""
    df = df.drop(columns=DROPPED)
    counts = {}
    for col, (lo, hi) in LIMITS.items():
        bad = (df[col] < lo) | (df[col] > hi)
        counts[f"{col} outside [{lo}, {hi}]"] = int(bad.sum())
        df.loc[bad, col] = np.nan
    # Diastolic at or above systolic is physically impossible in one reading. The DBP
    # value is the one dropped: DBP is the least-recorded of the three at hospital A (52%
    # of hours vs 85%+), so it is the likelier stray.
    bad = df["DBP"] >= df["SBP"]
    counts["DBP >= SBP (DBP set missing)"] = int(bad.sum())
    df.loc[bad, "DBP"] = np.nan
    return df, counts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- hospital A
    A, counts_a = clean(data.load_a())
    first = A[A[data.LABEL] == 1].groupby("patient_id")["hour"].min()
    early = first[first < WARMUP_ROWS].index
    A = A[~A["patient_id"].isin(early)].reset_index(drop=True)
    A["scored"] = A["hour"] >= WARMUP_ROWS

    pt = A.groupby("patient_id")[data.LABEL].max().rename("septic").reset_index()
    rest, test_int = train_test_split(pt, test_size=SPLIT_SHARES["test_internal"],
                                      stratify=pt["septic"], random_state=SEED)
    val_share = SPLIT_SHARES["val"] / (1 - SPLIT_SHARES["test_internal"])
    train, val = train_test_split(rest, test_size=val_share, stratify=rest["septic"],
                                  random_state=SEED)
    split = pd.concat([train.assign(split="train"), val.assign(split="val"),
                       test_int.assign(split="test_internal")])
    A = A.merge(split[["patient_id", "split"]], on="patient_id", how="left", validate="many_to_one")
    assert A["split"].notna().all()
    A.to_parquet(OUT / "a_hourly.parquet", index=False)

    # ---------------------------------------------------------------- hospital B
    B, counts_b = clean(data.load_b_inputs())
    B["scored"] = B["hour"] >= WARMUP_ROWS  # a fact about the row index, not the label
    B.to_parquet(OUT / "b_inputs.parquet", index=False)

    # ---------------------------------------------------------------- report
    print(f"Dropped {len(DROPPED)} variables; {len(KEPT)} measurements kept.")
    print(f"Cohort rule: excluded {len(early):,} hospital-A patients labelled before row {WARMUP_ROWS}.\n")
    print("Cleaning (values set to missing)   hospital A   hospital B")
    for k in counts_a:
        print(f"  {k:<34} {counts_a[k]:>8,}   {counts_b[k]:>8,}")

    s = A.groupby("split").agg(patients=("patient_id", "nunique"), hours=("hour", "size"),
                               scored_hours=("scored", "sum"),
                               scored_pos=(data.LABEL, lambda y: int(y[A.loc[y.index, "scored"]].sum())))
    s["septic_patients"] = split.groupby("split")["septic"].sum()
    s["septic_rate"] = s["septic_patients"] / s["patients"]
    s["scored_pos_rate"] = s["scored_pos"] / s["scored_hours"]
    s = s.loc[list(SPLIT_SHARES)]
    print("\nHospital A split:")
    print(s.to_string(formatters={"septic_rate": "{:.2%}".format, "scored_pos_rate": "{:.2%}".format}))
    print(f"\nHospital B: {B['patient_id'].nunique():,} patients, {int(B['scored'].sum()):,} "
          f"scored-eligible hours (label and cohort rule applied in Step 6)")

    meta = {"seed": SEED, "dropped": DROPPED, "kept": KEPT, "limits": LIMITS,
            "warmup_rows": WARMUP_ROWS, "excluded_early_septic_A": len(early),
            "split_shares": SPLIT_SHARES, "cleaning_counts": {"A": counts_a, "B": counts_b},
            "split_summary": json.loads(s.to_json(orient="index"))}
    (OUT / "cohort.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}/a_hourly.parquet, b_inputs.parquet, cohort.meta.json")


if __name__ == "__main__":
    main()
