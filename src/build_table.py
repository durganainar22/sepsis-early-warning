"""Combine the 40,336 per-patient .psv files into ONE long table.

Mechanical on purpose: no cleaning, no imputation, no filtering, no renaming. Every value
is exactly what PhysioNet shipped, NaN included. All decisions about the data belong to
later steps, where they can be argued for - a loader that quietly "fixes" things is how a
pipeline ends up with decisions nobody remembers making.

Adds only three columns that are facts about the FILE, not the patient's values:
    patient_id   file stem, e.g. "p000001"
    hospital     "A" or "B" - the two training folders are two hospital systems
    hour         0-based row index within the patient's file

Output: data/interim/hourly.parquet  (one row per patient-hour)

Usage:  python src/build_table.py
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "interim" / "hourly.parquet"
SETS = {"training_setA": "A", "training_setB": "B"}

# Demographic / admin columns stored as they are, the rest are float measurements.
# float32 halves memory for ~1.5M rows x 40 columns and loses nothing: the source files
# carry at most 2-3 decimals.
INT_LIKE = ["Gender", "Unit1", "Unit2", "ICULOS", "SepsisLabel"]


def read_one(path: Path, hospital: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|")
    df.insert(0, "hour", np.arange(len(df), dtype=np.int16))
    df.insert(0, "hospital", hospital)
    df.insert(0, "patient_id", path.stem)
    return df


def main() -> None:
    t0 = time.perf_counter()
    jobs = [(p, h) for folder, h in SETS.items() for p in sorted((RAW / folder).glob("*.psv"))]
    print(f"reading {len(jobs):,} files ...", flush=True)
    with ThreadPoolExecutor(8) as pool:
        frames = list(pool.map(lambda j: read_one(*j), jobs))

    cols = frames[0].columns
    # Every file must share one schema, or concat would silently NaN-fill a missing column.
    assert all((f.columns == cols).all() for f in frames), "files disagree on columns"
    df = pd.concat(frames, ignore_index=True)

    meas = [c for c in cols if c not in ("patient_id", "hospital", "hour", *INT_LIKE)]
    df[meas] = df[meas].astype("float32")
    # Nullable Int so a stray NaN in these would survive visibly instead of crashing or
    # being cast to a sentinel. Step 1 checks whether any actually occur.
    for c in INT_LIKE:
        df[c] = df[c].astype("Int16")
    df["hospital"] = df["hospital"].astype("category")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    per = df.groupby("hospital", observed=True)["patient_id"].nunique()
    print(f"{len(df):,} patient-hours, {df['patient_id'].nunique():,} patients "
          f"({', '.join(f'{h}: {n:,}' for h, n in per.items())}), {df.shape[1]} columns")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB) "
          f"in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
