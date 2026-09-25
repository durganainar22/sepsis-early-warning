"""Loaders for the combined hourly table - with the hospital B rule built in.

PROJECT_PLAN.md: hospital B is the locked external test set, and its labels are not read
before Step 6. That rule is enforced HERE, in code, rather than left to discipline: the
only way to get B before Step 6 is load_b_inputs(), which drops SepsisLabel before
returning. A notebook cell cannot peek at B's sepsis rate by accident.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HOURLY = ROOT / "data" / "interim" / "hourly.parquet"
LABEL = "SepsisLabel"

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
        "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
        "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets"]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
MEASUREMENTS = VITALS + LABS


def load_a(columns: list[str] | None = None) -> pd.DataFrame:
    """Hospital A, everything including the label. The development hospital."""
    df = pd.read_parquet(HOURLY, columns=columns, filters=[("hospital", "==", "A")])
    return df.reset_index(drop=True)


def load_b_inputs(columns: list[str] | None = None) -> pd.DataFrame:
    """Hospital B WITHOUT its label - what a hospital knows about itself before go-live."""
    if columns is not None and LABEL in columns:
        raise ValueError("hospital B labels are locked until Step 6 (PROJECT_PLAN.md)")
    df = pd.read_parquet(HOURLY, columns=columns, filters=[("hospital", "==", "B")])
    return df.drop(columns=[LABEL], errors="ignore").reset_index(drop=True)
