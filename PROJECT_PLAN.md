# Project Plan

Early warning for sepsis in ICU patients from hourly vitals and labs (PhysioNet/CinC
Challenge 2019), with a recurrent neural network benchmarked honestly against logistic
regression and XGBoost - and tested at a hospital the models never saw.

This file is the **plan of record**. Decisions are recorded when they are made, with the
reasoning, including any that are later reversed.

## Why this project

The previous project (`diabetes-readmission`) ended in a pre-registered tie: on static,
administrative data, entity embeddings did not beat regularized logistic regression. The
natural follow-up question is whether deep learning earns its place when the data has
**temporal structure** - which is exactly what a sequence model can use and a
per-row tabular model cannot. This project asks that question under the same rules.

## Working order

| # | Step | Status |
|---|---|---|
| 0 | Setup: environment, download, one combined table | done — `src/download_data.py`, `src/build_table.py` |
| 1 | Data audit and exploration | done — `notebooks/01_data_audit.ipynb` |
| 2 | Cohort, labels, prediction times, splits | next — starts with the 7 decisions listed at the end of notebook 01 |
| 3 | Features for the tabular models | |
| 4 | Baselines: logistic regression, XGBoost | |
| 5 | Deep learning: GRU on the hourly sequence | |
| 6 | Honest benchmark — the only look at the test hospital | |
| 7 | Interpretation, figures, write-up | |

## Working agreement

- One step at a time, in order. No end-to-end code dumps.
- Before writing code for a step: explain the approach, surface any design choice with
  more than one reasonable option, and ask rather than picking silently.
- After each step runs: plain-language summary of what happened and what the numbers mean.
- Comments explain *why*, and call out where a sequence-model choice diverges from a
  tabular-model habit.
- If the benchmark shows deep learning is not the right tool here either, say so plainly.

---

# Decisions

## Data — agreed 2026-09-25

**PhysioNet/CinC Challenge 2019 training data, v1.0.0** — 40,336 ICU patients, one row per
patient-hour, 40 variables (8 vitals, 26 labs, 6 demographic/admin) plus `SepsisLabel`.
CC BY 4.0, open access. Chosen over MIMIC-IV because it needs no credentialing and its
licence allows working with the rows directly.

`SepsisLabel` is 1 from **6 hours before** clinical sepsis onset (Sepsis-3) onward, and
always 0 for patients who never develop sepsis. The task is therefore early warning, not
diagnosis.

## Prediction task — agreed 2026-09-25

**Hourly prediction.** At every hour of a patient's stay, using only data up to and
including that hour, predict `SepsisLabel`. This is how an early-warning system is
actually deployed, it matches the challenge's labels and utility score, and it is the
framing in which a sequence model has a structural reason to help.

Rejected: one prediction per patient from a fixed window (e.g. first 24 h). Simpler, but
it discards the timing that makes this dataset worth using.

## Split — agreed 2026-09-25

**Build and tune everything on hospital A (`training_setA`); hospital B
(`training_setB`) is the locked test set, opened once, in Step 6.** An internal held-out
slice of A is also scored, so the report can separate "how good is the model" from "how
much does it lose at a new hospital".

This is external validation, and it is the harder, more honest question: clinical models
routinely lose performance when moved between hospitals (different patients, different
lab-ordering habits, different charting). A random split mixing both hospitals would
hide that. Exact within-A proportions are a Step 2 decision.

**Rule, in force from now:** no analysis in Steps 1–5 reads hospital B's labels. Step 1
may describe B's *inputs* (missingness, ranges) at a summary level, because that is
what a deploying hospital would know before go-live; B's sepsis rate and outcomes stay
unread until Step 6.
