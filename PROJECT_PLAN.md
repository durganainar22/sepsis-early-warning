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
| 2 | Cohort, labels, prediction times, splits | done — `src/make_cohort.py` |
| 3 | Features for the tabular models | done — `src/build_features.py` (164 features, causal by test) |
| 4 | Baselines: logistic regression, XGBoost | done 2026-09-27 — `src/train_logreg.py`, `src/train_xgb.py` |
| 5 | Deep learning: GRU on the hourly sequence | next — inputs and script built, timing pending |
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

**Within hospital A — agreed 2026-09-26:** 70 / 15 / 15 train / validation / internal test,
split by patient, stratified on whether the patient ever develops sepsis, fixed seed.
After the Step 2 cohort rule this gives 13,950 / 2,990 / 2,990 patients with 968 / 208 / 208
septic (~2% of scored hours positive in each). 208 septic patients is on the small side for
a stable PR-AUC, which is one more reason the paired bootstrap in Step 6 is not optional. Hospital B is not split: all of it is the external test.

**Rule, in force from now:** no analysis in Steps 1–5 reads hospital B's labels. Step 1
may describe B's *inputs* (missingness, ranges) at a summary level, because that is
what a deploying hospital would know before go-live; B's sepsis rate and outcomes stay
unread until Step 6.

## Step 2 decisions — agreed 2026-09-26

Each traces to a numbered finding at the end of `notebooks/01_data_audit.ipynb`.

1. **Score from hour 6.** Predictions are only scored from each patient's 7th row onward
   (row index ≥ 6, i.e. after 6 hours of data). Patients whose label switches on before
   row 6 are excluded — 406 patients in hospital A. One rule for everyone: the model is
   never judged on a patient it has had no time to observe, and the benchmark measures
   early *warning*, not detection of sepsis that is already present. Every record has at
   least 8 rows, so no non-septic patient loses all scored hours.
   *Consequence for hospital B:* this rule reads labels, so it is applied to B only
   inside the Step 6 benchmark, never before.
   Rejected: excluding only first-row cases (still rewards near-present detection);
   keeping everyone (the official challenge setting — most comparable, least honest).
2. **Causal features only.** Every feature at hour *t* uses data from hours ≤ *t*. No
   whole-stay summaries: septic records end 9 h after the label switches on, so record
   length alone would leak the outcome.
3. **Time in ICU comes from `ICULOS`,** never the row index (they differ for ~37% of
   patients).
4. **Missing values are information, not noise:** carry the last value forward, plus
   "hours since last measured" and "measured yet" per variable. No mean imputation.
   Lactate and FiO2 are measured ~1.9× as often before sepsis; imputation would erase that.
5. **Drop the 7 variables recorded in < 1% of hours at either hospital** — EtCO2,
   BaseExcess, HCO3, Chloride, Bilirubin_direct, TroponinI, Fibrinogen — before any model
   sees them. 27 measurements remain. Uses hospital B's *input* rates only, which a
   deploying hospital would know before go-live. Rejected: keeping all 34 (the external
   test would partly measure a mistake any deploying team would have avoided).
6. **Impossible values → missing,** by the documented limits in `src/make_cohort.py`.
   MAP/SBP/DBP disagreement (~1% of hours) is left as recorded: it is an effect of hourly
   summarising (arterial line vs cuff), not an error.
7. **PR-AUC on scored hourly predictions decides the comparison.** The challenge's
   normalised utility score is reported alongside, with its alarm threshold tuned on
   hospital A validation only. Rejected: utility as the decider (threshold-dependent, so
   the ranking could flip with the cutoff).

## Step 3 decisions — agreed 2026-09-26

- **Trend windows: 6 h and 24 h.** 6 h matches the warning horizon; 24 h catches slower
  drifts (e.g. the steady heart-rate climb in notebook 01 §5).
- **ICU type kept with an explicit "unknown" level** (medical / surgical / unknown).
  Missing for ~47% of patients; missingness may itself carry meaning, and nothing is
  invented.

## Step 4 decisions — agreed 2026-09-26

- **No class reweighting.** Train on the natural ~2% positive hours; the alarm threshold
  for the utility score is tuned on hospital A validation. Same as the readmission
  project, applied identically to all three models including the GRU, so the benchmark
  compares architectures rather than imbalance handling, and probabilities stay
  calibrated.
- **XGBoost search: 40 configurations × 3 seeds**, selected on mean val PR-AUC, winner
  re-run over 5 seeds. The same protocol the readmission project adopted after measuring
  the winner's curse of single-seed selection. ~2.5 h on 6 CPU cores. The GRU gets a
  comparable budget in Step 5.
- **Rows within a patient are not independent.** Every uncertainty estimate (Step 6
  bootstrap) resamples *patients*, never hours.

## Step 4 results — 2026-09-27 (hospital A validation only)

Base rate 1.99% of scored hours (101,524 val hours, 2,990 patients).

| Model | PR-AUC (decides) | ROC-AUC | Brier | Utility | Alarms / precision / recall |
|---|---|---|---|---|---|
| Logistic regression, L2 C=0.01 | 0.1126 | 0.834 | 0.0186 | 0.433 | 18.9% / 7.1% / 67.5% |
| XGBoost, 5 seeds | **0.1485 ± 0.0068** | 0.861 ± 0.002 | 0.0181 | **0.481 ± 0.005** | 17.1% / 8.2% / 69.8% |

- **XGBoost leads by +0.036 PR-AUC, ~5× its seed sd**, and wins on every other metric too —
  not a tie, unlike the readmission project. Logreg's PR-AUC is flat across its C grid; the
  winning trees are shallow (depth 3–4), so the gain comes from interactions a linear fit
  cannot express. Still validation — the paired bootstrap on test decides in Step 6.
- **Search: 40 configs × 3 seeds, 156 min.** Winner config 38: depth 4, lr 0.039, subsample
  0.75, colsample 0.92, min_child_weight 2.0, lambda 0.79, 101–261 trees. Its 3-seed search
  score (0.1499) held over 5 seeds (0.1485): no measurable winner's curse. The top three
  configs were within 0.001 of each other.
- **PR-AUC varies across seeds (0.139–0.156) but utility does not (0.474–0.487)** — the seed
  noise sits in parts of the ranking the alarm threshold never uses.
- **Early stopping is jittery:** some low-learning-rate configs stopped at 2–6 trees on one
  seed, because val aucpr peaks by chance in the first rounds and holds for 50. Noted, not
  changed mid-search; a minimum-rounds floor would be a protocol change.
- **Feature gain — carry into Step 7:** `ICULOS` leads by ~3× (0.077 vs 0.027). ICU time is
  a real risk factor, but septic records also end 9 h after onset (notebook 01), so part of
  this may be a dataset-construction artifact; check performance within ICULOS bands. Most
  of the rest are measurement-process features (`FiO2_hours_since`, `SBP_measured_yet`,
  `*_hours_since`), as with logreg's top coefficients: the models partly learn clinician
  behaviour, not only physiology.

## Step 5 decisions — agreed 2026-09-26

- **GRU input per hour: value + mask + time-since** for each of the 27 measurements — the
  last known value (standardized on train; 0 = the train mean before the first
  measurement), a measured-this-hour flag, and hours since last measured — plus the static
  context. The same missingness signal XGBoost gets, but raw: the network must learn its
  own trends instead of being handed them. Rejected: GRU-D (learned decay — more code,
  harder to tune on CPU, and a different question), and zero-filled values without masks
  (discards the signal notebook 01 showed matters).
- **Search budget matches XGBoost: 40 configurations × 3 seeds**, winner re-run over 5
  seeds. One run is timed first; if the full search is far above XGBoost's ~2.5 h, that is
  reported before running rather than silently cut.
- **No hand-built features go into the GRU.** A GRU fed XGBoost's 164 features would blur
  the one comparison the project exists to make.
