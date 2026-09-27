# Sepsis early warning from ICU time series

Predicting sepsis up to 6 hours before clinical onset from hourly vitals and labs, using
the open **PhysioNet/CinC Challenge 2019** data (40,336 ICU patients, two hospital
systems). A recurrent neural network is benchmarked against logistic regression and
XGBoost, and every model is tested at a hospital it never saw during development.

**Status:** Steps 0–4 done (audit, cohort, causal features, logistic regression and XGBoost
baselines); Step 5, the GRU, in progress. The test hospital has not been opened. See
[`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the plan of record and every decision made so far.

## Reproduce

```bash
conda env create -f environment.yml
conda activate sepsis
python src/download_data.py     # ~40k files from PhysioNet's public S3 mirror, MD5-verified
python src/build_table.py       # -> data/interim/hourly.parquet (one row per patient-hour)
python src/make_cohort.py       # cohort, cleaning, scored hours, hospital A split
python src/build_features.py    # 164 causal per-hour features (truncation-tested)
python src/train_logreg.py      # baseline 1
python src/train_xgb.py         # baseline 2 (40 configs x 3 seeds, ~2.5 h on 6 CPU cores)
python src/build_sequences.py   # per-hour input channels for the GRU
```

## Data

Reyna MA, Josef CS, Jeter R, et al. *Early Prediction of Sepsis From Clinical Data: The
PhysioNet/Computing in Cardiology Challenge 2019.* Critical Care Medicine 48(2):210–217
(2020). Data: https://physionet.org/content/challenge-2019/1.0.0/ — CC BY 4.0.

## Results

Not yet run. This section is reserved for the test-hospital numbers (Step 6); validation
results so far are in `PROJECT_PLAN.md`.
