# Sepsis early warning from ICU time series

Predicting sepsis up to 6 hours before clinical onset from hourly vitals and labs, using
the open **PhysioNet/CinC Challenge 2019** data (40,336 ICU patients, two hospital
systems). A recurrent neural network is benchmarked against logistic regression and
XGBoost, and every model is tested at a hospital it never saw during development.

**Status:** Step 0 of 7 — data downloaded and combined. See [`PROJECT_PLAN.md`](PROJECT_PLAN.md)
for the plan of record and every decision made so far.

## Reproduce

```bash
conda env create -f environment.yml
conda activate sepsis
python src/download_data.py     # ~40k files from PhysioNet's public S3 mirror, MD5-verified
python src/build_table.py       # -> data/interim/hourly.parquet (one row per patient-hour)
```

## Data

Reyna MA, Josef CS, Jeter R, et al. *Early Prediction of Sepsis From Clinical Data: The
PhysioNet/Computing in Cardiology Challenge 2019.* Critical Care Medicine 48(2):210–217
(2020). Data: https://physionet.org/content/challenge-2019/1.0.0/ — CC BY 4.0.

## Results

Not yet run.
