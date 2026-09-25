# data/

Everything here except this file is gitignored and regenerable.

| path | written by | contents |
|---|---|---|
| `raw/training_setA/*.psv`, `raw/training_setB/*.psv` | `src/download_data.py` | 40,336 files as shipped by PhysioNet, MD5-verified |
| `interim/hourly.parquet` | `src/build_table.py` | all files in one long table, values untouched, plus `patient_id`, `hospital`, `hour` |
