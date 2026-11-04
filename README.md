
This repository contains one preprocessing pipeline. `main.py` runs the shared Zainab transformation across datasets 1-23.

## Folders and files

- `Raw_DATASET/`: source CSV files.
- `pre_processed/`: one processed CSV per source file.
- `main.py`: pipeline entry point.
- `pre_processing/preprocess_shared.py`: shared fit and CSV export.
- `pre_processing/preprocess.py`: schema and cleaning helpers used by the pipeline.
- `requirements.txt`: Python dependencies.

The run also writes `preprocessing_config.json`, `preprocessing_report.json`, `feature_metadata.json`, and `preprocessing_summary.md` in `pre_processing/`. The outputs keep each source basename and row count. Dataset 23 may be named `Network_dataset_23.csv` or `Network_dataset_23(in).csv`; dataset 6 may contain an extra `uid` column, which is excluded.

## Run

Install dependencies into the existing environment if needed, then run from the repository root:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -B main.py
```

The preprocessor refuses to overwrite existing outputs. Move or rename the current output and report files before starting another run. This pipeline preprocesses data only; it does not train or evaluate a model.
