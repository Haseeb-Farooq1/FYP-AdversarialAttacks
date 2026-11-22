# LightGBM IoT intrusion detection pipeline

`main.py` rebuilds splits and evaluates both datasets:

- `Raw_DATASET/NF-UNSW-NB15-v3.csv`: split by `FLOW_START_MILLISECONDS` into earlier training, validation, and later test periods.
- `Raw_DATASET2/Network_dataset_1.csv` through `Network_dataset_23.csv`: the files are consecutive time chunks; chronological timestamp boundaries assign records to train, validation, or test.

Prepared data is written under `data_split/Raw_DATASET/` and `data_split/Raw_DATASET2/`. Imputation, categorical mappings, and scaling are fitted on training data only. Equal timestamps and exact duplicate records stay on the same side of each split boundary. The model excludes target and descriptive attack labels, IP addresses, timestamps, and internal time metadata. Raw attack labels are retained only for reporting.

For each dataset, the model compares three regularized LightGBM configurations using five expanding-window chronological folds over the training period. The configuration with the highest mean balanced accuracy is selected. Training and validation are then combined for the final fit, and the later test period is evaluated once. Class weighting addresses imbalance without resampling the validation or test sets. Output includes accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR, and the confusion matrix.

The two datasets are trained and evaluated independently. Each gets its own model and `metrics.json` under `model_results/Raw_DATASET/` or `model_results/Raw_DATASET2/`.

Run both independent pipelines from PowerShell:

```powershell
.venv/Scripts/python.exe -B main.py
```

Run one dataset by itself:

```powershell
.venv/Scripts/python.exe -B main.py --dataset Raw_DATASET
.venv/Scripts/python.exe -B main.py --dataset Raw_DATASET2
```
