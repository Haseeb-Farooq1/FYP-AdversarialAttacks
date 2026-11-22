# LightGBM intrusion detection pipeline

This project prepares and trains a LightGBM binary classifier using the single dataset in `Raw_DATASET` (NF-UNSW-NB15). The entry point, `main.py`, processes only this dataset.

## Data and preprocessing

Place `NF-UNSW-NB15-v3.csv` in `Raw_DATASET/`. The preprocessing pipeline divides rows by `FLOW_START_MILLISECONDS` into chronological train (about 70%), validation (about 15%), and test (about 15%) splits. Rows with equal timestamps stay together, so the final proportions can vary slightly.

The pipeline removes IP addresses, flow start/end timestamps, and the original target fields from the model features. Invalid or missing numeric feature values are replaced with zero. The binary `Label` target is stored as `label`; `Attack` is retained as `type` for reporting, and flow start time is retained as `__time` for chronological cross-validation. Prepared CSVs are written to `data_split/Raw_DATASET/{train,validation,test}/`.

## Model training and evaluation

The model compares three LightGBM configurations using five expanding-window chronological folds on the training split. It selects the configuration with the highest mean balanced accuracy, then fits the final model on train plus validation data. The later test split is evaluated once. The model uses class weights to account for class imbalance.

The test metrics include accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, false-positive rate, false-negative rate, and a confusion matrix. The matrix is printed in the terminal and stored in `model_results/Raw_DATASET/metrics.json` under `test_confusion_matrix_labels_0_1`. The fitted model is saved as `model_results/Raw_DATASET/lightgbm_model.joblib`.

## Run

From PowerShell in the project directory, with the virtual environment active:

```powershell
python main.py
```

Or run the virtual environment's interpreter directly:

```powershell
.venv/Scripts/python.exe -B main.py
```
