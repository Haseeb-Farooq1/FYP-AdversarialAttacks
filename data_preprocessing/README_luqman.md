# Preprocessing - Luqman's portion (Network_dataset_9.csv - 16.csv)

## Setup
```
pip install -r requirements_luqman.txt
```

## Folder layout expected
```
data/
  raw/
    Network_dataset_9.csv
    Network_dataset_10.csv
    ...
    Network_dataset_16.csv
```
Download these from the UNSW ToN_IoT page and place them under `data/raw/`.
This folder is git-ignored, do not commit the raw CSVs.

## Run
```
python preprocessing_luqman.py --data_dir ./data/raw --out_dir ./data/processed
```

## What it does
1. Loads and combines the 8 assigned files
2. Removes duplicate rows (known issue in this dataset)
3. Drops `src_ip`, `dst_ip`, `ts` by default (identifier/leakage risk,
   rerun with `--keep_ip_ts` if you want to study that effect specifically)
4. Fills missing values (`-` in the raw files -> 0 for numeric fields,
   `"none"` for categorical fields)
5. Label-encodes all categorical columns and both `label` (binary) and
   `type` (9 attack classes, normal included)
6. Scales numeric columns with StandardScaler
7. Stratified 80/20 train/test split (stratified on `type` so rare classes
   like MITM still appear in both sets)
8. Writes `processed_train.csv`, `processed_test.csv`, and
   `preprocessing_report_luqman.json` (row counts, class balance, missing
   value counts, output shapes) to `data/processed/`

## Comparing against Haseeb's and Zainab's pipelines
The report JSON is the thing to diff between the three of you, same keys
across all three reports (`raw_combined_shape`, `attack_type_distribution`,
`train_shape`/`test_shape`, etc.) make it easy to line up side by side when
you pick the best pipeline in the team meeting.
