# Zainab's shared preprocessing baseline

This shared preprocessing baseline fits transformation parameters using the selected preprocessing dataset as a whole. It is intended for preprocessing development and pipeline comparison. Before final model evaluation, raw records must first be divided into training/validation/test sets and all learned preprocessing parameters must be refitted using training data only.

## Architecture

Previous individual baseline: each file fitted its own means, scaler and mappings. This version keeps separate files but learns one shared set of exact medians, one scaler and deterministic mappings from all selected records. Feature order is unchanged (23 predictors plus label/type).

Numerical order: clean -> shared median -> log1p on ten non-port features -> shared StandardScaler. Ports use median -> scaler, without log1p. Category codes: missing=0, unknown=1, known=2+ in sorted order; Booleans: F=0, T=1, missing=2. Neither codes nor targets are scaled.

Ports are discrete network identifiers, not physical continuous measurements. Scaling is retained for baseline comparability and has limited semantic meaning. Later compare current ports, removing src_port, and categorical/bucketed dst_port.

## Row preservation

| Source/output filename | Input | Output | Retained | Invalid src_bytes |
|---|---:|---:|---:|---:|
| Network_dataset_1.csv | 1,000,000 | 1,000,000 | 100% | 175 |
| Network_dataset_2.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_3.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_4.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_5.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_6.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_7.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_8.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_9.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_10.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_11.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_12.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_13.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_14.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_15.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_16.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_17.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_18.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_19.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_20.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_21.csv | 1,000,000 | 1,000,000 | 100% | 0 |
| Network_dataset_22.csv | 1,000,000 | 1,000,000 | 100% | 690 |
| Network_dataset_23.csv | 339,021 | 339,021 | 100% | 4 |

Output missing values: 0; infinite values: 0. Runtime: 5936.405 seconds.

## Exact duplicates (retained)

{
  "definition": "Exact equality of all 46 parsed raw string fields, before cleaning, including targets/ts/IPs; CSV quoting differences ignored, whitespace preserved. Full JSON row keys in SQLite avoid hash collisions.",
  "duplicate_excess_rows": 1791971,
  "duplicate_percentage": 8.021707844761863,
  "unique_raw_records": 20547050,
  "rows_in_duplicate_groups": 2344017,
  "cross_file_duplicate_groups": 62,
  "rows_in_cross_file_groups": 358,
  "duplicate_excess_by_type": {
    "backdoor": 2731,
    "ddos": 82115,
    "dos": 1559419,
    "injection": 0,
    "mitm": 9,
    "normal": 107103,
    "password": 0,
    "ransomware": 40591,
    "scanning": 3,
    "xss": 0
  },
  "by_file_note": "Within-file excess counts local repeats; global excess is attributed to later occurrences in supplied file order. Cross-file group rows include all occurrences.",
  "removed_rows": 0,
  "retained": true
}

Per-file and per-type class distributions, missing counts and duplicate attribution are in preprocessing_report.json. Raw hashes are checked before and after processing.

## Reproducibility and memory

Numerical observations are spooled to disk by column. Exact float64 medians use in-place disk-backed partition, not sampling. SQLite compares full serialized raw rows for exact duplicates. Reads/writes are chunked; category sets remain in memory. Disk-backed work files are temporary. Shared parameters and source hashes are persisted.

## Processing notes

Duplicate rows are retained and no dataset split is performed. Category and Boolean codes are discrete identifiers. The fixed feature selection creates a compact 23-predictor baseline; it does not assert that excluded features are universally useless.

