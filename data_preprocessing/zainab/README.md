# Zainab's compact preprocessing baseline

This implementation processes the seven exact filenames `Network_dataset_17.csv`
through `Network_dataset_22.csv` and `Network_dataset_23(in).csv`. It preserves
every row and produces **23 model features**. Haseeb's implementation is unchanged.

## Run from the repository root

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r data_preprocessing/zainab/requirements.txt
.venv/Scripts/python.exe -B -m unittest data_preprocessing.zainab.test_preprocess -v
.venv/Scripts/python.exe -B -m data_preprocessing.zainab.preprocess
```

The default seed is 42 and the chunk size is 50,000. Both are CLI options.
Existing output files are never overwritten; use fresh `--output-dir` and
`--artifact-dir` paths for a new experiment. Keep large custom output directories
ignored by Git too. No raw CSV is modified, and this program never commits or pushes.

## Four simple passes

1. Verify the exact 46-column headers and binary/multiclass target consistency.
   Read attack types into a compact integer array. Shuffle indices within each
   type using seed 42, allocating 70% to training, 15% to validation, and the
   remainder to test. Train/validation counts are rounded down per type. The
   seed, exact filenames/order, and input hashes make the split reproducible.
2. Read chunks and select **only training rows** to learn numerical means and
   categorical vocabularies. Strip whitespace; treat `-` and blank as missing.
   Convert `src_bytes="0.0.0.0"` to missing without dropping the record. Reject
   other invalid numeric values, infinities, or unexpected Boolean values.
3. Fill numerical missing values using the training means and incrementally
   fit `StandardScaler` on **training rows only**. There is no log transform.
4. Apply the frozen transformations and write each split incrementally. Check
   finite outputs, record distributions/unknown categories, calculate balanced
   binary class weights from training counts, and verify raw content hashes.

This keeps only chunks and small row-code arrays in memory, rather than loading
all seven datasets or concatenating full DataFrames. Four reads keep imputation
and scaling separate and easy to explain. No SMOTE or other resampling is used.

## Features and codes

Numerical (12): `src_port`, `dst_port`, `duration`, `src_bytes`, `dst_bytes`,
`missed_bytes`, `src_pkts`, `src_ip_bytes`, `dst_pkts`, `dst_ip_bytes`,
`http_request_body_len`, `http_response_body_len`.

Categorical (7): `proto`, `service`, `conn_state`, `dns_qclass`, `dns_qtype`,
`dns_rcode`, `http_status_code`. Each gets **one column**: missing = 0,
unknown = 1, and training-observed strings get sorted deterministic codes 2+.
All training categories are retained; there is no rare-category grouping,
hashing, or one-hot encoding. Each mapping is saved in the config.

Boolean (4): `dns_AA`, `dns_RD`, `dns_RA`, `dns_rejected`: F = 0, T = 1,
missing = 2. Categorical and Boolean codes are **not scaled**. A literal zero
in a protocol-code field is a real category, distinct from the missing code.

`label` is the binary target and `type` is reporting metadata. Neither is in X.
The remaining 21 original predictors are excluded for the documented reasons
in `DROPPED_COLUMNS` and the generated report. There are no derived features
or extra missing indicators. Numerical imputation can produce a noninteger
mean; retain original discrete constraints for later adversarial evaluation.

## Outputs and later use

- `dataset/processed_zainab/train.csv`
- `dataset/processed_zainab/validation.csv`
- `dataset/processed_zainab/test.csv`
- `dataset/processed_zainab/split_assignments.npy` (one byte per source record;
  0 = train, 1 = validation, 2 = test, in report manifest order)
- `data_preprocessing/zainab/artifacts/preprocessing_config.json`
- `data_preprocessing/zainab/artifacts/preprocessing_report.json`

Each CSV has **25 columns: 23 inputs plus `label` and `type`**. Use the persisted
`feature_columns` list to select X, not every numerical column in the CSV.
Rows retain their source order within each split; shuffle training batches when
training a model rather than assuming the exported rows are already shuffled.
`CompactPreprocessor.from_config(config)` restores the frozen transform;
call `clean_features` before its `transform` method on new records.
The balanced training weight for class c is N_train / (2 * N_train,c).

The processed-data directory (including the assignment array) is Git-ignored.
Only the small config/report artifacts are intended for review alongside code.

Category codes have no numerical ordering or distance meaning. A later model
must consume them as categorical inputs, for example through separate embeddings;
feeding all 23 columns into ordinary dense layers would introduce artificial
ordinal relationships. Never continuously perturb these category/Boolean codes.
Ports also need protocol-aware discrete constraints in adversarial evaluation.

This requested baseline uses a row-stratified split and retains duplicates.
Exact duplicates and related traffic may cross splits; it is not a claim of
session-independent evaluation. There is no synthetic data and validation/test
class proportions are not rebalanced.
