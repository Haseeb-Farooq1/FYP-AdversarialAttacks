# Zainab's compact preprocessing baseline

## Shared-fit, separate-file baseline

`preprocess_shared.py` implements **selected raw files -> one shared configuration
-> separate original-named CSVs**. The earlier individual and split implementations
and their artifacts remain historical baselines. No model partitions, split
assignments, or training class weights are produced by the shared mode.
Here, shared means shared fitted parameters, not merged datasets: seven source
files remain seven separate processed files.

**Status:** files 17-23 completed and fully validated: 6,339,021 rows retained,
23 predictors per output, 694 invalid src_bytes values handled, and 226,119
duplicate occurrences retained. All output rows were reproduced from the saved
configuration; 14 post-generation shared tests passed. See the shared artifacts
and instructor report for measured results. The default output paths now exist;
reruns require fresh output/artifact/report paths rather than overwriting them.

```powershell
.venv/Scripts/python.exe -B -m data_preprocessing.zainab.preprocess_shared --start 17 --end 23
# Later: change only the range (all files must be present with the expected schema).
.venv/Scripts/python.exe -B -m data_preprocessing.zainab.preprocess_shared --start 1 --end 23
# Or pass explicit paths with --files path/to/file.csv path/to/another.csv
# Add --write-metadata when source_file/source_row/ts audit metadata is needed.
.venv/Scripts/python.exe -B -m unittest data_preprocessing.zainab.test_shared -v
```

The filename for 23 is `Network_dataset_23(in).csv`. Inputs must have unique
basenames and the expected 46-column schema; unsupported schema variants stop
with a diagnostic rather than being silently aligned. Outputs and artifact
directories must be fresh; existing data is never overwritten.

### Shared transformations

The exact existing 23-feature order is preserved. For all 12 numerical columns:
explicit cleaning -> one shared exact median imputation -> optional log1p ->
one shared StandardScaler. Medians use all observed finite cleaned values from
all selected files, including duplicates, written to per-column float64 disk
files. In-place memory-mapped partition selects the middle value (or average of
the two middle values). No sampling/approximate quantiles are used. An entirely
missing numerical column stops processing. Working files are removed on exit.

The ten log1p features are `duration`, `src_bytes`, `dst_bytes`, `missed_bytes`,
`src_pkts`, `src_ip_bytes`, `dst_pkts`, `dst_ip_bytes`,
`http_request_body_len`, and `http_response_body_len`. Every observed value is
checked for nonnegativity before fitting; imputed values are checked again before
log1p. Unexpected negatives/infinities or Boolean states cause clear errors.
`src_bytes="0.0.0.0"` becomes missing; rows and legitimate zeros are retained.

`src_port` and `dst_port` never receive log1p. Their numerical scaling is retained
only for controlled baseline comparability: ports are discrete network identifiers,
not physical continuous measurements. Later compare current ports, removing
`src_port`, and categorical/bucketed `dst_port`; this implementation performs no
port ablation. No additional feature selection is performed.

One shared sorted mapping per categorical feature assigns missing=0, unknown=1,
observed categories=2+. Booleans use F=0, T=1, missing=2. Codes are never scaled;
they are identifiers, not continuous measurements or continuous attack variables.
Each source uses the same fitted medians, scaler and mappings.

### Duplicate audit and memory

SQLite compares the full JSON serialization of all 46 parsed raw string values
before cleaning, including targets, timestamps and IPs. File/row identity is not
part of the key. CSV quoting differences are ignored, while whitespace and missing
marker differences remain distinct. Full keys avoid fingerprint hash-collision
ambiguity. Duplicate excess means occurrences beyond the first; the report also
records rows belonging to duplicate groups, within-file excess, cross-file groups,
and duplicate excess by attack type. Global excess attributed to a file depends
on the supplied file order. **No duplicates are removed.**

There are three chunked CSV passes (default 50,000 rows): inspection and shared
medians/vocabularies/duplicate audit; incremental shared scaler fitting; export.
Complete raw tables are never combined in RAM. Category vocabularies stay in RAM;
disk working space holds numeric observations and SQLite full-row keys/indexes.
Memory mapping uses the operating-system page cache, so resident memory is not
guaranteed to equal chunk size. Disk space and I/O can be substantial at 23 files.

### Outputs, configuration and limitations

- `dataset/processed_zainab_shared/`: seven original-named model CSVs, each with
  23 predictors plus unscaled `label` and `type` (25 columns).
- Optional `metadata.csv` with `--write-metadata`: `source_file`, `source_row`
  (one-based parsed record number, excluding the header), and original `ts`.
  Default runs do not write metadata. The existing full-run metadata file remains
  local and unchanged. Metadata never enters X.
- `data_preprocessing/zainab/artifacts/shared/`: `preprocessing_config.json`,
  `preprocessing_report.json`, `feature_metadata.json`, plus full-run validation
  evidence in `validation_report.json`. Test summaries are consolidated in
  `preprocessing_report.json`.
- `reports/zainab_shared_preprocessing_report.md`: instructor-facing report.

Large outputs/working files and instructor reports are ignored; code and small
JSON artifacts are not. Source hashes are verified after export. Failed runs may
leave incomplete output directories; they do not create a successful final report.
Use fresh paths after a failure rather than overwriting previous runs.
Source provenance uses repository-relative paths (basenames for external inputs),
with content hashes identifying inputs. Existing shared config paths were made
portable without changing fitted parameters; the corresponding digest migration
is documented in `validation_report.json`.

`SharedPreprocessor.from_config` reloads the frozen shared transform. Call
`checked_clean` before `transform`. `inverse_numerical` reverses scaling and
log1p into imputed raw units, not original missingness; it does not establish
attack validity. `decode_categories` uses reverse mappings and rejects invalid
or fractional codes. Feature metadata records observed ranges/types and uses
`review_required` for adversarial mutability; observed maxima are not asserted
as universal limits.

This shared preprocessing baseline fits transformation parameters using the selected preprocessing dataset as a whole. It is intended for preprocessing development and pipeline comparison. Before final model evaluation, raw records must first be divided into training/validation/test sets and all learned preprocessing parameters must be refitted using training data only.

For future evaluation, isolate duplicate/related flows and use identical raw
records, splits, classifier, hyperparameters and random seed across pipelines.
Compare macro-F1, balanced accuracy, per-class precision/recall/F1, confusion
matrix, normal false-positive rate and attack false-negative rate. Accuracy alone
is insufficient. Adversarial evaluation follows the clean baseline. Median/log
changes also differ from the old baseline, so an architecture-only experiment
must keep those choices fixed. No superiority is claimed and no model is trained.

The 21 dropped predictors remain excluded for the current controlled compact
baseline, not because they are universally useless. When files 1-23 are available,
revisit missingness, sparse HTTP/SSL/weird fields, per-class coverage, shortcuts
and ports using the full training partition before final evaluation.

## Previous independent-file baseline

Run `.venv/Scripts/python.exe -B -m data_preprocessing.zainab.preprocess_individual`
from the repository root. This mode writes seven files to
`dataset/processed_zainab_individual/`, using the exact original basenames:
`Network_dataset_17.csv` through `Network_dataset_22.csv`, and
`Network_dataset_23(in).csv`. The directory is Git-ignored.

Each entire input file is fitted independently, using its own imputation means,
StandardScaler and category mappings. There are still 23 features plus `label`
and `type`, with no row deletion, extra features, or train/validation/test split.
One config per input and an aggregate `preprocessing_report.json` are saved under
`data_preprocessing/zainab/artifacts/individual/`. Configs record `fitted_rows`
and `fit_scope=entire_individual_file`, rather than claiming training-only fitting.
They can be reloaded with `CompactPreprocessor.from_config`.

Category codes and scales can differ between these files; do not concatenate
their encoded features as if they used a shared transformation. For a later
held-out model evaluation, return to raw records, split them first, and fit
preprocessing only on training data. This mode does not compute training class
weights. The earlier split outputs, configs, and instructor reports remain as
records of the previous run; they do not describe this new independent run.

Tests: `.venv/Scripts/python.exe -B -m unittest data_preprocessing.zainab.test_individual -v`

## Previous train/validation/test workflow

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
