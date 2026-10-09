# Logistic Regression baseline

Evaluate already-preprocessed numeric CSVs using one Logistic Regression model
and one duplicate-safe holdout. No scaling, imputation, encoding, feature
selection, resampling for balance, or preprocessing changes are performed.

Requires Python with NumPy, pandas and scikit-learn. Run from the repository root.

## Standardized initial preprocessing-pipeline comparison

Everyone should use this command for the initial comparison:

```sh
python model_evaluation/logistic_regression.py \
    --data-dir <processed-data-directory> \
    --sample-size 500000
```

The backslashes above are shell line continuations; in PowerShell, enter the
command on one line. Use the same procedure for every compatible input directory:

- Sample size: **500000**, sampled without replacement before splitting,
  preserving binary class proportions as closely as possible.
- Random state: **42** for sampling, splitting and the model.
- Duplicate-safe grouping by exact predictor equality, excluding `label` and `type`.
- Only the **first** `StratifiedGroupKFold` split, with `n_splits=5` and
  `shuffle=True`: one approximately 80/20 holdout, not cross-validation.
- `LogisticRegression`, with `class_weight="balanced"` and `max_iter=1000`.
- **No additional preprocessing** or parameter tuning.
- The same complete set of metrics listed below for every experiment.

Use the same library versions when comparing runs, since other model parameters
use library defaults. Versions and parameters are recorded in each local result.

## Other commands

Full-data mode remains available, but **is not the standardized initial comparison**:

```sh
python model_evaluation/logistic_regression.py --data-dir <processed-data-directory> --full-data
python -m unittest model_evaluation/test_logistic_regression.py
```

Exactly one of `--sample-size` and `--full-data` is required. Full mode can need
several GiB of free RAM: the numeric matrix, exact grouping, train/test indexing
and solver require additional allocations. Sampling is never an automatic fallback.

## Inputs and loading

Select top-level `.csv` files (case-insensitive extension) in filename order,
excluding these exact names case-insensitively: `metadata.csv`, `audit.csv`,
`validation_report.csv`, `preprocessing_report.csv`, `split_assignments.csv`.
The selected filenames are printed. Other invalid CSVs cause an error, not silent
exclusion. Keep unrelated CSVs outside the supplied directory.

Each input requires `label` (0=Normal, 1=Attack). Optional `type` is excluded.
All remaining columns are predictors and must have identical names/order across
files, be numeric and finite. Both labels must exist across the selected inputs.
Duplicate headers, empty files and invalid inputs are rejected without repair.
Inputs must remain unchanged while the script runs.

A first chunked pass validates every source row and retains compact labels.
Sample mode chooses rows without replacement using seed 42 and nearest feasible
binary-class quotas. It preserves proportions, not equal class counts. A second
chunked pass loads selected rows into a single float64 matrix. No sampled or
split datasets are saved.

## Single holdout and model

Exact numeric predictor-row equality determines groups (`numpy.unique`, not
hash equality). `label` and `type` do not determine groups. Duplicates and
conflicting labels are retained. Only the **first split** from
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` is used.
This is one approximately 80/20 holdout, not cross-validation. Actual sizes,
class distributions and deviations are reported. Group overlap must be zero;
both classes must appear in each partition.

Model: `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)`.
Other parameters retain installed scikit-learn defaults and are recorded in the
report. Balanced class weights are computed by the model from training labels.
No tuning or model saving occurs. Convergence warnings and iterations are recorded.

## Results

`model_evaluation/results/logistic_regression_results.json` is overwritten by a
successful run. It includes provenance, row/class counts, duplicate statistics,
split proportions, parameters, library versions and convergence information.
Generated JSON files in this results directory are Git-ignored and remain local.
The evaluator creates the directory automatically if it does not exist.
Metrics: accuracy, balanced accuracy, attack precision/recall/F1, macro
precision/recall/F1, confusion matrix, normal recall/specificity, attack
recall/sensitivity, attack false-negative rate and normal false-positive rate.
Undefined precision/F1 uses zero. Confusion matrix order is `[0, 1]`, true labels
on rows and predictions on columns. Emphasize macro F1, balanced accuracy and
both class recalls, rather than accuracy alone.

Timings cover discovery/validation/loading (both passes), sample selection,
exact grouping/splitting, training, prediction and total evaluation runtime.
Total ends after metrics are calculated and excludes JSON writing/console output.

## Limits

Integer category codes are consumed directly as numerical quantities, imposing
artificial ordering/distances for Logistic Regression. Upstream transformations
fitted on all source files before this holdout make this a development baseline,
**not final leakage-free model evaluation**. Final experiments must split raw
records first and fit preprocessing using training data only.
Scores measure clean classification, not adversarial robustness. Different
datasets and samples can have different difficulty; results alone cannot establish
that one preprocessing pipeline is best.
