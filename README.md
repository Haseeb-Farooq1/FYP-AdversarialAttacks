# FYP-AdversarialAttacks
Adversarial Robustness Evaluation and Hardening of AIBased Network Intrusion Detection Systems 

## Compare preprocessed pipelines

Run the logistic-regression baseline against every `PreProcessed_*.csv` file:

```powershell
& ".venv\Scripts\python.exe" evaluate_logistic_regression.py
```

The default run evaluates up to 200,000 rows per file, uses an 80/20 stratified
test split, and writes `logistic_regression_results.json` and
`logistic_regression_results.csv`. The recommended pipeline is selected by
highest F1 score, then recall, then precision. Use `--max-rows 0` to evaluate
all rows.

The binary `label` column is the target. The `type` column is excluded because
it describes the same outcome and would leak target information into the model.
