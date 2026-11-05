# Network intrusion detection model baseline

`main.py` trains and evaluates the current LightGBM binary-classification baseline using the prepared CSVs in `pre_processed/`. The positive class is `label = 1` (attack); the textual `type` column is deliberately excluded from predictors to prevent target leakage.

For each CSV, the pipeline makes a reproducible stratified 80/20 train/holdout split, runs 5-fold stratified cross-validation on the training portion only, refits on that portion, and reports holdout metrics. It prints normal/attack counts, each fold's attack F1, per-class F1, and an always-attack F1 baseline. Metrics, trained models, and exact split row positions are saved in `model_results/`.

## Run

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -B main.py
```

Optional paths can be passed to `main.py`:

```powershell
.venv/Scripts/python.exe -B main.py --data-dir pre_processed --output-dir model_results
```

The evaluation brief describes a later, model-only comparison against one frozen dataset and fixed test rows. The saved holdout indices and model artifacts are intended to support that comparison. The current prepared features were produced by the shared preprocessing pipeline, whose fitted scaling/mappings include all selected files; this baseline therefore measures model performance on those supplied prepared values and is not a leakage-free end-to-end preprocessing evaluation.

`pre_processing/` contains the separate preprocessing implementation and its existing reports. `main.py` now runs only model training/evaluation.
