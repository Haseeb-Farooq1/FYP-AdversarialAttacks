# model_luqman: neural network (MLP) baseline

Binary normal-vs-attack classifier trained on all 23 pre-processed files, pooled, with 5-fold
duplicate-safe cross-validation. Run from the project root, with the virtual environment active.

| Order | Command | What it does |
|---|---|---|
| 1 | `python model_luqman\prepare_data.py` | Checks the 23 CSVs and builds a compact cache (`cache/`, never committed) |
| 2 | `python model_luqman\make_folds.py` | Builds 5 folds that keep identical rows together (seed 42) |
| 3 | `python model_luqman\train_cv.py --quick` | 2-minute test run (results go to `results_quick/`) |
| 4 | `python model_luqman\train_cv.py` | Real run: 5 folds x 5 epochs, saves `results/cv_results.json` |
| 5 | `python model_luqman\brief_report.py` | Speed/size measurements, writes `results/brief_luqman.json` (Model Evaluation Brief) |

Files: `nn_model.py` (network, metrics), `prepare_data.py`, `make_folds.py`, `train_cv.py`, `brief_report.py`.
