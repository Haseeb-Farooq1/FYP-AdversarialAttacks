"""Leakage-aware LightGBM training and evaluation for both prepared datasets."""

from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_ROOT / "data_split"
TARGET_COLUMN = "label"
SEED = 42
FOLDS = 5


def load_split(split_dir: Path) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load all source chunks in a split; metadata never enters model features."""
    paths = sorted(split_dir.glob("*.csv"), key=lambda path: path.name.casefold())
    if not paths:
        raise FileNotFoundError(f"No prepared CSV files found in {split_dir}")
    header = pd.read_csv(paths[0], nrows=0).columns.tolist()
    feature_columns = [column for column in header
                       if column not in {TARGET_COLUMN, "type", "__time"}]
    dtypes = {column: "float32" for column in feature_columns}
    dtypes.update({TARGET_COLUMN: "int8", "type": "string", "__time": "int64"})
    frames = [pd.read_csv(path, dtype=dtypes) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    del frames
    if TARGET_COLUMN not in frame:
        raise ValueError(f"{split_dir}: missing target column {TARGET_COLUMN!r}")
    y = frame[TARGET_COLUMN].astype("int8")
    time = pd.to_numeric(frame.pop("__time"), errors="raise").to_numpy()
    X = frame.drop(columns=[TARGET_COLUMN, "type"], errors="ignore")
    if X.empty or any(
        not np.isfinite(X[column].to_numpy(copy=False)).all()
        for column in X.columns
    ):
        raise ValueError(f"{split_dir}: features are empty or contain nonfinite values")
    return X, y, time


def candidates() -> list[dict]:
    """Small, regularized search space that limits depth-driven overfitting."""
    common = {
        "objective": "binary",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
        "subsample_freq": 1,
    }
    return [
        {**common, "num_leaves": 15, "max_depth": 5, "min_child_samples": 100,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        {**common, "num_leaves": 31, "max_depth": 7, "min_child_samples": 100,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {**common, "num_leaves": 31, "max_depth": -1, "min_child_samples": 250,
         "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 5.0},
    ]


def build_model(params: dict) -> LGBMClassifier:
    return LGBMClassifier(**params)


def metric_values(y_true, y_pred, y_score) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
    }


def chronological_cv(X: pd.DataFrame, y: pd.Series, time: np.ndarray):
    order = np.argsort(time, kind="stable")
    ordered_times = time[order]
    unique_times, group_index = np.unique(ordered_times, return_inverse=True)
    # Some datasets contain long single-class periods. Choose the earliest
    # warm-up point that permits five later, non-overlapping time windows with
    # both classes, while keeping equal timestamps together.
    y_values = y.to_numpy()
    ordered_y = y_values[order]
    group_sizes = np.bincount(group_index, minlength=len(unique_times))
    group_positives = np.bincount(
        group_index, weights=ordered_y, minlength=len(unique_times)
    )
    cumulative_positive = np.concatenate(([0], np.cumsum(group_positives)))
    cumulative_negative = np.concatenate((
        [0], np.cumsum(group_sizes - group_positives)
    ))
    selected = None
    for warmup in np.arange(0.05, 0.96, 0.01):
        boundaries = np.linspace(
            int(warmup * len(unique_times)), len(unique_times), FOLDS + 1,
            dtype=int,
        )
        candidate = []
        valid = True
        for fold in range(FOLDS):
            start, end = boundaries[fold], boundaries[fold + 1]
            fit_has_both = (
                cumulative_positive[start] > 0
                and cumulative_negative[start] > 0
            )
            valid_positive = cumulative_positive[end] - cumulative_positive[start]
            valid_negative = cumulative_negative[end] - cumulative_negative[start]
            if start >= end or not fit_has_both or valid_positive == 0 or valid_negative == 0:
                valid = False
                break
            candidate.append((start, end))
        if valid:
            selected = candidate
            print(f"  chronological CV warm-up: {warmup:.0%} of unique timestamps", flush=True)
            break
    if selected is None:
        raise ValueError(
            "Could not construct five chronological folds with both classes in "
            "every fit and validation window"
        )
    fold_results = [
        (
            order[group_index < start],
            order[(group_index >= start) & (group_index < end)],
        )
        for start, end in selected
    ]
    return fold_results


def tune_and_score(X: pd.DataFrame, y: pd.Series, time: np.ndarray):
    folds = chronological_cv(X, y, time)
    best_params = None
    best_score = -np.inf
    best_summary = None
    for candidate_index, params in enumerate(candidates(), start=1):
        results = []
        for fold, (fit_rows, valid_rows) in enumerate(folds, start=1):
            model = build_model(params)
            model.fit(X.iloc[fit_rows], y.iloc[fit_rows])
            score = model.predict_proba(X.iloc[valid_rows])[:, 1]
            pred = (score >= 0.5).astype("int8")
            values = metric_values(y.iloc[valid_rows], pred, score)
            results.append(values)
            print(f"  config {candidate_index}, fold {fold}/{FOLDS}: "
                  f"balanced_accuracy={values['balanced_accuracy']:.4f}, "
                  f"F1={values['f1']:.4f}", flush=True)
        mean_balanced = float(np.mean([item["balanced_accuracy"] for item in results]))
        print(f"  config {candidate_index}: mean CV balanced_accuracy={mean_balanced:.4f}", flush=True)
        if mean_balanced > best_score:
            best_score = mean_balanced
            best_params = params
            best_summary = {
                metric: float(np.mean([item[metric] for item in results]))
                for metric in results[0]
            }
    return best_params, best_summary


def train_dataset(name: str, dataset_dir: Path):
    train_dir, validation_dir, test_dir = (
        dataset_dir / "train", dataset_dir / "validation", dataset_dir / "test"
    )
    X_train, y_train, time_train = load_split(train_dir)
    feature_names = list(X_train.columns)
    train_rows = len(y_train)
    print(f"\n{name}: train={train_rows:,}; positive rate={y_train.mean():.4f}", flush=True)
    params, cv_metrics = tune_and_score(X_train, y_train, time_train)
    print(f"{name}: selected 5-fold chronological CV metrics", flush=True)
    for metric, value in cv_metrics.items():
        print(f"  {metric}: {value:.6f}", flush=True)

    # Validation data joins training only after model selection; test stays untouched.
    X_validation, y_validation, _ = load_split(validation_dir)
    if feature_names != list(X_validation.columns):
        raise ValueError(f"{name}: validation feature columns do not match training")
    validation_rows = len(y_validation)
    X_final = pd.concat([X_train, X_validation], ignore_index=True)
    y_final = pd.concat([y_train, y_validation], ignore_index=True)
    model = build_model(params).fit(X_final, y_final)
    del X_train, X_validation, X_final, y_final, time_train
    gc.collect()

    X_test, y_test, _ = load_split(test_dir)
    if feature_names != list(X_test.columns):
        raise ValueError(f"{name}: test feature columns do not match training")
    test_score = model.predict_proba(X_test)[:, 1]
    test_pred = (test_score >= 0.5).astype("int8")
    test_metrics = metric_values(y_test, test_pred, test_score)
    print(f"{name}: untouched chronological test metrics", flush=True)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}", flush=True)
    print(f"{name}: confusion matrix (rows=true, columns=predicted; labels=[0, 1])", flush=True)
    matrix = confusion_matrix(y_test, test_pred, labels=[0, 1])
    print(matrix, flush=True)

    # Keep each fitted model and its report in its own dataset-specific folder.
    result_dir = PROJECT_ROOT / "model_results" / dataset_dir.name
    result_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, result_dir / "lightgbm_model.joblib")
    report = {
        "dataset": name,
        "train_rows_for_cv": int(train_rows),
        "validation_rows_added_for_final_fit": int(validation_rows),
        "final_test_rows": int(len(y_test)),
        "best_parameters": params,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "test_confusion_matrix_labels_0_1": matrix.tolist(),
    }
    (result_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"{name}: saved separate model and metrics in {result_dir}", flush=True)
    return model


def run(dataset_names: list[str] | None = None) -> list[LGBMClassifier]:
    datasets = [
        ("Raw_DATASET", "Raw_DATASET (NF-UNSW-NB15)"),
        ("Raw_DATASET2", "Raw_DATASET2 (Zeek network chunks)"),
    ]
    if dataset_names is not None:
        requested = set(dataset_names)
        datasets = [item for item in datasets if item[0] in requested]
        if requested != {item[0] for item in datasets}:
            raise ValueError(f"Unknown dataset name(s): {sorted(requested)}")
    models = []
    for directory_name, name in datasets:
        path = SPLIT_DIR / directory_name
        if not path.exists():
            raise FileNotFoundError(f"Prepared splits missing for {name}: run main.py")
        models.append(train_dataset(name, path))
    return models
