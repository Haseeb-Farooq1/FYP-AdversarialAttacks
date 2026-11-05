"""Train a LightGBM binary classifier and report per-file validation results."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "pre_processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model_results"
TARGET_COLUMN = "label"
NON_FEATURE_COLUMNS = {"label", "type"}
SEED = 42
TEST_SIZE = 0.20
FOLDS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model() -> LGBMClassifier:
    """Return the same compact, seeded binary LightGBM baseline for every fold."""
    return LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def score_predictions(y_true: np.ndarray, probability: np.ndarray) -> dict:
    predicted = (probability >= 0.5).astype(np.int8)
    class_f1 = f1_score(y_true, predicted, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "f1_normal": float(class_f1[0]),
        "precision_attack": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_attack": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_attack": float(class_f1[1]),
        "macro_f1": float(np.mean(class_f1)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc_attack": float(average_precision_score(y_true, probability)),
        "actual_class_counts": {
            "normal": int(np.count_nonzero(y_true == 0)),
            "attack": int(np.count_nonzero(y_true == 1)),
        },
        "predicted_class_counts": {
            "normal": int(np.count_nonzero(predicted == 0)),
            "attack": int(np.count_nonzero(predicted == 1)),
        },
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # 'type' is the textual version of the target and must not be used as a predictor.
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if TARGET_COLUMN not in header:
        raise ValueError(f"{path.name}: required target column {TARGET_COLUMN!r} is missing")
    feature_columns = [column for column in header if column not in NON_FEATURE_COLUMNS]
    if not feature_columns:
        raise ValueError(f"{path.name}: no predictor columns were found")

    frame = pd.read_csv(path, usecols=feature_columns + [TARGET_COLUMN], dtype=np.float32)
    if frame.empty:
        raise ValueError(f"{path.name}: dataset is empty")
    y = frame.pop(TARGET_COLUMN).to_numpy(dtype=np.int8, copy=True)
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{path.name}: label must contain only binary values 0 and 1")
    X = frame.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(X).all():
        raise ValueError(f"{path.name}: predictors contain missing or non-finite values")
    if np.unique(y).size != 2:
        raise ValueError(f"{path.name}: both label classes are required for stratified validation")
    return X, y, feature_columns


def _save_split_indices(path: Path, train_indices: np.ndarray, test_indices: np.ndarray) -> None:
    """Persist exact row positions so the holdout can be reused for later model comparisons."""
    np.savez_compressed(path, train_indices=train_indices, test_indices=test_indices)


def evaluate_file(path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    print(f"\n[{path.name}] Loading prepared rows...", flush=True)
    X, y, feature_columns = load_dataset(path)
    counts = np.bincount(y, minlength=2)
    if counts.min() < FOLDS + 1:
        raise ValueError(f"{path.name}: the smallest class needs at least {FOLDS + 1} rows")

    indices = np.arange(len(y), dtype=np.uint32)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    train_counts = np.bincount(y_train, minlength=2)
    holdout_counts = np.bincount(y_test, minlength=2)
    attack_only_f1 = float(f1_score(y_test, np.ones_like(y_test), zero_division=0))
    print(
        f"[{path.name}] actual labels: normal={counts[0]:,} ({counts[0] / len(y):.2%}), "
        f"attacks={counts[1]:,} ({counts[1] / len(y):.2%})",
        flush=True,
    )
    print(
        f"[{path.name}] split: train normal={train_counts[0]:,}, attacks={train_counts[1]:,}; "
        f"holdout normal={holdout_counts[0]:,}, attacks={holdout_counts[1]:,}; "
        f"always-attack holdout F1={attack_only_f1:.4f}",
        flush=True,
    )

    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []
    for fold, (fit_rows, validation_rows) in enumerate(splitter.split(X_train, y_train), start=1):
        model = build_model()
        model.fit(X_train[fit_rows], y_train[fit_rows])
        fold_score = score_predictions(y_train[validation_rows], model.predict_proba(X_train[validation_rows])[:, 1])
        fold_scores.append(fold_score)
        print(f"[{path.name}] fold {fold}/{FOLDS}: attack F1={fold_score['f1_attack']:.4f}", flush=True)

    final_model = build_model()
    final_model.fit(X_train, y_train)
    holdout_scores = score_predictions(y_test, final_model.predict_proba(X_test)[:, 1])

    stem = path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{stem}.joblib"
    split_path = output_dir / f"{stem}_split.npz"
    metrics_path = output_dir / f"{stem}_metrics.json"
    joblib.dump(final_model, model_path)
    _save_split_indices(split_path, train_indices, test_indices)

    metric_names = [
        key for key, value in fold_scores[0].items()
        if isinstance(value, (int, float))
    ]
    cv_summary = {
        name: {
            "mean": float(np.mean([fold[name] for fold in fold_scores])),
            "std": float(np.std([fold[name] for fold in fold_scores], ddof=1)),
        }
        for name in metric_names
    }
    result = {
        "dataset": path.name,
        "dataset_sha256": sha256_file(path),
        "target_column": TARGET_COLUMN,
        "positive_class": 1,
        "feature_columns": feature_columns,
        "rows": int(len(y)),
        "class_counts": {"0_normal": int(counts[0]), "1_attack": int(counts[1])},
        "train_class_counts": {"0_normal": int(train_counts[0]), "1_attack": int(train_counts[1])},
        "holdout_class_counts": {"0_normal": int(holdout_counts[0]), "1_attack": int(holdout_counts[1])},
        "split": {
            "method": "stratified 80/20 train/holdout split",
            "seed": SEED,
            "test_size": TEST_SIZE,
            "training_rows": int(len(train_indices)),
            "holdout_rows": int(len(test_indices)),
            "fixed_row_positions_file": split_path.name,
            "split_file_sha256": sha256_file(split_path),
        },
        "cross_validation": {
            "method": "StratifiedKFold",
            "folds": FOLDS,
            "scope": "training partition only",
            "fold_metrics": fold_scores,
            "summary": cv_summary,
        },
        "holdout_metrics": holdout_scores,
        "holdout_always_attack_baseline": {"f1_attack": attack_only_f1},
        "model": {
            "name": "LightGBM LGBMClassifier",
            "parameters": final_model.get_params(),
            "artifact": model_path.name,
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    metrics_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(
        f"[{path.name}] COMPLETE | rows={len(y):,} | CV attack F1="
        f"{cv_summary['f1_attack']['mean']:.4f} ± {cv_summary['f1_attack']['std']:.4f} | "
        f"holdout attack F1={holdout_scores['f1_attack']:.4f} "
        f"(always-attack={attack_only_f1:.4f}) | normal F1={holdout_scores['f1_normal']:.4f} | "
        f"balanced accuracy={holdout_scores['balanced_accuracy']:.4f} | "
        f"predicted normal={holdout_scores['predicted_class_counts']['normal']:,}, "
        f"attack={holdout_scores['predicted_class_counts']['attack']:,} | "
        f"{result['runtime_seconds']:.1f}s",
        flush=True,
    )
    print(f"[{path.name}] saved metrics to {metrics_path}", flush=True)
    return result


def run(data_dir: Path = DEFAULT_DATA_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[dict]:
    files = sorted(data_dir.glob("*.csv"), key=lambda path: path.name.casefold())
    if not files:
        raise FileNotFoundError(f"No prepared CSV files found in {data_dir}")

    results = []
    failures = []
    for index, path in enumerate(files, start=1):
        print(f"\n=== Dataset {index}/{len(files)}: {path.name} ===", flush=True)
        try:
            results.append(evaluate_file(path, output_dir))
        except Exception as exc:
            failures.append({"dataset": path.name, "error": str(exc)})
            print(f"[{path.name}] FAILED: {exc}", flush=True)

    summary = {
        "model": "LightGBM binary baseline",
        "data_directory": str(data_dir),
        "successful_files": len(results),
        "failed_files": failures,
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"\nFinished: {len(results)}/{len(files)} CSV files validated.", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} dataset(s) failed; see {output_dir / 'summary.json'}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
