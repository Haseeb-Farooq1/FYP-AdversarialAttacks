"""One grouped holdout experiment on already-preprocessed numeric CSVs."""
import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
CHUNK_SIZE = 50000
# Exact reserved names only: other malformed CSVs must fail validation.
NON_MODEL_FILES = {"metadata.csv", "audit.csv", "validation_report.csv",
                   "preprocessing_report.csv", "split_assignments.csv"}


def discover_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    files = sorted((p for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() == ".csv"
                    and p.name.lower() not in NON_MODEL_FILES), key=lambda p: p.name)
    if not files:
        raise ValueError("No model CSV files found")
    return files


def inspect_files(files):
    """Validate every row, retaining only compact labels and per-file counts."""
    features = None
    labels, counts = [], []
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            header = next(csv.reader(stream), [])
        if len(header) != len(set(header)) or "label" not in header:
            raise ValueError(f"{path.name}: missing label or duplicate column names")
        current = [c for c in header if c not in {"label", "type"}]
        if not current or (features is not None and current != features):
            raise ValueError(f"{path.name}: incompatible predictor schema/order")
        features = current
        count = 0
        with pd.read_csv(path, chunksize=CHUNK_SIZE) as reader:
            for chunk in reader:
                if not all(pd.api.types.is_numeric_dtype(chunk[c]) for c in features):
                    raise ValueError(f"{path.name}: nonnumeric predictor near row {count + 1}")
                if not np.isfinite(chunk[features].to_numpy(dtype=np.float64)).all():
                    raise ValueError(f"{path.name}: NaN/infinite predictor near row {count + 1}")
                if not chunk.label.isin([0, 1]).all():
                    raise ValueError(f"{path.name}: label must contain only 0/1")
                labels.append(chunk.label.to_numpy(dtype=np.uint8))
                count += len(chunk)
        if not count:
            raise ValueError(f"{path.name}: empty model CSV")
        counts.append(count)
    y = np.concatenate(labels)
    if len(np.unique(y)) != 2:
        raise ValueError("Both normal (0) and attack (1) records are required")
    return features, y, counts


def sample_indices(y, size, seed=SEED):
    if size is None:
        return np.arange(len(y))
    if size < 2 or size > len(y):
        raise ValueError("Sample size must be between 2 and the source row count")
    counts = np.bincount(y, minlength=2)
    # Nearest feasible integer quotas; retain both classes without rebalancing.
    normal = int(np.floor(size * int(counts[0]) / len(y) + 0.5))
    normal = max(max(1, size - int(counts[1])),
                 min(normal, min(int(counts[0]), size - 1)))
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([
        rng.choice(np.flatnonzero(y == 0), normal, replace=False),
        rng.choice(np.flatnonzero(y == 1), size - normal, replace=False),
    ]))


def load_selected(files, features, indices, file_counts):
    """Second chunked pass: fill one matrix, without concatenating DataFrames."""
    X = np.empty((len(indices), len(features)), dtype=np.float64)
    offset = 0
    for path, expected in zip(files, file_counts):
        file_rows = 0
        with pd.read_csv(path, usecols=features, chunksize=CHUNK_SIZE) as reader:
            for chunk in reader:
                end = offset + len(chunk)
                lo, hi = np.searchsorted(indices, [offset, end])
                if hi > lo:
                    X[lo:hi] = chunk.loc[:, features].iloc[indices[lo:hi] - offset].to_numpy()
                offset = end
                file_rows += len(chunk)
        if file_rows != expected:
            raise ValueError(f"{path.name}: row count changed during loading")
    if not np.isfinite(X).all():
        raise ValueError("Selected predictors contain NaN/infinity")
    return X


def grouped_holdout(X, y):
    # Exact numeric row comparison, not a probabilistic fingerprint comparison.
    _, groups, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    groups = groups.reshape(-1)
    if len(counts) < 5:
        raise ValueError("At least five distinct predictor groups are required")
    normal = np.bincount(groups, weights=(y == 0), minlength=len(counts)).astype(np.int64)
    conflicts = (normal > 0) & (normal < counts)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, test = next(splitter.split(X, y, groups))
    crossing = len(np.intersect1d(groups[train], groups[test]))
    if crossing:
        raise ValueError("Duplicate predictor groups cross train/test")
    if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
        raise ValueError("Grouped holdout must contain both labels on each side")
    stats = {
        "grouping": "exact predictor equality, excluding label/type",
        "distinct_predictor_groups": int(len(counts)),
        "duplicate_groups": int(np.count_nonzero(counts > 1)),
        "repeated_rows_beyond_first": int(len(y) - len(counts)),
        "conflicting_label_groups": int(conflicts.sum()),
        "rows_in_conflicting_label_groups": int(counts[conflicts].sum()),
        "largest_group_rows": int(counts.max()),
        "crossing_groups": crossing,
    }
    return train, test, stats


def distribution(y):
    counts = np.bincount(y, minlength=2)
    return {str(i): {"count": int(counts[i]), "percentage": float(100 * counts[i] / len(y))}
            for i in range(2)}


def calculate_metrics(y, prediction):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    p, r, f, _ = precision_recall_fscore_support(y, prediction, average="binary", zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, prediction, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "attack_precision": float(p), "attack_recall": float(r), "attack_f1": float(f),
            "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
            "normal_recall_specificity": float(tn / (tn + fp)),
            "attack_recall_sensitivity": float(tp / (tp + fn)),
            "attack_false_negative_rate": float(fn / (tp + fn)),
            "normal_false_positive_rate": float(fp / (tn + fp)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "confusion_matrix_order": "rows=true, columns=predicted; labels=[0,1]"}


def save_result(result, path):
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def run(data_dir, sample_size=None):
    start = time.perf_counter()
    files = discover_files(data_dir)
    print("Selected model CSVs:", *(p.name for p in files), sep="\n", flush=True)
    features, source_y, file_counts = inspect_files(files)
    original_distribution = distribution(source_y)
    original_rows = len(source_y)
    loading = time.perf_counter() - start
    tick = time.perf_counter()
    indices = sample_indices(source_y, sample_size)
    y = source_y[indices]
    del source_y
    sampling = time.perf_counter() - tick
    print(f"Validated {original_rows:,} rows; loading {len(y):,} selected rows.", flush=True)
    tick = time.perf_counter()
    X = load_selected(files, features, indices, file_counts)
    del indices
    loading += time.perf_counter() - tick
    tick = time.perf_counter()
    train, test, duplicates = grouped_holdout(X, y)
    grouping = time.perf_counter() - tick
    print(f"Grouped holdout: {len(train):,} train / {len(test):,} test; training model.", flush=True)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    tick = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X[train], y[train])
    training = time.perf_counter() - tick
    tick = time.perf_counter()
    prediction = model.predict(X[test])
    prediction_time = time.perf_counter() - tick
    metrics = calculate_metrics(y[test], prediction)
    result = {
        "model": "LogisticRegression", "model_parameters": model.get_params(),
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "data_directory": Path(data_dir).as_posix(),
        "selected_filenames": [p.name for p in files], "source_file_count": len(files),
        "source_file_rows": dict(zip([p.name for p in files], file_counts)),
        "feature_names": features, "predictor_count": len(features),
        "original_rows": original_rows, "rows_used": len(y),
        "mode": "full-data" if sample_size is None else "sample", "random_state": SEED,
        "original_class_distribution": original_distribution,
        "sampled_class_distribution": distribution(y),
        "split_strategy": "First split only of StratifiedGroupKFold(5, shuffle=True, random_state=42)",
        "train_rows": len(train), "test_rows": len(test),
        "train_percentage": 100 * len(train) / len(y), "test_percentage": 100 * len(test) / len(y),
        "train_deviation_percentage_points": 100 * len(train) / len(y) - 80,
        "test_deviation_percentage_points": 100 * len(test) / len(y) - 20,
        "train_class_distribution": distribution(y[train]), "test_class_distribution": distribution(y[test]),
        "duplicates": duplicates, "metrics": metrics,
        "convergence": {"converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                        "iterations": model.n_iter_.tolist(), "warnings": [str(w.message) for w in caught]},
        "timings_seconds": {"discovery_validation_loading": loading, "sampling": sampling,
                            "duplicate_grouping_splitting": grouping, "training": training,
                            "prediction": prediction_time, "total": time.perf_counter() - start},
        "limitations": ["No additional preprocessing is performed.",
                        "Integer category codes are treated as numerical quantities by Logistic Regression.",
                        "Preprocessing fitted before this split can leak information; this is not proof of leakage-free evaluation.",
                        "Clean classification baseline, not adversarial robustness or a ranking of preprocessing pipelines."],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample-size", type=int)
    mode.add_argument("--full-data", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.data_dir, args.sample_size)
        path = Path(__file__).parent / "results" / "logistic_regression_results.json"
        save_result(result, path)
    except (ValueError, OSError, MemoryError, pd.errors.ParserError) as error:
        parser.exit(1, f"Evaluation stopped: {error}\n")
    print("\nLOGISTIC REGRESSION BASELINE")
    print(f"Files: {result['source_file_count']} | Rows used: {result['rows_used']:,} | Features: {result['predictor_count']}")
    print(f"Train: {result['train_rows']:,} | Test: {result['test_rows']:,}")
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "normal_recall_specificity", "attack_recall"):
        print(f"{key}: {result['metrics'][key]:.6f}")
    print("Confusion matrix:", result["metrics"]["confusion_matrix"])
    print("Convergence:", result["convergence"])
    print("Timings (seconds):", result["timings_seconds"])
    print("Results:", path)


if __name__ == "__main__":
    main()
