import csv
import hashlib
import json
import math
import time
from pathlib import Path


MISSING_VALUES = {"", "-", "na", "n/a", "nan", "null", "none"}
TARGET_COLUMNS = {"label", "type"}
IGNORED_COLUMNS = {"uid"}
HASH_BUCKETS = 1024


def _normalise(value):
    value = (value or "").strip()
    return "" if value.lower() in MISSING_VALUES else value


def _as_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_rows(file_path):
    with Path(file_path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        original_columns = reader.fieldnames or []
        columns = [column.strip() for column in original_columns]

        for row in reader:
            cleaned_row = {
                column: _normalise(row.get(original_column))
                for column, original_column in zip(columns, original_columns)
            }
            if any(cleaned_row.values()):
                yield cleaned_row, columns


def _feature_columns(columns, numeric_columns):
    features = []
    for column in columns:
        if column in IGNORED_COLUMNS:
            continue
        if column in TARGET_COLUMNS:
            features.append(column)
            continue
        features.extend((column, f"{column}__missing"))
    return features


def _hashed_category(value):
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % HASH_BUCKETS


def _discover_schema(input_paths):
    columns = []
    seen = set()
    numeric_candidates = {}
    dataset_stats = {}

    for input_path in input_paths:
        row_count = 0
        missing_values = 0
        non_empty_columns = set()
        for row, file_columns in _read_rows(input_path):
            row_count += 1
            for column in file_columns:
                if column not in seen:
                    columns.append(column)
                    seen.add(column)
                if row.get(column):
                    non_empty_columns.add(column)
                else:
                    missing_values += 1
                if column not in IGNORED_COLUMNS and column not in TARGET_COLUMNS:
                    numeric_candidates[column] = numeric_candidates.get(column, True) and (
                        not row[column] or _as_number(row[column]) is not None
                    )
            for column in columns:
                if column not in IGNORED_COLUMNS and column not in TARGET_COLUMNS:
                    numeric_candidates.setdefault(column, True)
        dataset_stats[input_path.name] = {
            "before_rows": row_count,
            "before_columns": len(file_columns) if row_count else 0,
            "before_missing_values": missing_values,
            "before_non_empty_columns": sorted(non_empty_columns),
        }

    numeric_columns = {
        column for column, is_numeric in numeric_candidates.items() if is_numeric
    }
    return columns, numeric_columns, dataset_stats


def _numeric_statistics(input_paths, numeric_columns):
    statistics = {column: [0, 0.0, 0.0] for column in numeric_columns}

    for input_path in input_paths:
        for row, _ in _read_rows(input_path):
            for column in numeric_columns:
                number = _as_number(row.get(column, ""))
                if number is not None:
                    count, mean, sum_squares = statistics[column]
                    count += 1
                    delta = number - mean
                    mean += delta / count
                    sum_squares += delta * (number - mean)
                    statistics[column] = [count, mean, sum_squares]

    return {
        column: (
            mean,
            math.sqrt(sum_squares / count)
            if count > 1 and sum_squares > 0
            else 1.0,
        )
        for column, (count, mean, sum_squares) in statistics.items()
    }


def _encode_row(row, columns, numeric_columns, statistics):
    encoded = {}
    for column in columns:
        if column in IGNORED_COLUMNS:
            continue
        value = row.get(column, "")
        if column in TARGET_COLUMNS:
            encoded[column] = value
            continue

        missing = not value
        encoded[f"{column}__missing"] = int(missing)
        if column in numeric_columns:
            number = _as_number(value)
            mean, scale = statistics[column]
            encoded[column] = 0.0 if number is None else round((number - mean) / scale, 6)
        else:
            encoded[column] = -1 if missing else _hashed_category(value)
    return encoded


def preprocess_csv(file_path):
    """Load one CSV and clean it, retaining the legacy rows/columns API."""
    rows = []
    columns = []

    for row, file_columns in _read_rows(file_path):
        if not columns:
            columns = file_columns
        rows.append(row)

    return rows, columns


def preprocess_datasets(input_paths, output_dir=None):
    """Preprocess all input files with one shared, streaming transformation."""
    started_at = time.perf_counter()
    input_paths = [Path(path) for path in input_paths]
    output_dir = Path(output_dir) if output_dir else input_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    columns, numeric_columns, dataset_stats = _discover_schema(input_paths)
    statistics = _numeric_statistics(input_paths, numeric_columns)
    output_columns = _feature_columns(columns, numeric_columns)
    results = []

    for index, input_path in enumerate(input_paths, start=1):
        output_path = output_dir / f"PreProcessed_{index}.csv"
        row_count = 0
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=output_columns)
            writer.writeheader()
            for row, _ in _read_rows(input_path):
                writer.writerow(_encode_row(row, columns, numeric_columns, statistics))
                row_count += 1
        dataset_stat = dataset_stats[input_path.name]
        dataset_stat.update(
            {
                "output_file": output_path.name,
                "after_rows": row_count,
                "after_columns": len(output_columns),
                "output_size_bytes": output_path.stat().st_size,
                "rows_removed": dataset_stat["before_rows"] - row_count,
                "columns_removed": dataset_stat["before_columns"] - len(output_columns),
                "columns_added": len(output_columns) - dataset_stat["before_columns"],
            }
        )
        results.append((output_path, row_count, len(output_columns)))

    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    categorical_columns = [
        column
        for column in columns
        if column not in numeric_columns
        and column not in TARGET_COLUMNS
        and column not in IGNORED_COLUMNS
    ]
    report = {
        "summary": {
            "datasets_processed": len(input_paths),
            "processing_time_seconds": elapsed_seconds,
            "processing_time_minutes": round(elapsed_seconds / 60, 2),
            "rows_per_dataset": dataset_stats[input_paths[0].name]["after_rows"],
            "output_columns": len(output_columns),
            "numeric_features": len(numeric_columns),
            "categorical_features": len(categorical_columns),
            "missing_indicators_added": len(
                [column for column in columns if column not in TARGET_COLUMNS and column not in IGNORED_COLUMNS]
            ),
            "targets_preserved": sorted(TARGET_COLUMNS),
            "uid_removed": bool(IGNORED_COLUMNS.intersection(columns)),
        },
        "datasets": [
            {
                "dataset": index,
                "input_columns": dataset_stats[path.name]["before_columns"],
                "output_columns": dataset_stats[path.name]["after_columns"],
                "missing_values": dataset_stats[path.name]["before_missing_values"],
                "output_size_mb": round(dataset_stats[path.name]["output_size_bytes"] / (1024 * 1024), 2),
                "rows_removed": dataset_stats[path.name]["rows_removed"],
            }
            for index, path in enumerate(input_paths, start=1)
        ],
    }
    report_path = output_dir / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return results


def compact_report(report_path):
    """Replace a detailed report with a short dataset and runtime summary."""
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    compact = {
        "total_runtime_seconds": report["runtime_seconds"],
        "datasets": [],
    }

    for name, dataset in report["datasets"].items():
        compact["datasets"].append(
            {
                "dataset": name,
                "output": dataset["output_file"],
                "rows_before": dataset["before_rows"],
                "rows_after": dataset["after_rows"],
                "features_before": dataset["before_columns"]
                - (1 if name == "Network_dataset_6.csv" else 0)
                - len(TARGET_COLUMNS),
                "features_after": dataset["after_columns"] - len(TARGET_COLUMNS),
                "output_size_mb": round(dataset["output_size_bytes"] / (1024 * 1024), 2),
            }
        )

    report_path.write_text(json.dumps(compact, indent=2), encoding="utf-8")
