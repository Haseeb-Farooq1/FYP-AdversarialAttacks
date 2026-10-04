"""Process files 17-23 independently, without constructing any data split."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .preprocess import (
    BOOLEAN_COLUMNS, CATEGORICAL_COLUMNS, DROPPED_COLUMNS, FEATURE_COLUMNS,
    FILENAMES, NUMERICAL_COLUMNS, PROJECT_ROOT, CompactPreprocessor,
    clean_features, read_chunks, sha256_file, validate_targets, verify_schema,
)


def cleaned_chunks(path, chunk_size):
    for _, raw in read_chunks([path], chunk_size):
        validate_targets(raw)
        yield clean_features(raw)


def process_file(path, output_dir, config_dir, chunk_size=50000):
    """Learn and apply one file's own means, scaler, and category mappings."""
    started = time.perf_counter()
    path, output_dir, config_dir = map(Path, (path, output_dir, config_dir))
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    output_path = output_dir / path.name
    config_path = config_dir / f"{path.stem}_config.json"
    if output_path.resolve() == path.resolve():
        raise ValueError("The output must not overwrite the raw input")
    if output_path.exists() or config_path.exists():
        raise FileExistsError("Output/config already exists; use fresh directories")
    verify_schema(path)
    original_hash = sha256_file(path)
    processor = CompactPreprocessor()
    print(f"{path.name}: 1/3 learning this file's means and mappings", flush=True)
    processor.learn_values(cleaned_chunks(path, chunk_size))
    print(f"{path.name}: 2/3 fitting this file's scaler", flush=True)
    processor.fit_scaler(cleaned_chunks(path, chunk_size))
    fitted_rows = int(processor.scaler.n_samples_seen_)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    print(f"{path.name}: 3/3 writing {output_path.name}", flush=True)
    rows = invalid_src_bytes = 0
    labels, types = Counter(), Counter()
    missing_numeric, missing_categories, missing_booleans = Counter(), Counter(), Counter()
    for _, raw in read_chunks([path], chunk_size):
        invalid_src_bytes += int(raw.src_bytes.str.strip().eq("0.0.0.0").sum())
        clean = clean_features(raw)
        label, attack_type = validate_targets(raw)
        for column in NUMERICAL_COLUMNS:
            missing_numeric[column] += int(clean[column].isna().sum())
        for column in CATEGORICAL_COLUMNS:
            missing_categories[column] += int(clean[column].eq("").sum())
        for column in BOOLEAN_COLUMNS:
            missing_booleans[column] += int(clean[column].eq(2).sum())
        output = processor.transform(clean)
        output["label"], output["type"] = label, attack_type
        if output.isna().any().any():
            raise ValueError("Unexpected missing output value")
        labels.update(label.astype(str))
        types.update(attack_type)
        output.to_csv(output_path, index=False, mode="w" if rows == 0 else "a", header=rows == 0)
        rows += len(output)
    if rows != fitted_rows:
        raise ValueError("Input row count changed between fitting and export")
    if sha256_file(path) != original_hash:
        raise ValueError("Raw input changed during processing")

    config = processor.to_config()
    config["standard_scaler"]["fitted_rows"] = config["standard_scaler"].pop("training_rows")
    config["fit_scope"] = "entire_individual_file"
    config["source_file"] = path.name
    config["source_sha256"] = original_hash
    config["split_performed"] = False
    config["fit_warning"] = (
        "Fitted on all rows of this source file, not a training partition. "
        "Other files have independent scales/category codes. For later held-out evaluation, "
        "split raw records first and refit preprocessing on training records only."
    )
    config_path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result = {
        "input_file": path.name, "output_file": output_path.name,
        "config_file": config_path.name, "raw_sha256": original_hash,
        "raw_sha256_unchanged": True, "original_rows": rows, "final_rows": rows,
        "rows_removed": 0, "original_columns": 46, "original_feature_count": 44,
        "final_model_feature_count": 23, "output_columns": 25,
        "invalid_src_bytes_count": invalid_src_bytes,
        "missing_numeric_before_imputation": dict(missing_numeric),
        "missing_categories_before_encoding": dict(missing_categories),
        "missing_booleans": dict(missing_booleans),
        "missing_after_preprocessing": 0, "nonfinite_after_preprocessing": 0,
        "label_distribution": dict(labels), "type_distribution": dict(types),
        "output_size_bytes": output_path.stat().st_size,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    print(f"{path.name}: complete, {rows:,} rows, {result['runtime_seconds']} seconds", flush=True)
    return result


def run_individual(data_dir, output_dir, config_dir, chunk_size=50000):
    started = time.perf_counter()
    data_dir, output_dir, config_dir = map(Path, (data_dir, output_dir, config_dir))
    paths = [data_dir / name for name in FILENAMES]
    report_path = config_dir / "preprocessing_report.json"
    # Fail before writing anything if any required input is missing or an output exists.
    if report_path.exists():
        raise FileExistsError(report_path)
    for path in paths:
        verify_schema(path)
        for destination in (output_dir / path.name, config_dir / f"{path.stem}_config.json"):
            if destination.exists():
                raise FileExistsError(destination)
    results = [process_file(path, output_dir, config_dir, chunk_size) for path in paths]
    report = {
        "mode": "independent_per_file", "split_performed": False,
        "files_processed": len(results), "chunk_size": chunk_size,
        "original_rows": sum(r["original_rows"] for r in results),
        "final_rows": sum(r["final_rows"] for r in results),
        "final_model_feature_count": 23, "feature_columns": FEATURE_COLUMNS,
        "target": "label", "reporting_only": "type", "dropped_columns": DROPPED_COLUMNS,
        "invalid_src_bytes_count": sum(r["invalid_src_bytes_count"] for r in results),
        "missing_after_preprocessing": 0, "nonfinite_after_preprocessing": 0,
        "steps": [
            "Verify each raw schema and preserve the original source filename.",
            "Select the same 23 predictors and clean values in chunks.",
            "Learn numerical imputation means and category mappings independently from each entire file.",
            "Fit a separate StandardScaler on each file's imputed numerical values.",
            "Write one unsplit CSV and one saved configuration per file, preserving row order and both targets.",
            "Verify row counts, finite outputs and unchanged raw-file hashes.",
        ],
        "limitations": [
            "Independent category codes and scales are not directly comparable across files.",
            "All source rows influence fitting. Later held-out evaluation requires refitting from raw training records.",
            "No splitting, deduplication, resampling, synthetic records, or training class weights are produced.",
        ],
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "files": results, "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Complete: seven independent files, {report['final_rows']:,} rows, {report['runtime_seconds']} seconds", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "dataset/processed_zainab_individual")
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).parent / "artifacts/individual")
    parser.add_argument("--chunk-size", type=int, default=50000)
    args = parser.parse_args()
    run_individual(args.data_dir, args.output_dir, args.config_dir, args.chunk_size)


if __name__ == "__main__":
    main()
