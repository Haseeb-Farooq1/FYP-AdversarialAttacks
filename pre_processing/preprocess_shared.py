"""One shared fit, separate source-named CSVs; no model-evaluation split."""

import argparse
import hashlib
import json
import sqlite3
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

from pre_processing.preprocess import (
    BOOLEAN_COLUMNS, CATEGORICAL_COLUMNS, EXPECTED_COLUMNS, FEATURE_COLUMNS,
    NUMERICAL_COLUMNS, PROJECT_ROOT, clean_features,
    sha256_file, validate_targets, verify_schema,
)

PORT_COLUMNS = ["src_port", "dst_port"]
LOG_COLUMNS = [c for c in NUMERICAL_COLUMNS if c not in PORT_COLUMNS]
DROPPED_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in FEATURE_COLUMNS + ["label", "type"]]
FIT_WARNING = (
    "This shared preprocessing baseline fits transformation parameters using the selected "
    "preprocessing dataset as a whole. It is intended for preprocessing development and "
    "pipeline comparison. Before final model evaluation, raw records must first be divided "
    "into training/validation/test sets and all learned preprocessing parameters must be "
    "refitted using training data only."
)
PORT_WARNING = (
    "Ports are discrete network identifiers, not physical continuous measurements. "
    "Scaling is retained for baseline comparability and has limited semantic meaning. "
    "Later compare current ports, removing src_port, and categorical/bucketed dst_port."
)


def input_paths(data_dir, start=1, end=23):
    """Filename convention only; run_shared also accepts arbitrary explicit CSV paths."""
    if not 1 <= start <= end <= 23:
        raise ValueError("Expected 1 <= start <= end <= 23")
    paths = []
    for index in range(start, end + 1):
        if index != 23:
            paths.append(Path(data_dir) / f"Network_dataset_{index}.csv")
            continue
        final_in = Path(data_dir) / "Network_dataset_23(in).csv"
        final_plain = Path(data_dir) / "Network_dataset_23.csv"
        paths.append(final_in if final_in.exists() or not final_plain.exists() else final_plain)
    return paths


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_reference(path):
    """Portable provenance; external inputs retain their basename and content hash."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def read_shared_chunks(paths, chunk_size, columns=None):
    """Close each pandas reader even if validation fails part-way through a file."""
    if columns is None:
        columns = FEATURE_COLUMNS + ["label", "type"]
    for path in paths:
        with pd.read_csv(path, encoding="utf-8-sig", header=0,
                         usecols=columns, dtype="string", na_filter=False,
                         chunksize=chunk_size, on_bad_lines="error") as reader:
            for frame in reader:
                yield path, frame


def checked_clean(raw, source):
    """Reuse baseline validation without changing the baseline implementation."""
    try:
        validate_targets(raw)
        return clean_features(raw)
    except (ValueError, TypeError) as exc:
        # Include the offending Boolean states, not just a generic type failure.
        for c in BOOLEAN_COLUMNS:
            values = raw[c].str.strip().replace("-", "")
            bad = values[~values.isin(["F", "T", ""])]
            if len(bad):
                raise ValueError(f"{source}: {c}: unexpected Boolean values {bad.unique()[:5].tolist()}") from exc
        raise ValueError(f"{source}: {exc}") from exc


class SharedPreprocessor:
    """Frozen shared medians/maps plus one incrementally fitted StandardScaler."""

    def __init__(self):
        self.medians = None
        self.mappings = {}
        self.observed = {}
        self.scaler = StandardScaler()

    def numeric_before_scaling(self, clean):
        if self.medians is None:
            raise ValueError("Shared medians have not been fitted")
        a = clean[NUMERICAL_COLUMNS].to_numpy(dtype=np.float64, copy=True)
        a = np.where(np.isnan(a), self.medians, a)
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite numeric value after shared imputation")
        for c in LOG_COLUMNS:
            i = NUMERICAL_COLUMNS.index(c)
            if (a[:, i] < 0).any():
                raise ValueError(f"{c}: negative value before log1p; refusing to clip")
            a[:, i] = np.log1p(a[:, i])
        return a

    def transform(self, clean):
        output = pd.DataFrame(index=clean.index)
        output[NUMERICAL_COLUMNS] = self.scaler.transform(self.numeric_before_scaling(clean))
        for c in CATEGORICAL_COLUMNS:
            output[c] = clean[c].map(self.mappings[c]).fillna(1).astype("int32")
            output.loc[clean[c] == "", c] = 0
        for c in BOOLEAN_COLUMNS:
            if not clean[c].isin([0, 1, 2]).all():
                raise ValueError(f"{c}: unexpected Boolean code")
            output[c] = clean[c].astype("int8")
        output = output[FEATURE_COLUMNS]
        if not np.isfinite(output.to_numpy(dtype=np.float64)).all():
            raise ValueError("Nonfinite transformed predictor")
        return output

    def to_config(self):
        return {
            "format_version": 1, "mode": "shared_fit_separate_files",
            "fit_scope": "all_selected_preprocessing_files", "fit_warning": FIT_WARNING,
            "split_performed": False, "feature_columns": FEATURE_COLUMNS,
            "numerical_columns": NUMERICAL_COLUMNS, "categorical_columns": CATEGORICAL_COLUMNS,
            "boolean_columns": BOOLEAN_COLUMNS, "dropped_columns": DROPPED_COLUMNS,
            "feature_selection_note": "Fixed 23-feature compact baseline, not a claim that excluded features are universally useless.",
            "numeric_imputation_medians": dict(zip(NUMERICAL_COLUMNS, self.medians.tolist())),
            "cleaning": {"strip_whitespace": True, "missing_markers": ["", "-"],
                         "src_bytes_invalid_as_missing": ["0.0.0.0"],
                         "other_invalid_numeric": "error", "unexpected_boolean": "error",
                         "preserve_legitimate_zeros": True},
            "median_method": "Exact median of all observed finite cleaned float64 values; disk-backed per-column in-place partition, no sampling.",
            "log1p_columns": LOG_COLUMNS, "port_warning": PORT_WARNING,
            "log1p_nonnegativity_verified": {c: self.observed[c]["min"] >= 0 for c in LOG_COLUMNS},
            "transformation_order": ["explicit numeric cleaning", "shared median imputation", "log1p on configured non-port columns", "shared StandardScaler"],
            "observed_numerical": self.observed,
            "standard_scaler": {"mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist(),
                                "var": self.scaler.var_.tolist(), "fitted_rows": int(self.scaler.n_samples_seen_)},
            "category_mappings": self.mappings,
            "category_codes": {"missing": 0, "unknown": 1, "known_start": 2},
            "boolean_mapping": {"F": 0, "T": 1, "missing": 2},
            "category_warning": "Category/Boolean codes are discrete identifiers, not continuous adversarial variables; never scaled.",
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        }

def exact_disk_median(path):
    count = path.stat().st_size // np.dtype("float64").itemsize
    if not count:
        raise ValueError(f"{path.stem}: entirely missing numerical feature; cannot learn median")
    values = np.memmap(path, dtype="float64", mode="r+", shape=(count,))
    try:
        lo, hi = (count - 1) // 2, count // 2
        values.partition((lo, hi))
        # Half-sums avoid overflowing for two large positive finite values.
        return float(values[lo] / 2 + values[hi] / 2)
    finally:
        values.flush()
        values._mmap.close()


def inspect_and_learn(paths, work_dir, processor, chunk_size):
    """Pass 1: validate all values, spool medians, learn vocabulary, count raw duplicates."""
    categories = {c: set() for c in CATEGORICAL_COLUMNS}
    observed = {c: {"count": 0, "min": None, "max": None, "integer_valued": True} for c in NUMERICAL_COLUMNS}
    manifests = []
    with ExitStack() as stack:
        handles = {c: stack.enter_context((work_dir / f"{c}.bin").open("wb")) for c in NUMERICAL_COLUMNS}
        db = sqlite3.connect(work_dir / "duplicates.sqlite")
        stack.callback(db.close)
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-32768")
        # Full serialized raw values are the key: no probabilistic hash equality.
        db.execute("CREATE TABLE records (payload TEXT PRIMARY KEY, attack TEXT, n INTEGER, first_file TEXT, cross_file INTEGER) WITHOUT ROWID")
        db.execute("CREATE TABLE current_file (payload TEXT PRIMARY KEY) WITHOUT ROWID")
        total_unique = 0
        for path in paths:
            print(f"Inspecting {path.name}", flush=True)
            before_hash = sha256_file(path)
            stats = {"file": path.name, "source_path": source_reference(path), "rows": 0,
                     "sha256": before_hash, "invalid_src_bytes": 0, "label_distribution": Counter(),
                     "type_distribution": Counter(), "missing_numeric": Counter(),
                     "missing_categorical": Counter(), "missing_boolean": Counter()}
            db.execute("DELETE FROM current_file")
            for _, raw in read_shared_chunks([path], chunk_size, EXPECTED_COLUMNS):
                clean = checked_clean(raw, f"{path.name}, data rows {stats['rows'] + 1}-{stats['rows'] + len(raw)}")
                label, attack = validate_targets(raw)
                stats["rows"] += len(raw)
                stats["invalid_src_bytes"] += int(raw.src_bytes.str.strip().eq("0.0.0.0").sum())
                stats["label_distribution"].update(label.astype(str))
                stats["type_distribution"].update(attack)
                for c in NUMERICAL_COLUMNS:
                    a = clean[c].to_numpy(dtype=np.float64)
                    a = a[np.isfinite(a)]
                    if c in LOG_COLUMNS and (a < 0).any():
                        raise ValueError(f"{path.name}: {c}: negative value before log1p")
                    a.tofile(handles[c])
                    stats["missing_numeric"][c] += len(clean) - len(a)
                    if len(a):
                        o = observed[c]
                        o["min"] = float(a.min()) if o["min"] is None else min(o["min"], float(a.min()))
                        o["max"] = float(a.max()) if o["max"] is None else max(o["max"], float(a.max()))
                        o["count"] += len(a)
                        o["integer_valued"] = bool(o["integer_valued"] and np.equal(a, np.floor(a)).all())
                for c in CATEGORICAL_COLUMNS:
                    categories[c].update(clean[c].unique())
                    stats["missing_categorical"][c] += int(clean[c].eq("").sum())
                for c in BOOLEAN_COLUMNS:
                    stats["missing_boolean"][c] += int(clean[c].eq(2).sum())
                payloads = Counter(json.dumps(list(row), ensure_ascii=False, separators=(",", ":"))
                                   for row in raw[EXPECTED_COLUMNS].itertuples(index=False, name=None))
                db.executemany("INSERT INTO records VALUES (?, ?, ?, ?, 0) ON CONFLICT(payload) DO UPDATE SET n=n+excluded.n, cross_file=MAX(cross_file, first_file!=excluded.first_file)",
                               ((payload, json.loads(payload)[-1].strip(), count, path.name) for payload, count in payloads.items()))
                db.executemany("INSERT OR IGNORE INTO current_file VALUES (?)", ((p,) for p in payloads))
                db.commit()
            if not stats["rows"]:
                raise ValueError(f"{path.name}: empty dataset")
            unique_here = db.execute("SELECT COUNT(*) FROM current_file").fetchone()[0]
            unique_all = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            stats["within_file_duplicate_excess"] = stats["rows"] - unique_here
            stats["global_duplicate_excess_attributed_to_file"] = stats["rows"] - (unique_all - total_unique)
            total_unique = unique_all
            manifests.append(stats)
        total = sum(f["rows"] for f in manifests)
        cross_groups, cross_rows = db.execute("SELECT COUNT(*), COALESCE(SUM(n),0) FROM records WHERE cross_file=1").fetchone()
        duplicates = {
            "definition": "Exact equality of all 46 parsed raw string fields, before cleaning, including targets/ts/IPs; CSV quoting differences ignored, whitespace preserved. Full JSON row keys in SQLite avoid hash collisions.",
            "duplicate_excess_rows": total - total_unique,
            "duplicate_percentage": 100 * (total - total_unique) / total,
            "unique_raw_records": total_unique,
            "rows_in_duplicate_groups": db.execute("SELECT COALESCE(SUM(n),0) FROM records WHERE n>1").fetchone()[0],
            "cross_file_duplicate_groups": cross_groups, "rows_in_cross_file_groups": cross_rows,
            "duplicate_excess_by_type": dict(db.execute("SELECT attack, SUM(n-1) FROM records GROUP BY attack")),
            "by_file_note": "Within-file excess counts local repeats; global excess is attributed to later occurrences in supplied file order. Cross-file group rows include all occurrences.",
            "removed_rows": 0, "retained": True,
        }
    processor.medians = np.array([exact_disk_median(work_dir / f"{c}.bin") for c in NUMERICAL_COLUMNS])
    processor.observed = observed
    processor.mappings = {c: {v: i for i, v in enumerate(sorted(categories[c] - {""}), 2)} for c in CATEGORICAL_COLUMNS}
    return manifests, duplicates


def feature_metadata(processor):
    result = []
    for c in FEATURE_COLUMNS:
        item = {"feature": c, "adversarial_mutability": "review_required", "log1p": c in LOG_COLUMNS,
                "scaling": c in NUMERICAL_COLUMNS}
        if c in NUMERICAL_COLUMNS:
            item.update(feature_type="port/discrete" if c in PORT_COLUMNS else "numerical",
                        raw_observed=processor.observed[c], missing_handling="shared median",
                        median=float(processor.medians[NUMERICAL_COLUMNS.index(c)]))
            item["observed_nonnegative"] = processor.observed[c]["min"] >= 0
            item["constraint_note"] = "Observed data property only; not proof of attack feasibility or a universal upper bound."
        elif c in CATEGORICAL_COLUMNS:
            item.update(feature_type="categorical", mapping=processor.mappings[c],
                        missing_handling="code 0", unknown_handling="code 1", integer_valued_raw=None)
        else:
            item.update(feature_type="Boolean", mapping={"F": 0, "T": 1, "missing": 2},
                        missing_handling="code 2", integer_valued_raw=False)
        result.append(item)
    return {"fit_scope": "all_selected_preprocessing_files", "features": result, "port_warning": PORT_WARNING}


def make_report(report):
    rows = ["# Zainab's shared preprocessing baseline", "", FIT_WARNING, "",
            "## Architecture", "",
            "Previous individual baseline: each file fitted its own means, scaler and mappings. "
            "This version keeps separate files but learns one shared set of exact medians, one scaler "
            "and deterministic mappings from all selected records. Feature order is unchanged (23 predictors plus label/type).",
            "", "Numerical order: clean -> shared median -> log1p on ten non-port features -> shared StandardScaler. "
            "Ports use median -> scaler, without log1p. Category codes: missing=0, unknown=1, known=2+ in sorted order; "
            "Booleans: F=0, T=1, missing=2. Neither codes nor targets are scaled.", "", PORT_WARNING, "",
            "## Row preservation", "", "| Source/output filename | Input | Output | Retained | Invalid src_bytes |",
            "|---|---:|---:|---:|---:|"]
    for f in report["files"]:
        rows.append(f"| {f['file']} | {f['rows']:,} | {f['output_rows']:,} | 100% | {f['invalid_src_bytes']:,} |")
    rows += ["", f"Output missing values: 0; infinite values: 0. Runtime: {report['runtime_seconds']} seconds.",
             "", "## Exact duplicates (retained)", "", json.dumps(report["duplicates"], indent=2), "",
             "Per-file class distributions and missing counts are in preprocessing_report.json. "
             "Raw hashes are checked before and after processing.", "",
             "## Reproducibility and memory", "",
             "Numerical observations are spooled to disk by column. Exact float64 medians use in-place disk-backed partition, "
             "not sampling. SQLite compares full serialized raw rows for exact duplicates. Reads/writes are chunked; "
             "category sets remain in memory. Disk-backed work files are temporary. Shared parameters and source hashes "
             "are persisted.", "",
             "## Processing notes", "",
             "Duplicate rows are retained and no dataset split is performed. Category and Boolean codes are discrete identifiers. "
             "The fixed feature selection creates a compact 23-predictor baseline; it does not assert that excluded features are universally useless."]
    return "\n".join(rows) + "\n"


def run_shared(paths, output_dir, artifact_dir, report_path, chunk_size=50000):
    """Three CSV passes: shared inspection/medians, shared scaler, separate exports."""
    started = time.perf_counter()
    paths = [Path(p).resolve() for p in paths]
    output_dir, artifact_dir, report_path = map(Path, (output_dir, artifact_dir, report_path))
    if not paths or chunk_size < 1:
        raise ValueError("Supply input files and a positive chunk_size")
    if len({p.name.casefold() for p in paths}) != len(paths):
        raise ValueError("Input basenames must be unique")
    for p in paths:
        verify_schema(p)
    if output_dir.resolve() == artifact_dir.resolve():
        raise ValueError("Output CSVs and preprocessing artifacts must use separate paths")
    artifact_paths = [
        artifact_dir / "preprocessing_config.json",
        artifact_dir / "preprocessing_report.json",
        artifact_dir / "feature_metadata.json",
        report_path,
    ]
    if output_dir.exists() or any(path.exists() for path in artifact_paths):
        raise FileExistsError("Outputs already exist; choose fresh output and report paths")
    processor = SharedPreprocessor()
    output_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix=".shared-work-", dir=output_dir) as work:
        manifests, duplicates = inspect_and_learn(paths, Path(work), processor, chunk_size)
    print("Fitting one shared scaler", flush=True)
    for path, raw in read_shared_chunks(paths, chunk_size):
        processor.scaler.partial_fit(processor.numeric_before_scaling(checked_clean(raw, path.name)))
    total = sum(f["rows"] for f in manifests)
    if int(processor.scaler.n_samples_seen_) != total:
        raise ValueError("Source row count changed during shared fitting")
    config = processor.to_config()
    config.update(source_files=json.loads(json.dumps(manifests)), total_fitted_rows=total, duplicates=duplicates,
                  invalid_src_bytes_count=sum(f["invalid_src_bytes"] for f in manifests))
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    for path, stats in zip(paths, manifests):
        print(f"Writing {path.name} using shared configuration", flush=True)
        count = 0
        columns = FEATURE_COLUMNS + ["label", "type"]
        for _, raw in read_shared_chunks([path], chunk_size, columns):
            result = processor.transform(checked_clean(raw, path.name))
            label, attack = validate_targets(raw)
            result["label"], result["type"] = label, attack
            result.to_csv(output_dir / path.name, index=False, mode="a" if count else "w", header=count == 0)
            count += len(raw)
        if count != stats["rows"] or sha256_file(path) != stats["sha256"]:
            raise ValueError(f"{path.name}: raw file changed during processing")
        stats.update(output_rows=count, retention_percentage=100.0,
                     raw_sha256_unchanged=True, shared_config_sha256=config_digest)
    # source_files in config are an immutable fit manifest, independent of export accounting.
    report = {"mode": "shared_fit_separate_files", "fit_warning": FIT_WARNING,
              "split_performed": False, "files": manifests, "original_rows": total,
              "final_rows": sum(f["output_rows"] for f in manifests), "original_predictors": 44,
              "final_model_feature_count": 23, "output_columns": 25, "feature_columns": FEATURE_COLUMNS,
              "dropped_columns": DROPPED_COLUMNS, "duplicates": duplicates,
              "invalid_src_bytes_count": config["invalid_src_bytes_count"],
              "missing_after_preprocessing": 0, "nonfinite_after_preprocessing": 0,
              "chunk_size": chunk_size,
              "runtime_seconds": round(time.perf_counter() - started, 3)}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "preprocessing_config.json", config)
    write_json(artifact_dir / "preprocessing_report.json", report)
    write_json(artifact_dir / "feature_metadata.json", feature_metadata(processor))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(make_report(report), encoding="utf-8")
    print(f"Complete: {len(paths)} separate files, {total:,} rows; no split performed", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "Raw_DATASET")
    parser.add_argument("--files", type=Path, nargs="+", help="Explicit input paths; overrides start/end range")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=23)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "pre_processed")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "pre_processing")
    parser.add_argument("--report-path", type=Path, default=PROJECT_ROOT / "pre_processing" / "preprocessing_summary.md")
    parser.add_argument("--chunk-size", type=int, default=50000)
    args = parser.parse_args()
    run_shared(args.files or input_paths(args.data_dir, args.start, args.end), args.output_dir,
               args.artifact_dir, args.report_path, args.chunk_size)


if __name__ == "__main__":
    main()
