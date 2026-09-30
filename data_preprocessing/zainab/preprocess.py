"""A chunked 23-feature baseline. Run from the repository root with -m."""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILENAMES = [f"Network_dataset_{i}.csv" for i in range(17, 23)] + [
    "Network_dataset_23(in).csv"
]
EXPECTED_COLUMNS = """
ts src_ip src_port dst_ip dst_port proto service duration src_bytes dst_bytes
conn_state missed_bytes src_pkts src_ip_bytes dst_pkts dst_ip_bytes dns_query
dns_qclass dns_qtype dns_rcode dns_AA dns_RD dns_RA dns_rejected ssl_version
ssl_cipher ssl_resumed ssl_established ssl_subject ssl_issuer http_trans_depth
http_method http_uri http_referrer http_version http_request_body_len
http_response_body_len http_status_code http_user_agent http_orig_mime_types
http_resp_mime_types weird_name weird_addl weird_notice label type
""".split()
NUMERICAL_COLUMNS = """
src_port dst_port duration src_bytes dst_bytes missed_bytes src_pkts
src_ip_bytes dst_pkts dst_ip_bytes http_request_body_len http_response_body_len
""".split()
CATEGORICAL_COLUMNS = """
proto service conn_state dns_qclass dns_qtype dns_rcode http_status_code
""".split()
BOOLEAN_COLUMNS = ["dns_AA", "dns_RD", "dns_RA", "dns_rejected"]
FEATURE_COLUMNS = [
    c for c in EXPECTED_COLUMNS
    if c in NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS + BOOLEAN_COLUMNS
]
TARGET_COLUMNS = ["label", "type"]
READ_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS
SPLIT_NAMES = ("train", "validation", "test")
ATTACK_TYPES = sorted([
    "normal", "backdoor", "ddos", "dos", "injection", "mitm", "password",
    "ransomware", "scanning", "xss",
])
BOOLEAN_MAPPING = {"F": 0, "T": 1, "": 2}
DROPPED_COLUMNS = {
    "ts": "Timestamp: exclude collection context from the classifier.",
    "src_ip": "Source identifier: avoid memorizing testbed addresses.",
    "dst_ip": "Destination identifier: avoid memorizing testbed addresses.",
    "dns_query": "Raw query vocabulary; omit text to keep the baseline compact.",
    **{c: "Over 99.98% missing in the initial files 17-23 inspection." for c in [
        "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established",
        "ssl_subject", "ssl_issuer",
    ]},
    "http_trans_depth": "Almost entirely missing; observed nonmissing value is 1.",
    "http_version": "Almost entirely missing; observed nonmissing value is 1.1.",
    **{c: "Over 99.89% missing; omit sparse HTTP text/categories." for c in [
        "http_method", "http_uri", "http_referrer", "http_user_agent",
        "http_orig_mime_types", "http_resp_mime_types",
    ]},
    **{c: "Over 99.99% missing in the initial inspection." for c in [
        "weird_name", "weird_addl", "weird_notice",
    ]},
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_schema(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), [])
    if [c.strip() for c in header] != EXPECTED_COLUMNS:
        raise ValueError(f"{path.name}: expected the exact 46-column schema/order")


def read_chunks(paths, chunk_size, columns=READ_COLUMNS):
    for path in paths:
        for frame in pd.read_csv(
            path, encoding="utf-8-sig", header=0, names=EXPECTED_COLUMNS,
            usecols=columns, dtype="string", na_filter=False,
            chunksize=chunk_size, on_bad_lines="error",
        ):
            yield path, frame


def validate_targets(frame):
    label = frame["label"].str.strip()
    attack_type = frame["type"].str.strip()
    if not label.isin(["0", "1"]).all() or not attack_type.isin(ATTACK_TYPES).all():
        raise ValueError("Missing or unrecognized label/type value")
    if not ((label == "0") == (attack_type == "normal")).all():
        raise ValueError("label and type disagree")
    return label.astype("int8"), attack_type


def inspect_inputs(paths, chunk_size):
    """Read only targets into a compact one-byte-per-row stratification array."""
    manifest, type_blocks = [], []
    for path in paths:
        verify_schema(path)
        count = 0
        for _, frame in read_chunks([path], chunk_size, TARGET_COLUMNS):
            _, attack_type = validate_targets(frame)
            codes = pd.Categorical(attack_type, categories=ATTACK_TYPES).codes
            type_blocks.append(codes.astype(np.uint8))
            count += len(frame)
        if not count:
            raise ValueError(f"{path.name}: no data rows")
        manifest.append({
            "file": path.name, "rows": count, "columns": 46,
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
        })
    return manifest, np.concatenate(type_blocks)


def make_split_assignments(type_codes, seed=42):
    """Shuffle row indices within each attack type; use integer 70/15/15 quotas."""
    rng = np.random.default_rng(seed)
    assignments = np.empty(len(type_codes), dtype=np.uint8)
    for code in np.unique(type_codes):
        indices = np.flatnonzero(type_codes == code)
        rng.shuffle(indices)
        n_train, n_validation = len(indices) * 70 // 100, len(indices) * 15 // 100
        if min(n_train, n_validation, len(indices) - n_train - n_validation) == 0:
            raise ValueError("An attack type has too few rows for all three splits")
        assignments[indices[:n_train]] = 0
        assignments[indices[n_train:n_train + n_validation]] = 1
        assignments[indices[n_train + n_validation:]] = 2
    return assignments


def clean_features(frame):
    """Explicit types; blank and '-' are missing, while real zeros remain zero."""
    clean = pd.DataFrame(index=frame.index)
    for column in FEATURE_COLUMNS:
        values = frame[column].str.strip().replace("-", "")
        if column in NUMERICAL_COLUMNS:
            if column == "src_bytes":
                values = values.replace("0.0.0.0", "")
            clean[column] = pd.to_numeric(values.replace("", pd.NA), errors="raise").to_numpy(
                dtype=np.float64, na_value=np.nan,
            )
            numeric = clean[column].to_numpy()
            if np.isinf(numeric).any() or (numeric < 0).any():
                raise ValueError(f"{column}: infinite or negative numeric value")
            observed = numeric[np.isfinite(numeric)]
            if column != "duration" and (observed != np.floor(observed)).any():
                raise ValueError(f"{column}: expected integer values")
            if column in ("src_port", "dst_port") and (observed > 65535).any():
                raise ValueError(f"{column}: value exceeds 65535")
        elif column in BOOLEAN_COLUMNS:
            if not values.isin(BOOLEAN_MAPPING).all():
                raise ValueError(f"{column}: expected F, T, or missing")
            clean[column] = values.map(BOOLEAN_MAPPING).astype("int8")
        else:
            clean[column] = values
    return clean


def training_chunks(paths, assignments, chunk_size):
    offset = 0
    for _, frame in read_chunks(paths, chunk_size):
        codes = assignments[offset:offset + len(frame)]
        offset += len(frame)
        training = frame.loc[codes == 0]
        if len(training):
            yield clean_features(training)
    if offset != len(assignments):
        raise ValueError("Input row count changed between passes")


class CompactPreprocessor:
    """Learn means/maps first, then fit StandardScaler on imputed training rows."""

    def __init__(self):
        self.means = None
        self.mappings = {}
        self.scaler = StandardScaler()

    def learn_values(self, chunks):
        sums = np.zeros(len(NUMERICAL_COLUMNS), dtype=np.float64)
        counts = np.zeros(len(NUMERICAL_COLUMNS), dtype=np.int64)
        categories = {c: set() for c in CATEGORICAL_COLUMNS}
        for frame in chunks:
            numeric = frame[NUMERICAL_COLUMNS].to_numpy(dtype=np.float64)
            sums += np.nansum(numeric, axis=0)
            counts += np.isfinite(numeric).sum(axis=0)
            for column in CATEGORICAL_COLUMNS:
                categories[column].update(frame[column].unique())
        if (counts == 0).any():
            empty = np.array(NUMERICAL_COLUMNS)[counts == 0].tolist()
            raise ValueError(f"Cannot learn training means for entirely missing columns: {empty}")
        self.means = sums / counts
        for column, values in categories.items():
            # Actual categories start at 2. Codes 0/1 never depend on held-out data.
            self.mappings[column] = {
                value: code for code, value in enumerate(sorted(values - {""}), start=2)
            }

    def imputed_numeric(self, frame):
        numeric = frame[NUMERICAL_COLUMNS].to_numpy(dtype=np.float64)
        return np.where(np.isnan(numeric), self.means, numeric)

    def fit_scaler(self, chunks):
        for frame in chunks:
            self.scaler.partial_fit(self.imputed_numeric(frame))

    def transform(self, frame):
        output = pd.DataFrame(index=frame.index)
        scaled = self.scaler.transform(self.imputed_numeric(frame))
        output[NUMERICAL_COLUMNS] = scaled
        for column in CATEGORICAL_COLUMNS:
            output[column] = frame[column].map(self.mappings[column]).fillna(1).astype("int32")
            output.loc[frame[column] == "", column] = 0
        output[BOOLEAN_COLUMNS] = frame[BOOLEAN_COLUMNS]
        output = output[FEATURE_COLUMNS]
        if not np.isfinite(output.to_numpy(dtype=np.float64)).all():
            raise ValueError("Nonfinite value in transformed model inputs")
        return output

    def to_config(self):
        return {
            "feature_columns": FEATURE_COLUMNS,
            "numerical_columns": NUMERICAL_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "boolean_columns": BOOLEAN_COLUMNS,
            "category_codes": {"missing": 0, "unknown": 1, "known_start": 2},
            "category_mappings": self.mappings,
            "boolean_mapping": {"F": 0, "T": 1, "missing": 2},
            "numeric_imputation_means": dict(zip(NUMERICAL_COLUMNS, self.means.tolist())),
            "standard_scaler": {
                "mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist(),
                "var": self.scaler.var_.tolist(),
                "training_rows": int(self.scaler.n_samples_seen_),
            },
            "categorical_warning": "Category/Boolean codes are discrete, not continuous measurements; never scale or continuously perturb them.",
            "log1p": False,
        }

    @classmethod
    def from_config(cls, config):
        if config["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError("Saved feature order does not match this implementation")
        result = cls()
        result.means = np.array([config["numeric_imputation_means"][c] for c in NUMERICAL_COLUMNS])
        result.mappings = config["category_mappings"]
        scaler = config["standard_scaler"]
        result.scaler.mean_ = np.array(scaler["mean"])
        result.scaler.scale_ = np.array(scaler["scale"])
        result.scaler.var_ = np.array(scaler["var"])
        result.scaler.n_samples_seen_ = scaler["training_rows"]
        result.scaler.n_features_in_ = len(NUMERICAL_COLUMNS)
        return result


def write_outputs(paths, assignments, processor, output_dir, chunk_size):
    summaries = {
        name: {"rows": 0, "label": Counter(), "type": Counter(),
               "missing_numeric_before_imputation": Counter(),
               "missing_categories_before_encoding": Counter(),
               "unknown_categories": Counter(), "missing_booleans": Counter(),
               "missing_after_preprocessing": 0, "nonfinite_after_preprocessing": 0}
        for name in SPLIT_NAMES
    }
    invalid_by_file = Counter()
    offset = 0
    for path, raw in read_chunks(paths, chunk_size):
        invalid_by_file[path.name] += int(raw["src_bytes"].str.strip().eq("0.0.0.0").sum())
        frame = clean_features(raw)
        label, attack_type = validate_targets(raw)
        codes = assignments[offset:offset + len(frame)]
        offset += len(frame)
        for split_code, name in enumerate(SPLIT_NAMES):
            selected = frame.loc[codes == split_code]
            if selected.empty:
                continue
            stats = summaries[name]
            for column in NUMERICAL_COLUMNS:
                stats["missing_numeric_before_imputation"][column] += int(selected[column].isna().sum())
            for column in CATEGORICAL_COLUMNS:
                missing = selected[column].eq("")
                stats["missing_categories_before_encoding"][column] += int(missing.sum())
                stats["unknown_categories"][column] += int((~missing & ~selected[column].isin(processor.mappings[column])).sum())
            for column in BOOLEAN_COLUMNS:
                stats["missing_booleans"][column] += int(selected[column].eq(2).sum())
            output = processor.transform(selected)
            output["label"] = label.loc[selected.index]
            output["type"] = attack_type.loc[selected.index]
            stats["missing_after_preprocessing"] += int(output.isna().sum().sum())
            # transform() already rejects all nonfinite numerical/model values.
            stats["label"].update(output["label"].astype(str))
            stats["type"].update(output["type"])
            output.to_csv(output_dir / f"{name}.csv", index=False,
                          mode="w" if stats["rows"] == 0 else "a", header=stats["rows"] == 0)
            stats["rows"] += len(output)
    if offset != len(assignments):
        raise ValueError("Input row count changed between passes")
    return summaries, dict(invalid_by_file)


def run_pipeline(data_dir, output_dir, artifact_dir, chunk_size=50000, seed=42):
    started = time.perf_counter()
    data_dir, output_dir, artifact_dir = map(Path, (data_dir, output_dir, artifact_dir))
    paths = [data_dir / filename for filename in FILENAMES]
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    generated = [output_dir / f"{name}.csv" for name in SPLIT_NAMES] + [
        output_dir / "split_assignments.npy",
        artifact_dir / "preprocessing_config.json", artifact_dir / "preprocessing_report.json",
    ]
    if any(path.exists() for path in generated):
        raise FileExistsError("Output artifacts already exist; choose fresh output/artifact directories")

    print("1/4: Verifying seven schemas and counting/stratifying targets...", flush=True)
    manifest, type_codes = inspect_inputs(paths, chunk_size)
    assignments = make_split_assignments(type_codes, seed)
    del type_codes
    print(f"      {len(assignments):,} rows; split counts {np.bincount(assignments).tolist()}", flush=True)
    processor = CompactPreprocessor()
    print("2/4: Learning numerical means and category mappings from training rows...", flush=True)
    processor.learn_values(training_chunks(paths, assignments, chunk_size))
    print("3/4: Fitting StandardScaler on imputed training rows only...", flush=True)
    processor.fit_scaler(training_chunks(paths, assignments, chunk_size))
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "split_assignments.npy", assignments, allow_pickle=False)
    print("4/4: Transforming and writing train/validation/test chunks...", flush=True)
    summaries, invalid_by_file = write_outputs(paths, assignments, processor, output_dir, chunk_size)
    # Compare content hashes after processing to prove raw files were not changed.
    for path, original in zip(paths, manifest):
        if sha256_file(path) != original["sha256"]:
            raise ValueError(f"Raw input changed during processing: {path.name}")
    retained = sum(summary["rows"] for summary in summaries.values())
    if retained != len(assignments):
        raise ValueError("Output row accounting failed")
    train_counts = summaries["train"]["label"]
    class_weights = {key: summaries["train"]["rows"] / (len(train_counts) * count)
                     for key, count in sorted(train_counts.items())}
    config = processor.to_config()
    config["class_weights"] = class_weights
    report = {
        "input_files": manifest, "original_rows": len(assignments), "final_rows": retained,
        "original_columns": 46, "original_feature_count": 44, "final_model_feature_count": 23,
        "output_columns": 25, "feature_columns": FEATURE_COLUMNS,
        "target": "label", "reporting_only": "type", "dropped_columns": DROPPED_COLUMNS,
        "missing_value_handling": {
            "numerical": "Training means, then training-fitted StandardScaler; legitimate zeros preserved.",
            "categorical": "Missing=0, unknown=1, training categories=2+; no scaling.",
            "boolean": "F=0, T=1, missing=2; unexpected values rejected; no scaling.",
            "invalid_src_bytes": "0.0.0.0 becomes missing numeric; rows retained.",
        },
        "invalid_src_bytes_count": sum(invalid_by_file.values()),
        "invalid_src_bytes_by_file": invalid_by_file,
        "split": {"seed": seed, "stratified_by": "type", "requested_ratios": [0.70, 0.15, 0.15],
                  "rounding": "Floor train and validation counts per type; remainder to test.",
                  "assignments": "split_assignments.npy: 0=train, 1=validation, 2=test, in manifest row order."},
        "splits": summaries, "training_binary_class_weights": class_weights,
        "preprocessing_steps": [
            "Verify schema and targets; construct seeded stratified split before learning.",
            "Strip whitespace; normalize missing markers and explicitly convert numeric/Boolean values.",
            "Keep 23 original predictors; exclude identifiers, sparse text and both targets from X.",
            "Learn imputation means and complete category vocabularies using training rows only.",
            "Fit StandardScaler on imputed training numerical columns only.",
            "Apply frozen transforms to all splits; retain original class distributions.",
            "Save config, split assignments, outputs and training binary class weights.",
        ],
        "limitations": [
            "Row-stratified split: duplicates and related flows can cross splits; no deduplication or group isolation.",
            "Category codes have no numerical distance/order meaning. A later model must treat them as categorical (e.g. embeddings).",
            "Ports remain numeric for this compact baseline; adversarial edits must respect discrete protocol semantics.",
        ],
        "raw_sha256_unchanged": True, "chunk_size": chunk_size,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "output_files": [str(path) for path in generated],
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    for filename, content in [("preprocessing_config.json", config), ("preprocessing_report.json", report)]:
        (artifact_dir / filename).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Done: {retained:,} rows, 23 model features, {report['runtime_seconds']} seconds.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "dataset" / "processed_zainab")
    parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.output_dir, args.artifact_dir, args.chunk_size, args.seed)


if __name__ == "__main__":
    main()
