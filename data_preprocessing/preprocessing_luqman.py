"""
ToN_IoT Network Dataset - Preprocessing Pipeline (Luqman's portion)
====================================================================
Handles Network_dataset_9.csv through Network_dataset_16.csv from the
UNSW ToN_IoT Processed_Network dataset.

Usage:
    python preprocessing_luqman.py --data_dir ./data/raw --out_dir ./data/processed

Output:
    - processed_train.csv
    - processed_test.csv
    - preprocessing_report.json   (summary stats, matches team's shared report format)
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FILE_RANGE = range(9, 17)  # Network_dataset_9.csv ... Network_dataset_16.csv

# Columns known to be simple identifiers / high-leakage risk for a generic
# classifier (raw IPs and raw timestamp let a model "memorise" the testbed
# instead of learning traffic behaviour). Dropped by default; keep them with
# --keep_ip_ts if you specifically want to study that effect later.
LEAKAGE_PRONE_COLS = ["src_ip", "dst_ip", "ts"]

TARGET_BINARY = "label"   # 0 = normal, 1 = attack
TARGET_MULTI = "type"     # normal / backdoor / ddos / dos / injection /
                           # mitm / password / ransomware / scanning / xss

# Columns that are categorical/string-typed in the raw ToN_IoT network CSVs.
CATEGORICAL_COLS = [
    "proto", "service", "conn_state",
    "dns_query", "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established",
    "ssl_subject", "ssl_issuer",
    "http_trans_depth", "http_method", "http_uri", "http_version",
    "http_user_agent", "http_orig_mime_types", "http_resp_mime_types",
    "weird_name", "weird_addl", "weird_notice",
]

# The raw files use "-" as their missing-value marker instead of a blank cell.
MISSING_MARKER = "-"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def load_files(data_dir: str, file_range=FILE_RANGE) -> pd.DataFrame:
    """Load and concatenate the assigned Network_dataset_N.csv files."""
    frames = []
    missing_files = []
    for i in file_range:
        fpath = Path(data_dir) / f"Network_dataset_{i}.csv"
        if not fpath.exists():
            missing_files.append(str(fpath))
            continue
        print(f"Loading {fpath.name} ...")
        df = pd.read_csv(fpath, low_memory=False, na_values=[MISSING_MARKER])
        frames.append(df)

    if missing_files:
        print("WARNING - the following expected files were not found:")
        for m in missing_files:
            print(f"   {m}")

    if not frames:
        raise FileNotFoundError(
            f"No Network_dataset_*.csv files found in {data_dir}. "
            "Download them from the UNSW ToN_IoT page first."
        )

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined shape before cleaning: {combined.shape}")
    return combined


def clean_data(df: pd.DataFrame, keep_ip_ts: bool = False) -> tuple[pd.DataFrame, dict]:
    """Drop duplicates, handle missing values, drop leakage-prone columns."""
    stats = {}
    stats["rows_before_dedup"] = len(df)

    df = df.drop_duplicates()
    stats["rows_after_dedup"] = len(df)
    stats["duplicates_removed"] = stats["rows_before_dedup"] - stats["rows_after_dedup"]

    if not keep_ip_ts:
        drop_cols = [c for c in LEAKAGE_PRONE_COLS if c in df.columns]
        df = df.drop(columns=drop_cols)
        stats["dropped_leakage_prone_columns"] = drop_cols

    # Missing values: numeric -> 0 (absence of a protocol field is
    # meaningful, e.g. no SSL handshake happened), categorical -> "none"
    missing_before = df.isna().sum()
    stats["missing_values_per_column_before_fill"] = {
        k: int(v) for k, v in missing_before[missing_before > 0].items()
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for target in (TARGET_BINARY, TARGET_MULTI):
        if target in numeric_cols:
            numeric_cols.remove(target)

    df[numeric_cols] = df[numeric_cols].fillna(0)

    cat_cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    df[cat_cols_present] = df[cat_cols_present].fillna("none")

    stats["final_shape_after_cleaning"] = list(df.shape)
    return df, stats


def encode_and_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns, scale numeric columns."""
    stats = {}
    encoders = {}

    cat_cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in cat_cols_present:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = len(le.classes_)
    stats["categorical_columns_encoded"] = encoders

    # Encode the multiclass target too, but keep a readable mapping.
    if TARGET_MULTI in df.columns:
        le_type = LabelEncoder()
        df[TARGET_MULTI + "_encoded"] = le_type.fit_transform(df[TARGET_MULTI].astype(str))
        stats["attack_type_class_mapping"] = {
            str(cls): int(code) for code, cls in enumerate(le_type.classes_)
        }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for target in (TARGET_BINARY, TARGET_MULTI + "_encoded"):
        if target in numeric_cols:
            numeric_cols.remove(target)

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    stats["numeric_columns_scaled"] = len(numeric_cols)

    return df, stats


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified split on attack type so rare classes appear in both sets."""
    stratify_col = df[TARGET_MULTI] if TARGET_MULTI in df.columns else df[TARGET_BINARY]
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=stratify_col
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess ToN_IoT files 9-16 (Luqman's portion)")
    parser.add_argument("--data_dir", default="./data/raw", help="Folder containing Network_dataset_*.csv")
    parser.add_argument("--out_dir", default="./data/processed", help="Where to write processed output")
    parser.add_argument("--keep_ip_ts", action="store_true", help="Keep src_ip/dst_ip/ts columns instead of dropping them")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()

    report = {"assigned_files": [f"Network_dataset_{i}.csv" for i in FILE_RANGE]}

    df = load_files(args.data_dir)
    report["raw_combined_shape"] = list(df.shape)

    df, clean_stats = clean_data(df, keep_ip_ts=args.keep_ip_ts)
    report["cleaning"] = clean_stats

    if TARGET_MULTI in df.columns:
        report["attack_type_distribution"] = df[TARGET_MULTI].value_counts().to_dict()
    if TARGET_BINARY in df.columns:
        report["binary_label_distribution"] = {
            str(k): int(v) for k, v in df[TARGET_BINARY].value_counts().to_dict().items()
        }

    df, encode_stats = encode_and_scale(df)
    report["encoding_and_scaling"] = encode_stats

    train_df, test_df = split_data(df, test_size=args.test_size)
    report["train_shape"] = list(train_df.shape)
    report["test_shape"] = list(test_df.shape)

    train_path = Path(args.out_dir) / "processed_train.csv"
    test_path = Path(args.out_dir) / "processed_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    report["output_files"] = [str(train_path), str(test_path)]
    report["runtime_seconds"] = round(time.time() - start, 2)
    report["processed_by"] = "Muhammad Luqman (files 9-16)"

    report_path = Path(args.out_dir) / "preprocessing_report_luqman.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDone in {report['runtime_seconds']}s")
    print(f"Train: {train_df.shape}  Test: {test_df.shape}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
