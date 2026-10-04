"""
ToN_IoT Network Dataset - Preprocessing Pipeline (Luqman's portion)
====================================================================
Handles Network_dataset_9.csv through Network_dataset_16.csv from the
UNSW ToN_IoT Processed_Network dataset.

v2: fixes a data-leakage bug from v1. The train/test split now happens
BEFORE any encoder or scaler is fitted, and those are fitted on the
training split only, then applied unchanged to the test split. The
fitted transformer is also saved to disk so the exact same transformation
can be reapplied later (e.g. to adversarially-generated traffic).

Usage:
    python preprocessing_luqman.py --data_dir ./data/raw --out_dir ./data/processed

Output:
    - processed_train.csv
    - processed_test.csv
    - fitted_preprocessor.joblib   (category mappings + scaler, for reuse)
    - preprocessing_report_luqman.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FILE_RANGE = range(9, 17)  # Network_dataset_9.csv ... Network_dataset_16.csv

LEAKAGE_PRONE_COLS = ["src_ip", "dst_ip", "ts"]

TARGET_BINARY = "label"
TARGET_MULTI = "type"

CATEGORICAL_COLS = [
    "proto", "service", "conn_state",
    "dns_query", "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established",
    "ssl_subject", "ssl_issuer",
    "http_trans_depth", "http_method", "http_uri", "http_version",
    "http_user_agent", "http_orig_mime_types", "http_resp_mime_types",
    "weird_name", "weird_addl", "weird_notice",
]

MISSING_MARKER = "-"
UNSEEN_CODE = -1  # reserved code for a category seen in test but never in train


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def load_files(data_dir: str, file_range=FILE_RANGE) -> pd.DataFrame:
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
    """Drop duplicates, handle missing values, drop leakage-prone columns.
    No statistics are learned from the data here (fills use fixed constants),
    so this step is safe to run before the train/test split."""
    stats = {}
    stats["rows_before_dedup"] = len(df)

    df = df.drop_duplicates()
    stats["rows_after_dedup"] = len(df)
    stats["duplicates_removed"] = stats["rows_before_dedup"] - stats["rows_after_dedup"]

    if not keep_ip_ts:
        drop_cols = [c for c in LEAKAGE_PRONE_COLS if c in df.columns]
        df = df.drop(columns=drop_cols)
        stats["dropped_leakage_prone_columns"] = drop_cols

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


def fit_transformers(train_df: pd.DataFrame) -> tuple[dict, dict]:
    """Learn category mappings and scaler parameters from the TRAINING split only."""
    stats = {}
    category_maps = {}

    cat_cols_present = [c for c in CATEGORICAL_COLS if c in train_df.columns]
    for col in cat_cols_present:
        uniques = sorted(train_df[col].astype(str).unique())
        category_maps[col] = {val: code for code, val in enumerate(uniques)}
    stats["categorical_columns_fitted"] = {k: len(v) for k, v in category_maps.items()}

    type_map = None
    if TARGET_MULTI in train_df.columns:
        uniques = sorted(train_df[TARGET_MULTI].astype(str).unique())
        type_map = {val: code for code, val in enumerate(uniques)}
        stats["attack_type_class_mapping"] = type_map

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    for target in (TARGET_BINARY,):
        if target in numeric_cols:
            numeric_cols.remove(target)

    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    stats["numeric_columns_fitted"] = numeric_cols

    transformer = {
        "category_maps": category_maps,
        "type_map": type_map,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols_present,
    }
    return transformer, stats


def apply_transformers(df: pd.DataFrame, transformer: dict) -> pd.DataFrame:
    """Apply already-fitted mappings/scaler to a split (train or test).
    Values not seen during fitting are mapped to UNSEEN_CODE rather than
    raising an error or silently refitting."""
    df = df.copy()

    for col, mapping in transformer["category_maps"].items():
        df[col] = df[col].astype(str).map(mapping).fillna(UNSEEN_CODE).astype(int)

    if transformer["type_map"] is not None and TARGET_MULTI in df.columns:
        df[TARGET_MULTI + "_encoded"] = (
            df[TARGET_MULTI].astype(str).map(transformer["type_map"]).fillna(UNSEEN_CODE).astype(int)
        )

    numeric_cols = transformer["numeric_cols"]
    df[numeric_cols] = transformer["scaler"].transform(df[numeric_cols])

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split BEFORE any encoder/scaler is fit, on the raw cleaned values."""
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
    parser.add_argument("--data_dir", default="./data/raw")
    parser.add_argument("--out_dir", default="./data/processed")
    parser.add_argument("--keep_ip_ts", action="store_true")
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

    # --- split FIRST, on cleaned but not-yet-encoded data ---
    train_df, test_df = split_data(df, test_size=args.test_size)
    report["train_shape_before_encoding"] = list(train_df.shape)
    report["test_shape_before_encoding"] = list(test_df.shape)

    # --- fit encoders/scaler on TRAIN ONLY ---
    transformer, fit_stats = fit_transformers(train_df)
    report["encoding_and_scaling_fitted_on_train_only"] = fit_stats

    # --- apply the frozen transformer to both splits ---
    train_df = apply_transformers(train_df, transformer)
    test_df = apply_transformers(test_df, transformer)

    report["train_shape"] = list(train_df.shape)
    report["test_shape"] = list(test_df.shape)

    train_path = Path(args.out_dir) / "processed_train.csv"
    test_path = Path(args.out_dir) / "processed_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # save the fitted transformer for reuse (e.g. on adversarial examples later)
    transformer_path = Path(args.out_dir) / "fitted_preprocessor.joblib"
    joblib.dump(transformer, transformer_path)

    report["output_files"] = [str(train_path), str(test_path), str(transformer_path)]
    report["runtime_seconds"] = round(time.time() - start, 2)
    report["processed_by"] = "Muhammad Luqman (files 9-16)"
    report["leakage_fix_applied"] = (
        "v2: encoder/scaler fitted on training split only, after the train/test split, "
        "then applied unchanged to the test split. No test-set information was used "
        "to fit any transformation."
    )

    report_path = Path(args.out_dir) / "preprocessing_report_luqman.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDone in {report['runtime_seconds']}s")
    print(f"Train: {train_df.shape}  Test: {test_df.shape}")
    print(f"Fitted transformer saved to {transformer_path}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
