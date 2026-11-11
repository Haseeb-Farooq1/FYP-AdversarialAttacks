"""
Step 1 of the neural-network pipeline: check the 23 pre-processed CSV files and
build a compact cache that the training code can load quickly.

What it does
------------
1. Finds Network_dataset_1 ... 23 in the pre_processed folder (numeric order).
2. Checks every file: same 25 columns in the same order, label is 0/1 and agrees
   with `type`, no missing/infinite predictor values, and categorical codes are
   ones the shared pre-processing pipeline could actually have produced.
3. Writes the 23 predictors (float32), label, attack type, file number and a
   per-row fingerprint (used later to keep duplicate rows in the same fold) into
   model_luqman/cache/ as .npy files. Nothing in the cache is ever committed.
4. Prints a per-file table and a summary, and saves cache/summary.json.

It is safe to stop and restart: finished files are skipped on the next run.

Usage
-----
    python model_luqman/prepare_data.py --max-files 2    # quick smoke test
    python model_luqman/prepare_data.py                  # all files
    python model_luqman/prepare_data.py --rebuild        # start again from zero
"""

import argparse
import json
import platform
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "pre_processed"
DEFAULT_CACHE_DIR = HERE / "cache"
DEFAULT_METADATA = PROJECT_ROOT / "pre_processing" / "feature_metadata.json"

FEATURES = [
    "src_port", "dst_port", "proto", "service", "duration", "src_bytes", "dst_bytes",
    "conn_state", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    "dns_qclass", "dns_qtype", "dns_rcode", "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "http_request_body_len", "http_response_body_len", "http_status_code",
]
LABEL, TYPE = "label", "type"
EXPECTED_COLUMNS = FEATURES + [LABEL, TYPE]
# Fixed order, so the numeric code of each attack type never depends on which files were read.
TYPE_NAMES = ["backdoor", "ddos", "dos", "injection", "mitm", "normal",
              "password", "ransomware", "scanning", "xss"]
TYPE_CODE = {name: i for i, name in enumerate(TYPE_NAMES)}


def find_files(data_dir: Path):
    found = {}
    for p in data_dir.glob("*.csv"):
        m = re.match(r"Network_dataset_(\d+)", p.name)
        if m:
            idx = int(m.group(1))
            if idx in found:
                sys.exit(f"Two files claim to be dataset {idx}: {found[idx].name} and {p.name}")
            found[idx] = p
    if not found:
        sys.exit(f"No Network_dataset_*.csv files found in {data_dir}")
    return [found[i] for i in sorted(found)]


def count_rows(path: Path) -> int:
    """Count data rows (lines minus the header) without parsing the file."""
    n, last = 0, b""
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 22)
            if not block:
                break
            n += block.count(b"\n")
            last = block[-1:]
    if last not in (b"\n", b""):
        n += 1
    return n - 1


def load_valid_codes(metadata_path: Path):
    if not metadata_path.exists():
        print(f"WARNING: {metadata_path} not found; skipping the category-code check.")
        return {}
    meta = json.load(open(metadata_path))
    valid = {}
    for f in meta["features"]:
        if "mapping" in f:
            codes = set(int(v) for v in f["mapping"].values())
            valid[f["feature"]] = codes | {0, 1} if f["feature_type"] == "categorical" else codes
    return valid


def open_arrays(cache_dir: Path, n_rows: int, create: bool):
    spec = {
        "X": (np.float32, (n_rows, len(FEATURES))),
        "y": (np.uint8, (n_rows,)),
        "t": (np.uint8, (n_rows,)),
        "f": (np.uint8, (n_rows,)),
        "h": (np.uint64, (n_rows,)),
    }
    mode = "w+" if create else "r+"
    return {k: np.lib.format.open_memmap(cache_dir / f"{k}.npy", mode=mode, dtype=dt, shape=sh)
            for k, (dt, sh) in spec.items()}


def process_file(path, idx, start, arrays, valid_codes):
    """Read one CSV, validate it, write it into the cache. Returns (summary, problems)."""
    problems = []
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    if header != EXPECTED_COLUMNS:
        missing = [c for c in EXPECTED_COLUMNS if c not in header]
        extra = [c for c in header if c not in EXPECTED_COLUMNS]
        return None, [f"{path.name}: columns differ from the expected 25 (missing {missing}, extra {extra}, "
                      f"or a different order)"]

    dtypes = {c: np.float32 for c in FEATURES}
    dtypes[LABEL] = np.int8
    dtypes[TYPE] = str
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=dtypes)
    n = len(df)

    type_clean = df[TYPE].astype(str).str.strip().str.lower()
    codes = type_clean.map(TYPE_CODE)
    if codes.isna().any():
        bad = sorted(type_clean[codes.isna()].unique())[:5]
        problems.append(f"{path.name}: unknown attack types {bad}")
        return None, problems
    codes = codes.to_numpy(dtype=np.uint8)

    y = df[LABEL].to_numpy()
    if not np.isin(y, [0, 1]).all():
        problems.append(f"{path.name}: label has values other than 0/1")
        return None, problems
    mismatch = int(((y == 0) != (codes == TYPE_CODE["normal"])).sum())
    if mismatch:
        problems.append(f"{path.name}: label disagrees with type in {mismatch:,} rows")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    bad_cells = ~np.isfinite(X)
    nonfinite = int(bad_cells.sum())
    if nonfinite:
        cols = [FEATURES[i] for i in np.where(bad_cells.any(axis=0))[0]]
        problems.append(f"{path.name}: {nonfinite:,} missing/infinite predictor values in {cols}")

    code_violations = {}
    for j, name in enumerate(FEATURES):
        if name in valid_codes:
            col = X[:, j]
            ok = np.isin(col, list(valid_codes[name]))
            if not ok.all():
                code_violations[name] = int((~ok).sum())
    if code_violations:
        problems.append(f"{path.name}: category codes outside the shared mapping: {code_violations}")

    fingerprint = pd.util.hash_pandas_object(df[FEATURES], index=False).to_numpy(dtype=np.uint64)

    end = start + n
    arrays["X"][start:end] = X
    arrays["y"][start:end] = y.astype(np.uint8)
    arrays["t"][start:end] = codes
    arrays["f"][start:end] = idx
    arrays["h"][start:end] = fingerprint
    for a in arrays.values():
        a.flush()

    type_counts = np.bincount(codes, minlength=len(TYPE_NAMES))
    top = int(type_counts.argmax())
    summary = {
        "file": path.name, "index": idx, "rows": n,
        "normal_rows": int((y == 0).sum()), "attack_rows": int((y == 1).sum()),
        "type_counts": {TYPE_NAMES[i]: int(c) for i, c in enumerate(type_counts) if c},
        "top_type": TYPE_NAMES[top], "top_type_share": float(type_counts[top] / n),
        "nonfinite_cells": nonfinite, "code_violations": code_violations,
    }
    return summary, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    ap.add_argument("--max-files", type=int, default=None, help="only use the first N files (smoke test)")
    ap.add_argument("--rebuild", action="store_true", help="delete the cache and start again")
    args = ap.parse_args()

    started = time.time()
    files = find_files(args.data_dir)
    if args.max_files:
        files = files[: args.max_files]
        args.cache_dir = args.cache_dir.parent / (args.cache_dir.name + "_smoke")
    print(f"Found {len(files)} files in {args.data_dir}")

    if args.rebuild and args.cache_dir.exists():
        shutil.rmtree(args.cache_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / ".gitignore").write_text("*\n")  # nothing in the cache is ever tracked by Git

    print("Counting rows ...")
    counts = [count_rows(p) for p in files]
    total = sum(counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    print(f"  {total:,} rows in total")

    progress_path = args.cache_dir / "progress.json"
    names = [p.name for p in files]
    fresh = True
    if progress_path.exists():
        prog = json.load(open(progress_path))
        if prog.get("names") == names and prog.get("counts") == counts:
            fresh = False
        else:
            sys.exit("The cache was built from different files. Run again with --rebuild.")
    if fresh:
        prog = {"names": names, "counts": counts, "done": {}}
    arrays = open_arrays(args.cache_dir, total, create=fresh)

    valid_codes = load_valid_codes(args.metadata)
    all_problems = []
    results = {}
    print()
    for k, (p, c, s) in enumerate(zip(files, counts, starts)):
        idx = int(re.match(r"Network_dataset_(\d+)", p.name).group(1))
        if p.name in prog["done"]:
            results[p.name] = prog["done"][p.name]
            print(f"[{k + 1}/{len(files)}] {p.name}: already checked and cached, skipped")
        else:
            t0 = time.time()
            summ, probs = process_file(p, idx, int(s), arrays, valid_codes)
            if summ is not None and summ["rows"] != c:
                probs.append(f"{p.name}: counted {c:,} lines but parsed {summ['rows']:,} rows")
            all_problems += probs
            if summ is None:
                print(f"[{k + 1}/{len(files)}] {p.name}: PROBLEM, see the list at the end")
                continue
            results[p.name] = summ
            if not probs:  # only a clean file is remembered as finished; a file with problems is re-checked next run
                prog["done"][p.name] = summ
                json.dump(prog, open(progress_path, "w"))
            print(f"[{k + 1}/{len(files)}] {p.name}: {summ['rows']:,} rows | "
                  f"normal {summ['normal_rows'] / summ['rows']:.1%} | "
                  f"top type {summ['top_type']} {summ['top_type_share']:.1%} | "
                  f"{'PROBLEMS' if probs else 'ok'} | {time.time() - t0:.0f}s")

    done = [results[n] for n in names if n in results]
    t_counts = np.zeros(len(TYPE_NAMES), dtype=np.int64)
    for d in done:
        for name, c in d["type_counts"].items():
            t_counts[TYPE_CODE[name]] += c
    rows_done = int(sum(d["rows"] for d in done))

    print("\nChecking duplicate rows (identical predictor values) ...")
    h = arrays["h"]
    _, grp = np.unique(h, return_counts=True)
    dup_groups = int((grp > 1).sum())
    in_dup = int(grp[grp > 1].sum())
    summary = {
        "passed": not all_problems and len(done) == len(files),
        "problems": all_problems,
        "files": done,
        "total_rows": rows_done,
        "type_counts": {TYPE_NAMES[i]: int(c) for i, c in enumerate(t_counts) if c},
        "normal_rows": int(sum(d["normal_rows"] for d in done)),
        "attack_rows": int(sum(d["attack_rows"] for d in done)),
        "files_with_one_type_above_90pct": int(sum(d["top_type_share"] > 0.9 for d in done)),
        "duplicates_in_predictor_space": {
            "groups_with_repeats": dup_groups, "rows_in_those_groups": in_dup,
            "extra_copies": in_dup - dup_groups, "share_of_rows_that_are_extra_copies": (in_dup - dup_groups) / max(rows_done, 1),
            "largest_group": int(grp.max()),
        },
        "type_names_in_code_order": TYPE_NAMES,
        "feature_order": FEATURES,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "platform": platform.platform()},
        "seconds": round(time.time() - started, 1),
    }
    json.dump(summary, open(args.cache_dir / "summary.json", "w"), indent=2)

    print("\n================ SUMMARY ================")
    print(f"Rows checked: {rows_done:,} in {len(done)} of {len(files)} files")
    print(f"Normal: {summary['normal_rows']:,} | Attack: {summary['attack_rows']:,}")
    for name, c in summary["type_counts"].items():
        print(f"  {name:<11}{c:>12,}  {c / rows_done:6.2%}")
    print(f"Files where one attack type is more than 90% of the rows: "
          f"{summary['files_with_one_type_above_90pct']} of {len(done)}")
    d = summary["duplicates_in_predictor_space"]
    print(f"Extra copies of identical predictor rows: {d['extra_copies']:,} "
          f"({d['share_of_rows_that_are_extra_copies']:.2%} of rows; largest group {d['largest_group']:,} rows)")
    if all_problems:
        print("\nPROBLEMS FOUND:")
        for p in all_problems:
            print("  -", p)
        print("\nCopy everything above and send it to Claude before doing anything else.")
        sys.exit(1)
    print(f"\nOK - all checks passed. Cache written to {args.cache_dir}  ({summary['seconds']}s)")


if __name__ == "__main__":
    main()
