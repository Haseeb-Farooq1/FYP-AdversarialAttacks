"""
Step 2: split the pooled dataset into 5 folds that keep duplicate rows together.

Why this matters
----------------
About 23% of rows are exact copies of another row's 23 predictor values. If a copy
sits in the training part and its twin in the test part, the model is graded on
something it has already seen. Here, every group of identical rows goes to ONE fold,
and each attack type is spread evenly over the 5 folds.

Deterministic: seed 42, rows in file order 1..23. The same data always gives the same
folds, so teammates can reproduce them. Output: cache/folds.npy (one number 0-4 per row).

Usage:  python model_luqman/make_folds.py
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from prepare_data import DEFAULT_CACHE_DIR, TYPE_NAMES

N_FOLDS = 5


def build_folds(h, t, y, seed=42, n_folds=N_FOLDS):
    n = len(h)
    rng = np.random.default_rng(seed)
    _, first, inv, size = np.unique(h, return_index=True, return_inverse=True, return_counts=True)
    group_type = t[first]                                  # a group's type = type of its first row
    group_fold = np.zeros(len(size), dtype=np.uint8)
    for c in np.unique(group_type):                        # spread every attack type evenly
        g = np.flatnonzero(group_type == c)
        g = g[rng.permutation(len(g))]                     # random order of groups
        cum = np.cumsum(size[g])
        mid = cum - size[g] / 2.0                          # centre of each group in the running row count
        group_fold[g] = np.minimum((mid / cum[-1] * n_folds).astype(np.int64), n_folds - 1)
    folds = group_fold[inv.reshape(-1)]
    # label conflicts: identical predictors but different labels
    ones = np.bincount(inv.reshape(-1), weights=y.astype(np.float64), minlength=len(size))
    conflicting = int(((ones > 0) & (ones < size)).sum())
    return folds.astype(np.uint8), {"groups": int(len(size)), "conflicting_label_groups": conflicting}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    h = np.load(args.cache_dir / "h.npy")
    t = np.load(args.cache_dir / "t.npy")
    y = np.load(args.cache_dir / "y.npy")
    print(f"Loaded {len(h):,} rows. Building {N_FOLDS} duplicate-safe, stratified folds ...")
    folds, info = build_folds(h, t, y, seed=args.seed)

    # verify: no group of identical rows is split across folds
    order = np.argsort(h, kind="stable")
    hs, fs = h[order], folds[order]
    same = hs[1:] == hs[:-1]
    leaks = int((same & (fs[1:] != fs[:-1])).sum())

    np.save(args.cache_dir / "folds.npy", folds)
    digest = hashlib.sha256(folds.tobytes()).hexdigest()[:16]

    print(f"\n{'fold':<6}{'rows':>12}{'normal':>11}{'normal %':>10}")
    per_fold = []
    for k in range(N_FOLDS):
        m = folds == k
        nm = int((t[m] == TYPE_NAMES.index("normal")).sum())
        per_fold.append({"fold": k, "rows": int(m.sum()), "normal_rows": nm,
                         "type_counts": {TYPE_NAMES[c]: int((t[m] == c).sum()) for c in np.unique(t)}})
        print(f"{k:<6}{m.sum():>12,}{nm:>11,}{nm / m.sum():>10.2%}")
    print(f"\nIdentical-row groups: {info['groups']:,} | split across folds: {leaks} (must be 0)")
    print(f"Groups with identical predictors but different labels: {info['conflicting_label_groups']:,}")
    print(f"Fold fingerprint (for comparing with teammates): {digest}")

    summary = {"seed": args.seed, "n_folds": N_FOLDS, "fingerprint": digest, "split_groups": leaks,
               **info, "folds": per_fold, "seconds": round(time.time() - t0, 1)}
    json.dump(summary, open(args.cache_dir / "folds_summary.json", "w"), indent=2)
    if leaks:
        raise SystemExit("LEAK: identical rows ended up in different folds. Send this to Claude.")
    print(f"\nOK - folds saved to {args.cache_dir / 'folds.npy'}  ({summary['seconds']}s)")


if __name__ == "__main__":
    main()
