"""
Step 3: train the neural network with 5-fold, duplicate-safe cross-validation on all 23 files.

Each fold: train on 4/5 of the rows for --epochs epochs (minimum 5 for the real run), then
predict the held-out 1/5. Every row is predicted exactly once, so the pooled result covers
the whole dataset. Saves results/cv_results.json and one model file per fold.

Usage
-----
    python model_luqman/train_cv.py --quick      # ~2 minutes: tiny test run to check everything works
    python model_luqman/train_cv.py              # the real run (5 folds x 5 epochs)
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nn_model import (DEFAULT_METADATA, MLP, PROJECT_ROOT, binary_metrics, cardinalities, feature_split,
                      load_inputs, per_type_recall, predict_proba)
from prepare_data import DEFAULT_CACHE_DIR, FEATURES, TYPE_NAMES

HERE = Path(__file__).resolve().parent
N_FOLDS = 5


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="tiny test run on a small random subset")
    ap.add_argument("--quick-frac", type=float, default=0.02)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 2)
    results_dir = args.results_dir or HERE / ("results_quick" if args.quick else "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    y = np.load(args.cache_dir / "y.npy")
    t = np.load(args.cache_dir / "t.npy")
    folds = np.load(args.cache_dir / "folds.npy")
    n = len(y)
    keep = np.ones(n, dtype=bool)
    if args.quick:
        keep = np.random.default_rng(args.seed).random(n) < args.quick_frac
        print(f"QUICK MODE: using {keep.sum():,} of {n:,} rows, {args.epochs} epochs. Not a real result.")

    card = cardinalities(args.metadata)
    cat_idx, num_idx = feature_split(card)
    cards = [card[FEATURES[i]] for i in cat_idx]
    print(f"Loading inputs: {len(num_idx)} numeric columns + {len(cat_idx)} code columns ...")
    xnum, xcat = load_inputs(args.cache_dir, cat_idx, num_idx)
    print(f"Loaded {n:,} rows in {time.time() - t_start:.0f}s | CPU threads: {torch.get_num_threads()}")

    oof = np.full(n, np.nan, dtype=np.float32)
    fold_results = []
    for k in range(N_FOLDS):
        torch.manual_seed(args.seed + k)
        rng = np.random.default_rng(args.seed + k)
        tr = np.flatnonzero((folds != k) & keep)
        te = np.flatnonzero((folds == k) & keep)
        n0 = int((y[tr] == 0).sum()); n1 = len(tr) - n0
        w = torch.tensor([len(tr) / (2 * max(n0, 1)), len(tr) / (2 * max(n1, 1))], dtype=torch.float32)  # "balanced"
        model = MLP(cards, len(num_idx))
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        n_batches = (len(tr) + args.batch_size - 1) // args.batch_size
        print(f"\n=== Fold {k + 1}/{N_FOLDS}: train {len(tr):,} rows, test {len(te):,} rows ===")
        t_train = time.time()
        for ep in range(args.epochs):
            model.train()
            perm = rng.permutation(tr)
            run, t_ep = [], time.time()
            for bi, s in enumerate(range(0, len(perm), args.batch_size)):
                b = np.sort(perm[s:s + args.batch_size])
                yb = torch.from_numpy(y[b]).float()
                logit = model(torch.from_numpy(xnum[b]), torch.from_numpy(xcat[b]).long())
                wt = w[yb.long()]
                loss = (bce(logit, yb) * wt).sum() / wt.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                run.append(loss.item())
                if (bi + 1) % max(n_batches // 4, 1) == 0 or bi + 1 == n_batches:
                    done = (bi + 1) / n_batches
                    eta = (time.time() - t_ep) / done * (1 - done)
                    print(f"  epoch {ep + 1}/{args.epochs}  {done:4.0%}  loss {np.mean(run[-200:]):.4f}  "
                          f"(about {eta / 60:.1f} min left in this epoch)")
        train_seconds = time.time() - t_train

        prob = predict_proba(model, xnum, xcat, te)
        oof[te] = prob
        m = binary_metrics(y[te], prob)
        m.update({"fold": k, "train_rows": int(len(tr)), "test_rows": int(len(te)), "train_seconds": train_seconds})
        fold_results.append(m)
        print(f"  fold {k + 1}: accuracy {m['accuracy']:.4f} | balanced acc {m['balanced_accuracy']:.4f} | "
              f"macro F1 {m['macro']['f1']:.4f} | attack recall {m['attack']['recall']:.4f} | "
              f"normal recall {m['normal']['recall']:.4f} | trained in {train_seconds / 60:.1f} min")
        torch.save({"state_dict": model.state_dict(), "cards": cards, "n_num": len(num_idx),
                    "cat_idx": cat_idx, "num_idx": num_idx}, results_dir / f"model_fold{k}.pt")
        del model, opt

    ok = ~np.isnan(oof)
    pooled = binary_metrics(y[ok], oof[ok])
    by_type = per_type_recall(t[ok], oof[ok])
    keys = ["accuracy", "balanced_accuracy"]
    spread = {k_: {"mean": float(np.mean([f[k_] for f in fold_results])), "std": float(np.std([f[k_] for f in fold_results]))}
              for k_ in keys}
    for grp, k_ in [("macro", "f1"), ("attack", "recall"), ("normal", "recall"), ("attack", "f1")]:
        spread[f"{grp}_{k_}"] = {"mean": float(np.mean([f[grp][k_] for f in fold_results])),
                                 "std": float(np.std([f[grp][k_] for f in fold_results]))}
    out = {
        "quick_mode": args.quick, "commit": git_commit(),
        "config": {"model": "MLP with embeddings", "epochs": args.epochs, "batch_size": args.batch_size,
                   "learning_rate": args.lr, "optimizer": "Adam", "loss": "weighted BCE (balanced class weights)",
                   "seed": args.seed, "folds": N_FOLDS, "threshold": 0.5,
                   "hidden_units": [128, 64], "embedding_cardinalities": dict(zip([FEATURES[i] for i in cat_idx], cards))},
        "rows_total": int(n), "rows_evaluated": int(ok.sum()),
        "fold_fingerprint": json.load(open(args.cache_dir / "folds_summary.json")).get("fingerprint")
        if (args.cache_dir / "folds_summary.json").exists() else None,
        "pooled_out_of_fold": pooled, "per_attack_type": by_type,
        "fold_mean_std": spread, "folds": fold_results,
        "total_training_seconds": float(sum(f["train_seconds"] for f in fold_results)),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform(), "threads": torch.get_num_threads()},
        "seconds_total": round(time.time() - t_start, 1),
    }
    json.dump(out, open(results_dir / "cv_results.json", "w"), indent=2)

    print("\n================ POOLED RESULT (every row predicted once) ================")
    print(f"Rows evaluated: {out['rows_evaluated']:,}")
    print(f"Accuracy {pooled['accuracy']:.4f} | Balanced accuracy {pooled['balanced_accuracy']:.4f} | "
          f"Macro F1 {pooled['macro']['f1']:.4f}")
    print(f"Attack: precision {pooled['attack']['precision']:.4f} recall {pooled['attack']['recall']:.4f} f1 {pooled['attack']['f1']:.4f}")
    print(f"Normal: precision {pooled['normal']['precision']:.4f} recall {pooled['normal']['recall']:.4f} f1 {pooled['normal']['f1']:.4f}")
    print(f"ROC-AUC {pooled['roc_auc']} | PR-AUC {pooled['pr_auc']}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {pooled['confusion_matrix']}")
    print("Share correct by traffic type:")
    for name, d in by_type.items():
        print(f"  {name:<11}{d['rows']:>12,}  {d['correct_share']:.4f}")
    print(f"Fold-to-fold spread (mean +/- std): macro F1 {spread['macro_f1']['mean']:.4f} +/- {spread['macro_f1']['std']:.4f}")
    print(f"\nDone in {out['seconds_total'] / 60:.1f} min. Saved to {results_dir / 'cv_results.json'}")


if __name__ == "__main__":
    main()
