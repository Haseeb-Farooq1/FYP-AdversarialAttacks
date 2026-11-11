"""
Step 4: measure speed and size, and write the Model Evaluation Brief record (one JSON file).

Reads results/cv_results.json and results/model_fold0.pt (made by train_cv.py).
Speed is measured the way the Brief asks: 5 warm-up passes, then 10 timed passes over a fixed set
of rows (the first --timing-rows rows of fold 0's held-out part), batch size 8192.

Usage:  python model_luqman/brief_report.py          (add --quick to read the quick-run results)
"""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from nn_model import MLP, feature_split, cardinalities, load_inputs
from prepare_data import DEFAULT_CACHE_DIR, FEATURES

HERE = Path(__file__).resolve().parent


def sha(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--timing-rows", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--hardware-notes", default="AMD Ryzen 7 4700U, 8 logical processors, 7.4 GB RAM, "
                                                "AMD Radeon integrated graphics (not used), laptop on mains power")
    args = ap.parse_args()
    rdir = HERE / ("results_quick" if args.quick else "results")
    res = json.load(open(rdir / "cv_results.json"))
    pooled = res["pooled_out_of_fold"]
    model_path = rdir / "model_fold0.pt"

    # model load time (median of 5 loads)
    loads = []
    for _ in range(5):
        t0 = time.perf_counter()
        ck = torch.load(model_path)
        model = MLP(ck["cards"], ck["n_num"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        loads.append(time.perf_counter() - t0)

    # fixed rows for timing
    folds = np.load(args.cache_dir / "folds.npy")
    idx = np.flatnonzero(folds == 0)[: args.timing_rows]
    xnum, xcat = load_inputs(args.cache_dir, ck["cat_idx"], ck["num_idx"], rows=idx)
    tn, tc = torch.from_numpy(xnum), torch.from_numpy(xcat).long()
    n = len(idx)

    def one_pass():
        t0 = time.perf_counter()
        with torch.no_grad():
            for s in range(0, n, args.batch_size):
                model(tn[s:s + args.batch_size], tc[s:s + args.batch_size])
        return time.perf_counter() - t0

    for _ in range(5):
        one_pass()
    passes = np.array([one_pass() for _ in range(10)])
    lat_ms = passes / n * 1000.0

    params = int(sum(p.numel() for p in model.parameters()))
    emb_params = int(sum(p.numel() for e in model.embs for p in e.parameters()))
    a, nm = pooled["attack"], pooled["normal"]
    record = {
        "schema_version": "1.0",
        "evaluation_status": "partial" if res["quick_mode"] else "complete",
        "model": {
            "member": "Luqman", "model_name": "MLP with embeddings (PyTorch), binary normal-vs-attack",
            "version_or_commit": res.get("commit"), "artifact": str(model_path.relative_to(HERE.parent)),
            "inference_entry_point": "nn_model.MLP.forward(xnum, xcat); nn_model.predict_proba(...)",
            "embedded_preprocessing": "11 code-type columns are passed through learned embeddings; the 12 numeric "
                                      "columns are used as delivered by the shared pre-processing pipeline",
        },
        "dataset": {
            "filename_or_id": "pre_processed/Network_dataset_1..23.csv (shared pre-processing pipeline), pooled",
            "version_or_hash": f"sha256 of row fingerprints (cache/h.npy), first 16 chars: {sha(args.cache_dir / 'h.npy')}",
            "rows_evaluated": res["rows_evaluated"], "feature_count": len(FEATURES),
            "feature_order_or_schema": ", ".join(FEATURES), "target_column": "label",
            "label_mapping_and_order": ["0 = normal", "1 = attack"], "positive_class": "1 (attack)",
            "fixed_test_split_id": f"5-fold, duplicate-safe, stratified by attack type, seed {res['config']['seed']}, "
                                   f"fold fingerprint {res.get('fold_fingerprint')}; metrics pooled over the 5 held-out folds",
        },
        "environment": {
            "os": platform.platform(), "runtime_and_version": f"Python {platform.python_version()}",
            "framework_and_version": f"PyTorch {torch.__version__} (CPU build), NumPy {np.__version__}",
            "device": f"CPU: {platform.processor()}", "hardware_notes": args.hardware_notes,
        },
        "performance": {
            "accuracy": pooled["accuracy"],
            "precision": {"positive_class": a["precision"], "macro": pooled["macro"]["precision"], "weighted": pooled["weighted"]["precision"]},
            "recall": {"positive_class": a["recall"], "macro": pooled["macro"]["recall"], "weighted": pooled["weighted"]["recall"]},
            "f1": {"positive_class": a["f1"], "macro": pooled["macro"]["f1"], "weighted": pooled["weighted"]["f1"]},
            "roc_auc": {"value": pooled["roc_auc"], "averaging": "binary, attack as positive class",
                        "score_source": "sigmoid of the network output, pooled out-of-fold"},
            "pr_auc": {"value": pooled["pr_auc"], "averaging": "binary, attack as positive class",
                       "score_source": "sigmoid of the network output, pooled out-of-fold"},
            "per_class": [
                {"label": "0 (normal)", "precision": nm["precision"], "recall": nm["recall"], "f1": nm["f1"], "support": nm["support"]},
                {"label": "1 (attack)", "precision": a["precision"], "recall": a["recall"], "f1": a["f1"], "support": a["support"]},
            ],
            "confusion_matrix": {"labels_in_order": [0, 1], "matrix": pooled["confusion_matrix"]},
        },
        "timing": {
            "model_load_seconds": float(np.median(loads)),
            "training_seconds": res["total_training_seconds"],
            "inference": {
                "warmup_passes": 5, "timed_passes": 10, "batch_size": args.batch_size,
                "mean_latency_ms_per_example": float(lat_ms.mean()),
                "p50_latency_ms_per_example": float(np.percentile(lat_ms, 50)),
                "p95_latency_ms_per_example": float(np.percentile(lat_ms, 95)),
                "throughput_examples_per_second": float(n / passes.mean()),
                "measurement_notes": f"Fixed set of {n:,} rows (first rows of fold 0's held-out part, not the full "
                                     f"test set). Latency per example = pass time / rows; mean, p50 and p95 are taken "
                                     f"over the 10 timed passes. Data already in memory. training_seconds is the sum "
                                     f"of 5 fold trainings.",
            },
        },
        "complexity": {
            "parameter_count": params, "model_file_size_bytes": os.path.getsize(model_path),
            "architecture": "11 embedding tables + concat with 12 numeric inputs -> Linear 128 -> ReLU -> Linear 64 -> ReLU -> Linear 1",
            "architecture_counts": {"embedding_tables": len(model.embs), "embedding_parameters": emb_params,
                                    "linear_layers": 3, "hidden_units": [128, 64]},
            "peak_inference_memory_bytes": None, "memory_measurement_method": None,
        },
        "limitations_or_failure_reason": [
            "The shared pre-processing was fitted on all 23 files, including rows that are test rows here; a final evaluation should split first and refit on training rows only.",
            "Folds keep rows with identical predictor values together, but similar (not identical) flows from the same attack session can still sit in different folds.",
            "The data is stored sorted by attack type and some types are rare (mitm: 1,052 rows), so per-type results are noisy.",
            "A fixed decision threshold of 0.5 was used; no tuning.",
            "Inference timing used a fixed subset, not the full test set; peak memory was not measured.",
            "This measures clean accuracy, not adversarial robustness.",
        ],
        "notes": [f"epochs per fold: {res['config']['epochs']}", f"commit: {res.get('commit')}",
                  "Metrics are pooled over the 5 held-out folds, so every row is predicted exactly once."],
    }
    out_path = rdir / "brief_luqman.json"
    json.dump(record, open(out_path, "w"), indent=2)
    print(f"Parameters: {params:,} | model file: {os.path.getsize(model_path):,} bytes")
    print(f"Model load: {np.median(loads) * 1000:.1f} ms | latency: mean {lat_ms.mean():.5f} ms/example, p95 {np.percentile(lat_ms, 95):.5f} | "
          f"throughput {n / passes.mean():,.0f} examples/s")
    print(f"Saved the Brief record to {out_path}")


if __name__ == "__main__":
    main()
