from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             average_precision_score)

SEED = 42
EPOCHS = 5
BATCH = 8192
LR = 1e-3
QUICK = "--quick" in sys.argv
if QUICK:
    EPOCHS = 2

SPLIT = Path("data_split/Raw_DATASET2")
OUT = Path("falsification_luqman/results")
OUT.mkdir(parents=True, exist_ok=True)

FEATS = ["src_port","dst_port","proto","service","duration","src_bytes","dst_bytes",
         "conn_state","missed_bytes","src_pkts","src_ip_bytes","dst_pkts","dst_ip_bytes",
         "dns_qclass","dns_qtype","dns_rcode","dns_AA","dns_RD","dns_RA","dns_rejected",
         "http_request_body_len","http_response_body_len","http_status_code"]

class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def load(split, limit=None):
    dfs = [pd.read_csv(p, dtype={"type": "category"})
           for p in sorted((SPLIT / split).glob("*.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    if limit:
        df = df.iloc[:limit]
    X = df[FEATS].to_numpy(np.float32)
    y = df["label"].to_numpy(np.float32)
    t = df["type"].astype(str).to_numpy()
    tm = df["__time"].to_numpy(np.int64)
    del df, dfs
    return X, y, t, tm

def fit(Xtr, ytr):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    Xtr = (Xtr - mu) / sd
    pos = float(ytr.sum())
    pos_weight = torch.tensor((len(ytr) - pos) / max(pos, 1.0))
    model = MLP(Xtr.shape[1])
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(xt[idx]), yt[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f"  epoch {ep+1}/{EPOCHS} loss={tot/len(xt):.4f}", flush=True)
    return model, mu, sd

def predict(model, mu, sd, X):
    model.eval()
    Xs = (X - mu) / sd
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(
            torch.tensor(Xs[i:i+BATCH]))).numpy()
            for i in range(0, len(Xs), BATCH)])

def metrics(y, score, t):
    pred = (score >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "per_type_share_correct": {a: float((pred[t == a] == y[t == a]).mean())
                                    for a in np.unique(t)},
    }

print("Loading train split...", flush=True)
Xtr, ytr, ttr, ttr_time = load("train", 2_000_000 if QUICK else None)
print(f"train rows: {len(ytr)}", flush=True)

# Expanding-window chronological CV on train (warm-up block + 5 forward windows)
order = np.argsort(ttr_time, kind="stable")
blocks = np.array_split(order, 6)
cv = []
for i in range(1, 6):
    if QUICK and i > 2:
        break
    tr_idx = np.concatenate(blocks[:i])
    va_idx = blocks[i]
    print(f"CV fold {i}: train={len(tr_idx)} val={len(va_idx)}", flush=True)
    model, mu, sd = fit(Xtr[tr_idx], ytr[tr_idx])
    score = predict(model, mu, sd, Xtr[va_idx])
    m = metrics(ytr[va_idx], score, ttr[va_idx])
    cv.append(m)
    print(f"  fold {i} balanced_acc={m['balanced_accuracy']:.4f} "
          f"macro_f1={m['macro_f1']:.4f}", flush=True)

print("Loading validation + test...", flush=True)
Xva, yva, tva, _ = load("validation", 500_000 if QUICK else None)
Xte, yte, tte, _ = load("test", 500_000 if QUICK else None)

print("Final fit on train + validation...", flush=True)
Xfull = np.concatenate([Xtr, Xva]); yfull = np.concatenate([ytr, yva])
model, mu, sd = fit(Xfull, yfull)
torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd},
           OUT / "mlp_final.pt")

print("Evaluating on test (once)...", flush=True)
score = predict(model, mu, sd, Xte)
res = {
    "config": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "seed": SEED},
    "cv_folds": cv,
    "cv_mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in cv])),
    "test": metrics(yte, score, tte),
}
with open(OUT / "mlp_results.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res["test"], indent=2))
print("DONE")