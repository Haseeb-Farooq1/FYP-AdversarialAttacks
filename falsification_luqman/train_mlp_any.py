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

DATASET = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "Raw_DATASET2"
QUICK = "--quick" in sys.argv
SEED = 42
EPOCHS = 2 if QUICK else 5
BATCH = 8192
LR = 1e-3

SPLIT = Path("data_split") / DATASET
OUT = Path("falsification_luqman/results") / DATASET
OUT.mkdir(parents=True, exist_ok=True)

class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def load(split, feats, limit=None):
    dfs = [pd.read_csv(p, dtype={"type": "category"})
           for p in sorted((SPLIT / split).glob("*.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    if limit:
        df = df.iloc[:limit]
    if feats is None:
        feats = [c for c in df.columns if c not in ("label", "type", "__time")]
    X = df[feats].to_numpy(np.float32)
    y = df["label"].to_numpy(np.float32)
    t = df["type"].astype(str).to_numpy()
    tm = df["__time"].to_numpy(np.int64)
    del df, dfs
    return X, y, t, tm, feats

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
    m = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "per_type_share_correct": {a: float((pred[t == a] == y[t == a]).mean())
                                    for a in np.unique(t)},
    }
    if len(np.unique(y)) == 2:
        m["roc_auc"] = float(roc_auc_score(y, score))
        m["pr_auc"] = float(average_precision_score(y, score))
    else:
        m["roc_auc"] = None
        m["pr_auc"] = None
        m["note"] = "single-class evaluation window, roc_auc/pr_auc undefined"
    return m

print("Dataset:", DATASET, flush=True)
Xtr, ytr, ttr, ttr_time, feats = load("train", None, 2_000_000 if QUICK else None)
print(f"train rows: {len(ytr)}, features: {len(feats)}", flush=True)

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
    m = metrics(ytr[va_idx], predict(model, mu, sd, Xtr[va_idx]), ttr[va_idx])
    cv.append(m)
    print(f"  fold {i} balanced_acc={m['balanced_accuracy']:.4f} "
          f"macro_f1={m['macro_f1']:.4f}", flush=True)

Xva, yva, tva, _, _ = load("validation", feats, 500_000 if QUICK else None)
Xte, yte, tte, _, _ = load("test", feats, 500_000 if QUICK else None)

print("Final fit on train + validation...", flush=True)
Xfull = np.concatenate([Xtr, Xva]); yfull = np.concatenate([ytr, yva])
model, mu, sd = fit(Xfull, yfull)
torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
            "features": feats}, OUT / "mlp_final.pt")

score = predict(model, mu, sd, Xte)
res = {"config": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "seed": SEED},
       "cv_folds": cv,
       "cv_mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in cv])),
       "test": metrics(yte, score, tte)}
with open(OUT / "mlp_results.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res["test"], indent=2))
print("DONE")