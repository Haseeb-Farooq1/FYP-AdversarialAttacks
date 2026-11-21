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
K = 5
EPOCHS = 5
BATCH = 8192
LR = 1e-3
QUICK = "--quick" in sys.argv

base = list(Path(r"D:\Final Year Project").rglob("NF-UNSW-NB15-v3_clean.csv"))[0].parent
df = pd.read_csv(base / "NF-UNSW-NB15-v3_clean.csv")
folds = np.load(base / "NF-UNSW-NB15-v3_folds.npy")

feat_cols = [c for c in df.columns if c not in ("Label", "Attack")]
X = df[feat_cols].to_numpy(dtype=np.float32)
y = df["Label"].to_numpy(dtype=np.float32)
t = df["Attack"].astype(str).to_numpy()

if QUICK:
    rng = np.random.default_rng(SEED)
    keep = rng.choice(len(y), size=len(y) // 10, replace=False)
    keep.sort()
    X, y, t, folds = X[keep], y[keep], t[keep], folds[keep]
    print("QUICK MODE: using", len(y), "rows")

class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

torch.manual_seed(SEED)
oof = np.zeros(len(y), dtype=np.float32)
out_dir = Path("results_nfunswb15"); out_dir.mkdir(exist_ok=True)

for k in range(K):
    tr, te = folds != k, folds == k
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd

    pos = float(y[tr].sum())
    pos_weight = torch.tensor((len(y[tr]) - pos) / max(pos, 1.0))
    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xtr, ytr_t = torch.tensor(Xtr), torch.tensor(y[tr])

    for ep in range(EPOCHS):
        perm = torch.randperm(len(xtr))
        tot = 0.0
        for i in range(0, len(xtr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(xtr[idx]), ytr_t[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f"Fold {k} epoch {ep+1}/{EPOCHS} loss={tot/len(xtr):.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        p = [torch.sigmoid(model(torch.tensor(Xte[i:i+BATCH]))).numpy()
             for i in range(0, len(Xte), BATCH)]
        oof[te] = np.concatenate(p)
    torch.save(model.state_dict(), out_dir / f"model_fold{k}.pt")
    print(f"Fold {k} done", flush=True)

pred = (oof >= 0.5).astype(int)
res = {
    "rows": int(len(y)),
    "accuracy": float(accuracy_score(y, pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    "macro_f1": float(f1_score(y, pred, average="macro")),
    "roc_auc": float(roc_auc_score(y, oof)),
    "pr_auc": float(average_precision_score(y, oof)),
    "confusion_matrix": confusion_matrix(y, pred).tolist(),
    "per_attack_share_correct": {a: float((pred[t == a] == y[t == a]).mean()) for a in np.unique(t)},
}
print(json.dumps(res, indent=2))
with open(out_dir / "cv_results.json", "w") as f:
    json.dump(res, f, indent=2)
print("DONE")