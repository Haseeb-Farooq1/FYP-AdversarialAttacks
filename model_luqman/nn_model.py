"""Shared pieces: the neural network, input loading, prediction and metrics."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from prepare_data import FEATURES, TYPE_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA = PROJECT_ROOT / "pre_processing" / "feature_metadata.json"


def cardinalities(metadata_path=DEFAULT_METADATA):
    """Number of distinct codes (0..max) for each code-type predictor, from the shared pipeline's mapping."""
    meta = json.load(open(metadata_path))
    return {f["feature"]: max(max(int(v) for v in f["mapping"].values()), 1) + 1
            for f in meta["features"] if "mapping" in f}


def feature_split(card):
    cat_idx = [i for i, f in enumerate(FEATURES) if f in card]
    num_idx = [i for i, f in enumerate(FEATURES) if f not in card]
    return cat_idx, num_idx


class MLP(nn.Module):
    """Code columns -> small embeddings; numeric columns used as they are; then 3 fully connected layers."""

    def __init__(self, cards, n_num, emb_dim=8, h1=128, h2=64):
        super().__init__()
        self.cards = list(cards)
        self.embs = nn.ModuleList([nn.Embedding(c, min(emb_dim, max(2, c // 2 + 1))) for c in self.cards])
        d = sum(e.embedding_dim for e in self.embs) + n_num
        self.net = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))

    def forward(self, xnum, xcat):
        z = [e(xcat[:, i]) for i, e in enumerate(self.embs)]
        return self.net(torch.cat(z + [xnum], dim=1)).squeeze(1)


def load_inputs(cache_dir, cat_idx, num_idx, rows=None, block=2_000_000):
    """Read the cached predictors into two compact arrays (numeric float32, codes int16)."""
    X = np.load(Path(cache_dir) / "X.npy", mmap_mode="r")
    n = len(X) if rows is None else len(rows)
    xnum = np.empty((n, len(num_idx)), dtype=np.float32)
    xcat = np.empty((n, len(cat_idx)), dtype=np.int16)
    for s in range(0, n, block):
        blk = np.asarray(X[s:s + block] if rows is None else X[rows[s:s + block]])
        xnum[s:s + len(blk)] = blk[:, num_idx]
        xcat[s:s + len(blk)] = blk[:, cat_idx].astype(np.int16)
    return xnum, xcat


@torch.no_grad()
def predict_proba(model, xnum, xcat, idx, batch=65536):
    model.eval()
    out = np.empty(len(idx), dtype=np.float32)
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        logit = model(torch.from_numpy(xnum[b]), torch.from_numpy(xcat[b]).long())
        out[s:s + len(b)] = torch.sigmoid(logit).numpy()
    return out


def binary_metrics(y, prob, thr=0.5):
    """Positive class = attack (label 1). Rows of the confusion matrix = true [0, 1], columns = predicted [0, 1]."""
    pred, yt = prob >= thr, y.astype(bool)
    tp, tn = int((pred & yt).sum()), int((~pred & ~yt).sum())
    fp, fn = int((pred & ~yt).sum()), int((~pred & yt).sum())
    n = tp + tn + fp + fn

    def prf(a, b, c):  # true positives, false positives, false negatives
        p = a / (a + b) if a + b else 0.0
        r = a / (a + c) if a + c else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    pa, ra, fa = prf(tp, fp, fn)
    pn, rn, fn_ = prf(tn, fn, fp)
    sa, sn = tp + fn, tn + fp
    w = lambda a, b: (a * sa + b * sn) / n
    out = {
        "n": n, "accuracy": (tp + tn) / n, "balanced_accuracy": (ra + rn) / 2,
        "attack": {"precision": pa, "recall": ra, "f1": fa, "support": sa},
        "normal": {"precision": pn, "recall": rn, "f1": fn_, "support": sn},
        "macro": {"precision": (pa + pn) / 2, "recall": (ra + rn) / 2, "f1": (fa + fn_) / 2},
        "weighted": {"precision": w(pa, pn), "recall": w(ra, rn), "f1": w(fa, fn_)},
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "roc_auc": None, "pr_auc": None,
    }
    if 0 < sa < n:
        out["roc_auc"] = float(roc_auc_score(yt, prob))
        out["pr_auc"] = float(average_precision_score(yt, prob))
    return out


def per_type_recall(t, prob, thr=0.5):
    """Share of each attack type caught (and of normal traffic correctly passed)."""
    pred = prob >= thr
    res = {}
    for c in np.unique(t):
        m = t == c
        name = TYPE_NAMES[c]
        res[name] = {"rows": int(m.sum()),
                     "correct_share": float((~pred[m]).mean() if name == "normal" else pred[m].mean())}
    return res
