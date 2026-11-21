from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

DATA = list(Path(r"D:\Final Year Project").rglob("NF-UNSW-NB15-v3_clean.csv"))[0]
OUT = DATA.with_name("NF-UNSW-NB15-v3_folds.npy")
SEED = 42
K = 5

print("Loading:", DATA)
df = pd.read_csv(DATA)
feat_cols = [c for c in df.columns if c not in ("Label", "Attack")]
X = df[feat_cols].to_numpy(dtype=np.float64)
y = df["Label"].to_numpy()
t = df["Attack"].astype(str).to_numpy()
print("Rows:", len(df), "| Features:", len(feat_cols))

# Group rows with identical features (keep them in the same fold)
xb = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()
group_id = pd.factorize(xb)[0]
n_groups = group_id.max() + 1
print("Identical-feature groups:", n_groups)

# Assign whole groups to folds: shuffle inside each attack type, then round-robin
rng = np.random.default_rng(SEED)
group_fold = np.full(n_groups, -1, dtype=np.int8)
for cat in np.unique(t):
    gids = np.unique(group_id[t == cat])
    gids = rng.permutation(gids)
    for i, g in enumerate(gids):
        group_fold[g] = i % K

fold_of_row = group_fold[group_id]
assert (group_fold >= 0).all(), "some group got no fold"

# Verification
print("\n===== FOLD SUMMARY =====")
total = len(df)
for k in range(K):
    m = fold_of_row == k
    print(f"Fold {k}: rows={int(m.sum())} ({100*m.mean():.1f}%) attack%={100*y[m].mean():.2f}")

folds_per_group = pd.Series(fold_of_row).groupby(group_id).nunique()
split_groups = int((folds_per_group > 1).sum())
print("Groups split across folds:", split_groups)

fp = hashlib.md5(np.ascontiguousarray(fold_of_row).tobytes()).hexdigest()[:16]
print("Fold fingerprint:", fp)

np.save(OUT, fold_of_row)
print("Saved:", OUT)
print("DONE")