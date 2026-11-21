from pathlib import Path
import numpy as np
import pandas as pd

DATA = list(Path(r"D:\Final Year Project").rglob("NF-UNSW-NB15-v3.csv"))[0]
print("Loading:", DATA)
df = pd.read_csv(DATA)

print("\nTotal rows:", len(df))
print("Total columns:", len(df.columns))

print("\n===== Label counts (0=normal, 1=attack) =====")
print(df["Label"].value_counts(dropna=False).to_string())

print("\n===== Attack category counts =====")
print(df["Attack"].value_counts(dropna=False).to_string())

print("\n===== Label vs Attack consistency =====")
print(pd.crosstab(df["Attack"], df["Label"]).to_string())

print("\n===== Missing values (only columns that have any) =====")
miss = df.isna().sum()
miss = miss[miss > 0]
print(miss.to_string() if len(miss) else "None")

num = df.select_dtypes(include=[np.number])
print("\nInfinite values in numeric columns:", int(np.isinf(num).sum().sum()))

feat_cols = [c for c in df.columns if c not in ("Label", "Attack")]
print("Fully duplicate rows:", int(df.duplicated().sum()))
print("Duplicate feature rows (same features, ignoring labels):", int(df.duplicated(subset=feat_cols).sum()))
print("\nDONE")