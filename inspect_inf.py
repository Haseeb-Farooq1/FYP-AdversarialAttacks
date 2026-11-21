from pathlib import Path
import numpy as np
import pandas as pd

DATA = list(Path(r"D:\Final Year Project").rglob("NF-UNSW-NB15-v3.csv"))[0]
df = pd.read_csv(DATA)

num = df.select_dtypes(include=[np.number])
inf_per_col = np.isinf(num).sum()
inf_per_col = inf_per_col[inf_per_col > 0]
print("===== Columns containing infinite values =====")
print(inf_per_col.to_string())

print("\n===== Same for NaN (top 10) =====")
nan_per_col = df.isna().sum().sort_values(ascending=False)
print(nan_per_col.head(10).to_string())

print("\n===== Rows affected by inf or NaN =====")
bad = np.isinf(num).any(axis=1) | df.isna().any(axis=1)
print("Rows with any inf/NaN:", int(bad.sum()), "out of", len(df))
print("Their Label distribution:")
print(df.loc[bad, "Label"].value_counts().to_string())
print("Their Attack distribution:")
print(df.loc[bad, "Attack"].value_counts().head(10).to_string())
print("\nDONE")