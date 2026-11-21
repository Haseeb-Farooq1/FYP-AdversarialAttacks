from pathlib import Path
import numpy as np
import pandas as pd

DATA = list(Path(r"D:\Final Year Project").rglob("NF-UNSW-NB15-v3.csv"))[0]
OUT = DATA.with_name("NF-UNSW-NB15-v3_clean.csv")

print("Loading:", DATA)
df = pd.read_csv(DATA)

# 1. Drop identifiers and raw timestamps (keep FLOW_DURATION_MILLISECONDS)
drop = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR",
        "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]
df = df.drop(columns=drop)

# 2. Replace infinite values with missing, then fill missing with 0
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
before_inf = int(np.isinf(df[num_cols].to_numpy()).sum())
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
before_nan = int(df[num_cols].isna().sum().sum())
df[num_cols] = df[num_cols].fillna(0)

# 3. Validation before saving
assert not np.isinf(df[num_cols].to_numpy()).any(), "still has infinite values"
assert not df[num_cols].isna().any().any(), "still has missing values"
assert set(df["Label"].unique()) <= {0, 1}, "bad Label values"
assert df["Attack"].notna().all(), "missing Attack values"

print("Rows:", len(df))
print("Feature columns:", len(df.columns) - 2)
print("Fixed inf cells:", before_inf, "| filled NaN cells:", before_nan)
print("Label balance:")
print(df["Label"].value_counts().to_string())

df.to_csv(OUT, index=False)
print("\nSaved:", OUT)
print("DONE")