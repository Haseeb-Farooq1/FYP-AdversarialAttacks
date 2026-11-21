from pathlib import Path
import pandas as pd

# Find the CSV automatically, wherever it is
found = list(Path(r"D:\Final Year Project").rglob("NF-UNSW-NB15-v3.csv"))
if len(found) != 1:
    print("PROBLEM: found", len(found), "files:", found)
    raise SystemExit(1)

DATA = found[0]
print("Data file:", DATA)

FEAT = DATA.with_name("NetFlow_v3_Features.csv")
print("\n===== FEATURE DICTIONARY =====")
print(pd.read_csv(FEAT).to_string())

print("\n===== COLUMN NAMES AND TYPES (from first 5 rows) =====")
preview = pd.read_csv(DATA, nrows=5)
print("Number of columns:", len(preview.columns))
for c in preview.columns:
    print(f"{c}  ->  {preview[c].dtype}")