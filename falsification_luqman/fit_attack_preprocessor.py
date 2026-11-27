from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile

import joblib
import numpy as np
import pandas as pd

from pre_processing.preprocess import (
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERICAL_COLUMNS,
)
from pre_processing.preprocess_shared import (
    SharedPreprocessor,
    clean_checked,
    read_chunks,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "Raw_DATASET2"
PREPARED_DIR = ROOT / "data_split" / "Raw_DATASET2" / "train"
OUTPUT = (
    ROOT / "falsification_luqman" / "results" /
    "Raw_DATASET2" / "attack_preprocessor.joblib"
)


def file_number(path):
    return int(path.stem.rsplit("_", 1)[-1])


raw_paths = sorted(RAW_DIR.glob("*.csv"), key=file_number)

if len(raw_paths) < 3:
    raise FileNotFoundError(f"Expected the raw ToN_IoT CSV files in: {RAW_DIR}")

# This is the same training-file rule used by run_zeek_pipeline.
train_end = max(1, int(len(raw_paths) * 0.70))
train_paths = raw_paths[:train_end]

print(f"Found {len(raw_paths)} raw ToN_IoT files.")
print(f"Fitting only on the original {len(train_paths)} training files:")
for path in train_paths:
    print(f"  {path.name}")

processor = SharedPreprocessor()

# The temporary folder contains only intermediate fitting values.
# It is automatically removed when fitting finishes.
with tempfile.TemporaryDirectory(prefix=".attack-fit-work-", dir=ROOT) as work:
    processor.fit(train_paths, Path(work))

print("\nVerifying the reconstructed transformations against prepared CSV data...")

# File 1 belongs completely to train, so its raw and prepared rows align.
raw_sample = next(read_chunks(train_paths[0])).head(5000).copy()
clean_sample = clean_checked(raw_sample, train_paths[0].name)
expected = processor.transform(clean_sample).reset_index(drop=True)

prepared_sample = pd.read_csv(
    PREPARED_DIR / train_paths[0].name,
    nrows=len(raw_sample),
)[FEATURE_COLUMNS].reset_index(drop=True)

expected_values = expected.to_numpy(dtype=np.float64)
prepared_values = prepared_sample.to_numpy(dtype=np.float64)

if expected_values.shape != prepared_values.shape:
    raise RuntimeError(
        f"Verification shape mismatch: {expected_values.shape} versus "
        f"{prepared_values.shape}"
    )

max_difference = float(
    np.max(np.abs(expected_values - prepared_values))
)

if not np.allclose(
    expected_values,
    prepared_values,
    rtol=1e-9,
    atol=1e-9,
):
    raise RuntimeError(
        "Reconstructed preprocessing does not match the prepared CSV. "
        f"Maximum absolute difference: {max_difference}"
    )

artifact = {
    "dataset": "Raw_DATASET2",
    "raw_training_files": [path.name for path in train_paths],
    "numerical_columns": NUMERICAL_COLUMNS,
    "categorical_columns": CATEGORICAL_COLUMNS,
    "boolean_columns": BOOLEAN_COLUMNS,
    "feature_columns": FEATURE_COLUMNS,
    "medians": processor.medians,
    "mappings": processor.mappings,
    "scaler": processor.scaler,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(artifact, OUTPUT)

print(f"\nVerification passed on {len(raw_sample):,} raw/prepared row pairs.")
print(f"Maximum absolute numerical difference: {max_difference:.3g}")
print(f"Saved preprocessing metadata to: {OUTPUT}")