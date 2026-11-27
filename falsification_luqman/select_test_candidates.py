import joblib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data_split" / "Raw_DATASET2"
RESULTS = ROOT / "falsification_luqman" / "results" / "Raw_DATASET2"
CHECKPOINT = RESULTS.parent / "mlp_final.pt"
OUTPUT_DIR = RESULTS / "attack"
OUTPUT = OUTPUT_DIR / "test_candidates.npz"

SEED = 42
CHUNK_SIZE = 100_000
PER_TYPE = 400


class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

preprocessor_artifact = joblib.load(RESULTS / "attack_preprocessor.joblib")
features = checkpoint.get("features", preprocessor_artifact["feature_columns"])
model = MLP(len(features))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

mu = torch.as_tensor(checkpoint["mu"], dtype=torch.float32)
sd = torch.as_tensor(checkpoint["sd"], dtype=torch.float32)


@torch.no_grad()
def predict_scores(x):
    outputs = []

    for start in range(0, len(x), 8192):
        values = torch.from_numpy(x[start:start + 8192])
        normalized = (values - mu) / sd
        logits = model(normalized)
        outputs.append(torch.sigmoid(logits).numpy())

    return np.concatenate(outputs)


rng = np.random.default_rng(SEED)
reservoirs = defaultdict(list)
seen_selected = Counter()

seen_attacks = Counter()
detected_attacks = Counter()

total_rows = 0
total_attacks = 0
total_detected = 0

read_columns = features + ["label", "type", "__time"]
files = sorted((DATASET_DIR / "test").glob("*.csv"))

if not files:
    raise FileNotFoundError("No test CSV files were found.")

print("Scanning test data for correctly detected attacks...")

for path in files:
    print(f"Reading {path.name}", flush=True)
    offset = 0

    with pd.read_csv(
        path,
        usecols=read_columns,
        chunksize=CHUNK_SIZE,
    ) as reader:
        for chunk in reader:
            x = chunk[features].to_numpy(dtype=np.float32)
            y = chunk["label"].to_numpy(dtype=np.int8)
            types = chunk["type"].astype(str).to_numpy()
            times = chunk["__time"].to_numpy(dtype=np.float64)

            if not np.isfinite(x).all():
                raise ValueError(f"Nonfinite model input in {path.name}")

            scores = predict_scores(x)
            predictions = (scores >= 0.5).astype(np.int8)

            attack_indices = np.flatnonzero(y == 1)
            detected_indices = np.flatnonzero((y == 1) & (predictions == 1))

            total_rows += len(chunk)
            total_attacks += len(attack_indices)
            total_detected += len(detected_indices)

            for index in attack_indices:
                seen_attacks[types[index]] += 1

            # Reservoir sampling avoids taking only the first rows of a type.
            for index in detected_indices:
                attack_type = types[index]
                detected_attacks[attack_type] += 1
                seen_selected[attack_type] += 1

                item = {
                    "x": x[index].copy(),
                    "type": attack_type,
                    "time": float(times[index]),
                    "source_file": path.name,
                    "row_index": int(offset + index),
                    "score": float(scores[index]),
                }

                reservoir = reservoirs[attack_type]
                occurrences = seen_selected[attack_type]

                if len(reservoir) < PER_TYPE:
                    reservoir.append(item)
                else:
                    replacement = int(rng.integers(0, occurrences))
                    if replacement < PER_TYPE:
                        reservoir[replacement] = item

            offset += len(chunk)

items = [
    item
    for attack_type in sorted(reservoirs)
    for item in reservoirs[attack_type]
]

if not items:
    raise RuntimeError("No correctly detected validation attacks were found.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT,
    X=np.stack([item["x"] for item in items]).astype(np.float32),
    y=np.ones(len(items), dtype=np.int8),
    type=np.array([item["type"] for item in items]),
    time=np.array([item["time"] for item in items], dtype=np.float64),
    source_file=np.array([item["source_file"] for item in items]),
    row_index=np.array([item["row_index"] for item in items], dtype=np.int64),
    original_score=np.array(
        [item["score"] for item in items],
        dtype=np.float32,
    ),
    features=np.array(features),
)

print("\nTest scan summary:")
print(f"Rows scanned: {total_rows:,}")
print(f"Attacks found: {total_attacks:,}")
print(f"Attacks detected: {total_detected:,}")
print(f"Test attack recall: {total_detected / total_attacks:.6f}")

print("\nSelected candidates by attack type:")
for attack_type in sorted(reservoirs):
    print(
        f"{attack_type}: selected={len(reservoirs[attack_type]):,}, "
        f"detected_available={detected_attacks[attack_type]:,}, "
        f"total_attacks={seen_attacks[attack_type]:,}"
    )

print(f"\nTotal pilot candidates: {len(items):,}")
print(f"Saved candidate set to: {OUTPUT}")