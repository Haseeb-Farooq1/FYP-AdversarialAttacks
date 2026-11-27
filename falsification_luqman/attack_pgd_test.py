from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from pre_processing.preprocess import NUMERICAL_COLUMNS


RESULTS = ROOT / "falsification_luqman" / "results" / "Raw_DATASET2"
ATTACK_DIR = RESULTS / "attack"
CANDIDATES = ATTACK_DIR / "test_candidates.npz"
CHECKPOINT = RESULTS.parent / "mlp_final.pt"
PREPROCESSOR = RESULTS / "attack_preprocessor.joblib"
CONSTRAINTS = RESULTS / "constraints.json"

STEPS = 20
ALPHA = 0.05
EPSILON = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50
RUN_TAG = f"eps_{EPSILON:.2f}".replace(".", "p")
THRESHOLD = 0.5

DIRECT = [
    "src_port",
    "dst_port",
    "duration",
    "src_bytes",
    "src_pkts",
    "src_ip_bytes",
]

# src_ip_bytes is derived from the other byte/packet features.
INDEPENDENT = [
    "src_port",
    "dst_port",
    "duration",
    "src_bytes",
    "src_pkts",
]

PORTS = {"src_port", "dst_port"}
INTEGER_FEATURES = set(DIRECT) - {"duration"}


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


artifact = joblib.load(PREPROCESSOR)
scaler = artifact["scaler"]
mappings = artifact["mappings"]

with CONSTRAINTS.open(encoding="utf-8") as file:
    constraints = json.load(file)

candidates = np.load(CANDIDATES, allow_pickle=False)
X0 = candidates["X"].astype(np.float32)
original_score = candidates["original_score"].astype(np.float32)
attack_types = candidates["type"].astype(str)
features = [str(value) for value in candidates["features"]]
feature_index = {column: features.index(column) for column in features}

n_rows = len(X0)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

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
        logits = model((values - mu) / sd)
        outputs.append(torch.sigmoid(logits).numpy())

    return np.concatenate(outputs)


def inverse_numeric(column, prepared):
    """Convert a shared prepared value to a raw network value."""
    prepared = np.asarray(prepared, dtype=np.float64)
    index = NUMERICAL_COLUMNS.index(column)
    value = prepared * scaler.scale_[index] + scaler.mean_[index]

    if column not in PORTS:
        value = np.expm1(value)

    return value


def forward_numeric(column, raw):
    """Convert a raw network value to a shared prepared value."""
    raw = np.asarray(raw, dtype=np.float64)

    if column not in PORTS:
        raw = np.log1p(raw)

    index = NUMERICAL_COLUMNS.index(column)
    return (raw - scaler.mean_[index]) / scaler.scale_[index]


# Confirm that candidate loading and model scoring are consistent.
loaded_scores = predict_scores(X0)
score_difference = float(np.max(np.abs(loaded_scores - original_score)))

if score_difference > 1e-6:
    raise RuntimeError(
        f"Model reproduction failed. Score difference: {score_difference}"
    )

print(
    "Model reproduction verified. "
    f"Maximum score difference: {score_difference:.3g}"
)

# Recover raw original values.
raw_original = {
    column: inverse_numeric(column, X0[:, feature_index[column]])
    for column in DIRECT
}

for column in INTEGER_FEATURES:
    raw_original[column] = np.rint(raw_original[column])

raw_original["duration"] = np.maximum(raw_original["duration"], 0)

packets_original = raw_original["src_pkts"]
header_original = np.full(n_rows, np.nan, dtype=np.float64)

valid_packet_rows = packets_original > 0
header_original[valid_packet_rows] = (
    raw_original["src_ip_bytes"][valid_packet_rows]
    - raw_original["src_bytes"][valid_packet_rows]
) / packets_original[valid_packet_rows]

header_min = constraints["src_ip_bytes_rule"]["header_min"]
header_max = constraints["src_ip_bytes_rule"]["header_max"]

# Features outside the observed training range remain locked for that row.
active = {}
lower_prepared = {}
upper_prepared = {}
lower_raw = {}
upper_raw = {}

for column in INDEPENDENT:
    original = X0[:, feature_index[column]].astype(np.float64)
    bounds = constraints["prepared_bounds"][column]

    active[column] = (
        (original >= bounds["min"])
        & (original <= bounds["max"])
    )

    lower = np.maximum(original - EPSILON, bounds["min"])
    upper = np.minimum(original + EPSILON, bounds["max"])

    lower = np.where(active[column], lower, original)
    upper = np.where(active[column], upper, original)

    lower_prepared[column] = lower
    upper_prepared[column] = upper
    lower_raw[column] = inverse_numeric(column, lower)
    upper_raw[column] = inverse_numeric(column, upper)

ip_bounds = constraints["prepared_bounds"]["src_ip_bytes"]
original_ip_prepared = X0[:, feature_index["src_ip_bytes"]].astype(np.float64)

dependency_ok = (
    valid_packet_rows
    & np.isfinite(header_original)
    & (header_original >= header_min)
    & (header_original <= header_max)
    & active["src_bytes"]
    & active["src_pkts"]
    & (original_ip_prepared >= ip_bounds["min"])
    & (original_ip_prepared <= ip_bounds["max"])
)

print(
    "Rows eligible for linked byte/packet changes: "
    f"{dependency_ok.sum():,}/{n_rows:,}"
)


def category_code(column, raw_value):
    if raw_value == "":
        return 0
    return int(mappings[column].get(raw_value, 1))


# Build the protocol/service destination-port lookup.
port_rules = {}

for rule in constraints["dst_port_by_proto_service"]:
    key = (
        category_code("proto", rule["proto"]),
        category_code("service", rule["service"]),
    )
    port_rules[key] = np.asarray(rule["observed_ports"], dtype=np.int64)

proto_codes = X0[:, feature_index["proto"]].astype(np.int64)
service_codes = X0[:, feature_index["service"]].astype(np.int64)


def project(stepped):
    """Project a PGD step back into valid raw and prepared feature space."""
    output = X0.copy()
    raw_new = {}

    for column in INDEPENDENT:
        index = feature_index[column]
        prepared = np.clip(
            stepped[:, index],
            lower_prepared[column],
            upper_prepared[column],
        )
        raw = inverse_numeric(column, prepared)

        if column == "duration":
            raw = np.clip(
                raw,
                lower_raw[column],
                upper_raw[column],
            )
        else:
            low = np.ceil(lower_raw[column] - 1e-8)
            high = np.floor(upper_raw[column] + 1e-8)

            raw = np.rint(raw)
            raw = np.clip(raw, low, high)

            invalid_interval = low > high
            raw[invalid_interval] = raw_original[column][invalid_interval]

        raw[~active[column]] = raw_original[column][~active[column]]
        raw_new[column] = raw

    # Snap destination ports to observed values for the same proto/service.
    for row in range(n_rows):
        if not active["dst_port"][row]:
            continue

        key = (
            int(proto_codes[row]),
            int(service_codes[row]),
        )
        allowed = port_rules.get(key)

        if allowed is None:
            raw_new["dst_port"][row] = raw_original["dst_port"][row]
            continue

        low = int(np.ceil(lower_raw["dst_port"][row] - 1e-8))
        high = int(np.floor(upper_raw["dst_port"][row] + 1e-8))
        choices = allowed[(allowed >= low) & (allowed <= high)]

        if choices.size:
            current = raw_new["dst_port"][row]
            nearest = np.argmin(np.abs(choices - current))
            raw_new["dst_port"][row] = choices[nearest]
        else:
            raw_new["dst_port"][row] = raw_original["dst_port"][row]

    for column in ["src_port", "dst_port", "duration"]:
        index = feature_index[column]
        output[:, index] = forward_numeric(column, raw_new[column])
        output[~active[column], index] = X0[~active[column], index]

    # Preserve raw header overhead while bytes and packets change.
    rows = np.flatnonzero(dependency_ok)
    source_bytes = raw_new["src_bytes"][rows]
    source_packets = np.maximum(raw_new["src_pkts"][rows], 1)

    header_total = np.rint(source_packets * header_original[rows])
    header_total = np.maximum(
        header_total,
        np.ceil(header_min * source_packets),
    )
    header_total = np.minimum(
        header_total,
        np.floor(header_max * source_packets),
    )

    source_ip_bytes = source_bytes + header_total

    raw_ip_bounds = constraints["features"]["src_ip_bytes"]
    valid = (
        (source_ip_bytes >= raw_ip_bounds["min"])
        & (source_ip_bytes <= raw_ip_bounds["max"])
    )

    valid_rows = rows[valid]

    for column, values in [
        ("src_bytes", source_bytes),
        ("src_pkts", source_packets),
        ("src_ip_bytes", source_ip_bytes),
    ]:
        output[valid_rows, feature_index[column]] = forward_numeric(
            column,
            values[valid],
        )

    # If the linked result exceeds observed limits, keep the original triplet.
    fallback_rows = rows[~valid]

    for column in ["src_bytes", "src_pkts", "src_ip_bytes"]:
        output[fallback_rows, feature_index[column]] = X0[
            fallback_rows,
            feature_index[column],
        ]

    return output.astype(np.float32)


# src_ip_bytes is derived and is not independently attacked.
gradient_mask = np.zeros_like(X0, dtype=np.float32)

for column in ["src_port", "dst_port", "duration"]:
    gradient_mask[:, feature_index[column]] = active[column].astype(
        np.float32
    )

for column in ["src_bytes", "src_pkts"]:
    gradient_mask[:, feature_index[column]] = dependency_ok.astype(
        np.float32
    )

print(
    f"Starting PGD: steps={STEPS}, alpha={ALPHA}, "
    f"epsilon={EPSILON} in shared prepared units"
)

adversarial = X0.copy()

for step in range(1, STEPS + 1):
    attack_tensor = torch.from_numpy(adversarial).clone().requires_grad_(True)
    logits = model((attack_tensor - mu) / sd)

    # Target normal class 0: descending this loss lowers the attack score.
    loss = F.binary_cross_entropy_with_logits(
        logits,
        torch.zeros_like(logits),
    )

    gradient = torch.autograd.grad(loss, attack_tensor)[0].numpy()
    stepped = (
        adversarial
        - ALPHA * np.sign(gradient) * gradient_mask
    )
    adversarial = project(stepped)

    if step in {1, 5, 10, 20}:
        current_score = predict_scores(adversarial)
        evasions = int((current_score < THRESHOLD).sum())

        print(
            f"Step {step:02d}: "
            f"mean score={current_score.mean():.6f}, "
            f"evasions={evasions}/{n_rows}",
            flush=True,
        )

# Save scores before final raw-value canonicalization.
pre_canonical_score = predict_scores(adversarial)

raw_adversarial_exact = {
    column: inverse_numeric(
        column,
        adversarial[:, feature_index[column]],
    )
    for column in DIRECT
}

# Raw integer values are the canonical generated network values.
raw_adversarial = {}

for column in DIRECT:
    if column in INTEGER_FEATURES:
        raw_adversarial[column] = np.rint(
            raw_adversarial_exact[column]
        )
    else:
        raw_adversarial[column] = raw_adversarial_exact[column]

# Raw duration must be physically nonnegative.
raw_adversarial["duration"] = np.maximum(
    raw_adversarial["duration"],
    0.0,
)

# Re-enforce the byte/packet dependency after canonical integer rounding.
dependency_fallback = np.zeros(n_rows, dtype=bool)
dependency_rows = np.flatnonzero(dependency_ok)

canonical_bytes = raw_adversarial["src_bytes"][dependency_rows]
canonical_packets = raw_adversarial["src_pkts"][dependency_rows]

canonical_header_total = np.rint(
    canonical_packets * header_original[dependency_rows]
)

canonical_header_total = np.maximum(
    canonical_header_total,
    np.ceil(header_min * canonical_packets),
)
canonical_header_total = np.minimum(
    canonical_header_total,
    np.floor(header_max * canonical_packets),
)

canonical_ip_bytes = canonical_bytes + canonical_header_total

raw_ip_bounds = constraints["features"]["src_ip_bytes"]

canonical_dependency_valid = (
    (canonical_packets > 0)
    & (canonical_ip_bytes >= raw_ip_bounds["min"])
    & (canonical_ip_bytes <= raw_ip_bounds["max"])
)

valid_dependency_rows = dependency_rows[canonical_dependency_valid]
fallback_dependency_rows = dependency_rows[~canonical_dependency_valid]

raw_adversarial["src_ip_bytes"][valid_dependency_rows] = (
    canonical_ip_bytes[canonical_dependency_valid]
)

# If no valid linked result exists, preserve the original raw triplet.
for column in ["src_bytes", "src_pkts", "src_ip_bytes"]:
    raw_adversarial[column][fallback_dependency_rows] = (
        raw_original[column][fallback_dependency_rows]
    )

dependency_fallback[fallback_dependency_rows] = True

# Regenerate model inputs from the canonical raw values.
canonical = X0.copy()

for column in DIRECT:
    canonical[:, feature_index[column]] = forward_numeric(
        column,
        raw_adversarial[column],
    )

# Preserve exact prepared values for locked and fallback byte/packet rows.
canonical_byte_lock = (~dependency_ok) | dependency_fallback

for column in ["src_bytes", "src_pkts", "src_ip_bytes"]:
    canonical[canonical_byte_lock, feature_index[column]] = X0[
        canonical_byte_lock,
        feature_index[column],
    ]

canonical = canonical.astype(np.float32)

prepared_shift = float(
    np.max(np.abs(canonical - adversarial))
)

adversarial = canonical
adversarial_score = predict_scores(adversarial)
evasion_success = adversarial_score < THRESHOLD

score_shift = float(
    np.max(np.abs(adversarial_score - pre_canonical_score))
)

print(f"\nCanonical prepared-value shift: {prepared_shift:.6g}")
print(f"Canonical model-score shift: {score_shift:.6g}")

changed = {}

for column in DIRECT:
    if column in INTEGER_FEATURES:
        changed[column] = (
            raw_adversarial[column] != raw_original[column]
        )
    else:
        changed[column] = ~np.isclose(
            raw_adversarial[column],
            raw_original[column],
            atol=1e-6,
            rtol=0,
        )

# Validate the exact canonical raw values that will be saved.
range_violations = 0

for column in DIRECT:
    bounds = constraints["features"][column]
    values = raw_adversarial[column]

    range_violations += int(np.count_nonzero(
        changed[column]
        & (
            (values < bounds["min"] - 1e-5)
            | (values > bounds["max"] + 1e-5)
        )
    ))

integer_violations = 0

for column in INTEGER_FEATURES:
    values = raw_adversarial[column]

    integer_violations += int(np.count_nonzero(
        np.abs(values - np.rint(values)) > 1e-4
    ))

locked_columns = [
    column
    for column in features
    if column not in DIRECT
]

locked_violations = 0

for column in locked_columns:
    index = feature_index[column]

    locked_violations += int(np.count_nonzero(
        adversarial[:, index] != X0[:, index]
    ))

byte_lock_violations = 0

for column in ["src_bytes", "src_pkts", "src_ip_bytes"]:
    index = feature_index[column]

    byte_lock_violations += int(np.count_nonzero(
        (~dependency_ok)
        & (adversarial[:, index] != X0[:, index])
    ))

dependency_violations = 0
rows = np.flatnonzero(dependency_ok)
packets = raw_adversarial["src_pkts"][rows]
valid_packets = packets > 0

if not valid_packets.all():
    dependency_violations += int((~valid_packets).sum())

if valid_packets.any():
    checked_rows = rows[valid_packets]
    checked_packets = packets[valid_packets]

    headers = (
        raw_adversarial["src_ip_bytes"][checked_rows]
        - raw_adversarial["src_bytes"][checked_rows]
    ) / checked_packets

    dependency_violations += int(np.count_nonzero(
        (headers < header_min - 1e-4)
        | (headers > header_max + 1e-4)
    ))

dst_port_violations = 0

for row in range(n_rows):
    if not changed["dst_port"][row]:
        continue

    key = (
        int(proto_codes[row]),
        int(service_codes[row]),
    )
    allowed = port_rules.get(key)
    value = int(raw_adversarial["dst_port"][row])

    if allowed is None or value not in allowed:
        dst_port_violations += 1

negative_violations = sum(
    int(np.count_nonzero(raw_adversarial[column] < 0))
    for column in DIRECT
)        

constraint_results = {
    "negative_value_violations": negative_violations,
    "range_violations": range_violations,
    "integer_violations": integer_violations,
    "locked_feature_violations": locked_violations,
    "byte_lock_violations": byte_lock_violations,
    "dependency_violations": dependency_violations,
    "destination_port_rule_violations": dst_port_violations,
}

constraint_results["total"] = int(
    sum(constraint_results.values())
)

print("\nConstraint validation:")
print(json.dumps(constraint_results, indent=2))

if constraint_results["total"] != 0:
    raise RuntimeError(
        "Constraint validation failed. "
        "Adversarial outputs were not saved."
    )

changed_count = sum(
    changed[column].astype(np.int32)
    for column in DIRECT
)

per_type = {}

for attack_type in sorted(set(attack_types)):
    selected = attack_types == attack_type
    count = int(selected.sum())
    successes = int(evasion_success[selected].sum())

    per_type[attack_type] = {
        "selected": count,
        "evasion_successes": successes,
        "evasion_success_rate": successes / count,
        "mean_original_score": float(
            original_score[selected].mean()
        ),
        "mean_adversarial_score": float(
            adversarial_score[selected].mean()
        ),
    }

results = {
    "data": "ToN_IoT held-out test candidates only",
    "model": "Chronological train+validation MLP",
    "attack": "Constrained numeric PGD held-out evaluation",
    "config": {
        "steps": STEPS,
        "alpha": ALPHA,
        "epsilon_shared_prepared_units": EPSILON,
        "threshold": THRESHOLD,
        "direct_features": DIRECT,
        "dns_and_http_fields": "locked in this pilot",
    },
    "selected_correctly_detected_attacks": n_rows,
    "evasion_successes": int(evasion_success.sum()),
    "evasion_success_rate": float(evasion_success.mean()),
    "attack_recall_before_on_selected": 1.0,
    "attack_recall_after_on_selected": float(
        (adversarial_score >= THRESHOLD).mean()
    ),
    "mean_original_score": float(original_score.mean()),
    "mean_adversarial_score": float(adversarial_score.mean()),
    "mean_direct_features_changed": float(changed_count.mean()),
    "dependency_eligible_rows": int(dependency_ok.sum()),
    "canonical_prepared_value_max_shift": prepared_shift,
    "canonical_model_score_max_shift": score_shift,
    "constraint_violations": constraint_results,
    "per_type": per_type,
        "interpretation_limit": (
        "Denominator is correctly detected held-out test attacks only. "
        "All three attack configurations were frozen using validation data "
        "before this evaluation."
    ),
}

# Full generated examples in the model's prepared representation.
prepared_output = pd.DataFrame(
    adversarial,
    columns=features,
)
prepared_output["label"] = 1
prepared_output["type"] = attack_types
prepared_output["__time"] = candidates["time"]
prepared_output["source_file"] = candidates["source_file"]
prepared_output["row_index"] = candidates["row_index"]
prepared_output["original_score"] = original_score
prepared_output["adversarial_score"] = adversarial_score
prepared_output["evasion_success"] = evasion_success.astype(np.int8)

# Readable raw before-and-after values.
raw_output = pd.DataFrame({
    "type": attack_types,
    "source_file": candidates["source_file"],
    "row_index": candidates["row_index"],
    "original_score": original_score,
    "adversarial_score": adversarial_score,
    "evasion_success": evasion_success.astype(np.int8),
    "direct_features_changed": changed_count,
})

for column in DIRECT:
    raw_output[f"{column}_original"] = raw_original[column]
    raw_output[f"{column}_adversarial"] = raw_adversarial[column]

with (ATTACK_DIR / f"pgd_test_{RUN_TAG}_results.json").open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        results,
        file,
        indent=2,
        allow_nan=False,
    )

prepared_output.to_csv(
    ATTACK_DIR / f"pgd_test_{RUN_TAG}_adversarial_prepared.csv",
    index=False,
)

raw_output.to_csv(
    ATTACK_DIR / f"pgd_test_{RUN_TAG}_raw_changes.csv",
    index=False,
)

print("\nHeld-out test results:")
print(json.dumps(results, indent=2))
print(f"\nSaved attack outputs to: {ATTACK_DIR}")