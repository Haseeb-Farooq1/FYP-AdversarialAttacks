from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
from collections import defaultdict

import joblib
import numpy as np

from pre_processing.preprocess import NUMERICAL_COLUMNS
from pre_processing.preprocess_shared import clean_checked, read_chunks


RESULTS = ROOT / "falsification_luqman" / "results" / "Raw_DATASET2"
ARTIFACT = RESULTS / "attack_preprocessor.joblib"
OUTPUT = RESULTS / "constraints.json"
RAW_DIR = ROOT / "Raw_DATASET2"

DIRECT = [
    "src_port",
    "dst_port",
    "duration",
    "src_bytes",
    "src_pkts",
    "src_ip_bytes",
]

CONDITIONAL = [
    "dns_qtype",
    "dns_qclass",
    "dns_RD",
    "http_request_body_len",
]

PORTS = {"src_port", "dst_port"}

artifact = joblib.load(ARTIFACT)
scaler = artifact["scaler"]
mappings = artifact["mappings"]
train_paths = [RAW_DIR / name for name in artifact["raw_training_files"]]

for path in train_paths:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw training file: {path}")

values = {column: [] for column in DIRECT + ["http_request_body_len"]}
header_parts = []
port_pairs = defaultdict(set)
training_rows = 0


def to_prepared(column, raw_values):
    """Convert observed raw numeric values to shared prepared values."""
    raw_values = np.asarray(raw_values, dtype=np.float64)

    if column not in PORTS:
        raw_values = np.log1p(raw_values)

    index = NUMERICAL_COLUMNS.index(column)
    return (raw_values - scaler.mean_[index]) / scaler.scale_[index]


def describe(parts):
    values_array = np.concatenate(parts).astype(np.float64)
    return {
        "min": float(values_array.min()),
        "max": float(values_array.max()),
        "q01": float(np.quantile(values_array, 0.01)),
        "q99": float(np.quantile(values_array, 0.99)),
        "observed_values": int(values_array.size),
    }


print("Learning raw constraints from training files only...")

for path in train_paths:
    print(f"Reading {path.name}", flush=True)

    for raw in read_chunks(path):
        clean = clean_checked(raw, path.name)
        training_rows += len(clean)

        for column in DIRECT:
            observed = clean[column].dropna().to_numpy(dtype=np.float64)
            if observed.size:
                values[column].append(observed)

        http_values = clean.loc[
            clean["service"].eq("http"),
            "http_request_body_len",
        ].dropna().to_numpy(dtype=np.float64)

        if http_values.size:
            values["http_request_body_len"].append(http_values)

        dependency_rows = clean[
            ["src_bytes", "src_pkts", "src_ip_bytes"]
        ].notna().all(axis=1) & clean["src_pkts"].gt(0)

        if dependency_rows.any():
            selected = clean.loc[dependency_rows]
            header = (
                (selected["src_ip_bytes"] - selected["src_bytes"])
                / selected["src_pkts"]
            ).to_numpy(dtype=np.float64)

            header = header[np.isfinite(header)]
            if header.size:
                header_parts.append(header)

        # Save observed destination ports for protocol/service consistency.
        selected = clean[["proto", "service", "dst_port"]].dropna()
        for (proto, service), group in selected.groupby(["proto", "service"]):
            ports = group["dst_port"].to_numpy(dtype=np.int64)
            port_pairs[(str(proto), str(service))].update(ports.tolist())

if not header_parts:
    raise ValueError("No valid rows available for the src_ip_bytes rule.")

constraints = {
    "dataset": "Raw_DATASET2",
    "source": "Raw ToN_IoT training files only",
    "training_rows": training_rows,
    "preprocessor_artifact": ARTIFACT.name,
    "modifiable": DIRECT,
    "conditional": CONDITIONAL,
    "features": {},
    "prepared_bounds": {},
}

print("\nRaw and prepared feature bounds:")

for column in DIRECT + ["http_request_body_len"]:
    if not values[column]:
        raise ValueError(f"No observed training values for: {column}")

    stats = describe(values[column])
    stats["integer"] = column != "duration"
    constraints["features"][column] = stats

    raw_bounds = [stats[key] for key in ["min", "max", "q01", "q99"]]
    prepared = to_prepared(column, raw_bounds)

    constraints["prepared_bounds"][column] = {
        key: float(prepared[index])
        for index, key in enumerate(["min", "max", "q01", "q99"])
    }

    print(
        f"{column}: raw min={stats['min']:.6g}, "
        f"q01={stats['q01']:.6g}, q99={stats['q99']:.6g}, "
        f"max={stats['max']:.6g}"
    )

header_values = np.concatenate(header_parts)
constraints["src_ip_bytes_rule"] = {
    "definition": "(src_ip_bytes - src_bytes) / src_pkts in raw units",
    "header_min": float(np.quantile(header_values, 0.01)),
    "header_max": float(np.quantile(header_values, 0.99)),
    "observed_rows": int(header_values.size),
}

constraints["valid_codes"] = {}

for column in ["dns_qtype", "dns_qclass"]:
    ordered = sorted(mappings[column].items(), key=lambda item: item[1])
    constraints["valid_codes"][column] = {
        "raw_to_prepared": {key: int(value) for key, value in ordered},
        "raw_values": [key for key, _ in ordered],
        "prepared_codes": [int(value) for _, value in ordered],
    }

constraints["valid_codes"]["dns_RD"] = {
    "raw_to_prepared": {"F": 0, "T": 1},
    "raw_values": ["F", "T"],
    "prepared_codes": [0, 1],
    "note": "Missing code 2 is not introduced into DNS flows by the attack.",
}

service_mapping = mappings["service"]

for required_service in ["dns", "http"]:
    if required_service not in service_mapping:
        raise ValueError(f"Training service mapping lacks: {required_service}")

constraints["applicability"] = {
    "dns_service_labels": ["dns"],
    "dns_service_codes": [int(service_mapping["dns"])],
    "http_service_labels": ["http"],
    "http_service_codes": [int(service_mapping["http"])],
}

constraints["dst_port_by_proto_service"] = [
    {
        "proto": proto,
        "service": service,
        "observed_ports": sorted(int(port) for port in ports),
    }
    for (proto, service), ports in sorted(port_pairs.items())
]

with OUTPUT.open("w", encoding="utf-8") as file:
    json.dump(constraints, file, indent=2, allow_nan=False)

print("\nRaw IP-header bytes per packet:")
print(constraints["src_ip_bytes_rule"])

print("\nDNS query types:")
print(constraints["valid_codes"]["dns_qtype"]["raw_values"])

print("\nDNS classes:")
print(constraints["valid_codes"]["dns_qclass"]["raw_values"])

print("\nService applicability:")
print(constraints["applicability"])

print(
    f"\nSaved {len(constraints['dst_port_by_proto_service'])} "
    "protocol/service destination-port rules."
)
print(f"Saved corrected constraints to: {OUTPUT}")