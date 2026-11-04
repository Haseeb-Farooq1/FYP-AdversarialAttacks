"""Shared schema and cleaning helpers for the Zainab preprocessing pipeline."""

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_COLUMNS = """
ts src_ip src_port dst_ip dst_port proto service duration src_bytes dst_bytes
conn_state missed_bytes src_pkts src_ip_bytes dst_pkts dst_ip_bytes dns_query
dns_qclass dns_qtype dns_rcode dns_AA dns_RD dns_RA dns_rejected ssl_version
ssl_cipher ssl_resumed ssl_established ssl_subject ssl_issuer http_trans_depth
http_method http_uri http_referrer http_version http_request_body_len
http_response_body_len http_status_code http_user_agent http_orig_mime_types
http_resp_mime_types weird_name weird_addl weird_notice label type
""".split()
NUMERICAL_COLUMNS = """
src_port dst_port duration src_bytes dst_bytes missed_bytes src_pkts
src_ip_bytes dst_pkts dst_ip_bytes http_request_body_len http_response_body_len
""".split()
CATEGORICAL_COLUMNS = """
proto service conn_state dns_qclass dns_qtype dns_rcode http_status_code
""".split()
BOOLEAN_COLUMNS = ["dns_AA", "dns_RD", "dns_RA", "dns_rejected"]
FEATURE_COLUMNS = [
    column for column in EXPECTED_COLUMNS
    if column in NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS + BOOLEAN_COLUMNS
]
ATTACK_TYPES = sorted([
    "normal", "backdoor", "ddos", "dos", "injection", "mitm", "password",
    "ransomware", "scanning", "xss",
])
BOOLEAN_MAPPING = {"F": 0, "T": 1, "": 2}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_schema(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), [])
    columns = [column.strip() for column in header]
    if len(columns) != len(set(columns)):
        raise ValueError(f"{Path(path).name}: duplicate column names are not supported")
    if "uid" in columns:
        columns.remove("uid")
    if columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"{Path(path).name}: expected the 46-column schema/order, with optional uid"
        )


def validate_targets(frame):
    label = frame["label"].str.strip()
    attack_type = frame["type"].str.strip()
    if not label.isin(["0", "1"]).all() or not attack_type.isin(ATTACK_TYPES).all():
        raise ValueError("Missing or unrecognized label/type value")
    if not ((label == "0") == (attack_type == "normal")).all():
        raise ValueError("label and type disagree")
    return label.astype("int8"), attack_type


def clean_features(frame):
    """Normalize and validate selected raw features without dropping rows."""
    clean = pd.DataFrame(index=frame.index)
    for column in FEATURE_COLUMNS:
        values = frame[column].str.strip().replace("-", "")
        if column in NUMERICAL_COLUMNS:
            if column == "src_bytes":
                values = values.replace("0.0.0.0", "")
            clean[column] = pd.to_numeric(
                values.replace("", pd.NA), errors="raise"
            ).to_numpy(dtype=np.float64, na_value=np.nan)
            numeric = clean[column].to_numpy()
            if np.isinf(numeric).any() or (numeric < 0).any():
                raise ValueError(f"{column}: infinite or negative numeric value")
            observed = numeric[np.isfinite(numeric)]
            if column != "duration" and (observed != np.floor(observed)).any():
                raise ValueError(f"{column}: expected integer values")
            if column in ("src_port", "dst_port") and (observed > 65535).any():
                raise ValueError(f"{column}: value exceeds 65535")
        elif column in BOOLEAN_COLUMNS:
            if not values.isin(BOOLEAN_MAPPING).all():
                raise ValueError(f"{column}: expected F, T, or missing")
            clean[column] = values.map(BOOLEAN_MAPPING).astype("int8")
        else:
            clean[column] = values
    return clean
