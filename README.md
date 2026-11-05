# FYP-AdversarialAttacks
Adversarial Robustness Evaluation and Hardening of AIBased Network Intrusion Detection Systems 

## Compare preprocessed pipelines

Run the logistic-regression baseline against every `PreProcessed_*.csv` file:

```powershell
& ".venv\Scripts\python.exe" evaluate_logistic_regression.py
```

The default run evaluates up to 200,000 rows per file, uses an 80/20 stratified
test split, and writes `logistic_regression_results.json` and
`logistic_regression_results.csv`. The recommended pipeline is selected by
highest F1 score, then recall, then precision. Use `--max-rows 0` to evaluate
all rows.

The binary `label` column is the target. The `type` column is excluded because
it describes the same outcome and would leak target information into the model.

## Convert a PCAP to flow features

Install the packet parser dependency and pass one Ethernet PCAP or PCAPNG file
and an output CSV path to the main entry point:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -B main.py --pcap captures/traffic.pcap --output pre_processed/traffic.csv
```

The exporter creates one row per bidirectional IP flow, with numeric packet/byte
counts, duration, packet-size and interarrival summaries, TCP flag counts,
protocol number, and ports. Endpoint IPs and timestamps are omitted to keep the
predictor columns numeric and reduce identity/time leakage. Flows split after
60 seconds idle by default; override with `--idle-timeout`. It supports Ethernet and Linux-cooked IPv4/IPv6
PCAP/PCAPNG captures (format is detected from the file header, not its
extension). It skips non-IP packets and keeps IPv4-shaped `ETH_P_ALL` packets
with malformed version headers, marking them in `malformed_ip_packets`.

PCAP files do not contain ground-truth labels in general, so output labels are
blank unless you provide verified labels for the capture. For a labeled normal
capture use `--label 0`; for an attack capture use `--label 1 --type scanning`
(replace `scanning` with the known attack type). This assigns that label to all
flows in the file; do not use it for a capture containing mixed labels.
