import argparse
import sys
from pathlib import Path

from data_preprocessing.preprocess import compact_report, preprocess_datasets
from data_preprocessing.pcap_to_csv import export_pcap


DATASET_PATH = Path(__file__).parent / "Network_dataset_1.csv"
DATASET_PATTERN = "Network_dataset_*.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CSV datasets or extract PCAP flows."
    )
    parser.add_argument("--pcap", type=Path, help="Extract flows from an Ethernet PCAP/PCAPNG")
    parser.add_argument("--output", type=Path, help="Output CSV path for --pcap mode")
    parser.add_argument("--idle-timeout", type=float, default=60.0)
    parser.add_argument(
        "--label", type=int, choices=(0, 1),
        help="Optional ground truth: 0 normal, 1 attack",
    )
    parser.add_argument(
        "--type", dest="attack_type", help="Optional attack type, e.g. scanning or ddos"
    )
    args, _ = parser.parse_known_args()
    if args.pcap:
        if not args.output:
            parser.error("--output is required with --pcap")
        report = export_pcap(
            args.pcap, args.output, args.idle_timeout, args.label, args.attack_type
        )
        print(
            f"Exported {report['flows_written']:,} flows from {report['packets_seen']:,} packets "
            f"({report['packets_skipped']:,} skipped, "
            f"{report['malformed_ip_packets']:,} malformed-IP packets retained) "
            f"to {report['output_file']}",
            flush=True,
        )
        if report["label"] is None:
            print(
                "Rows are unlabeled; supply --label/--type only when ground truth is known.",
                flush=True,
            )
        return

    report_path = DATASET_PATH.parent / "preprocessing_report.json"
    if "--compact-report" in sys.argv:
        compact_report(report_path)
        print(f"Compact report saved to {report_path.name}.")
        return

    dataset_paths = sorted(DATASET_PATH.parent.glob(DATASET_PATTERN))
    results = preprocess_datasets(dataset_paths)

    for output_path, row_count, column_count in results:
        print(f"Saved {row_count} rows and {column_count} columns to {output_path.name}.")

    print(f"Report saved to {report_path.name}.")


if __name__ == "__main__":
    main()
