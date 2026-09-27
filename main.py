from pathlib import Path

import sys

from data_preprocessing.preprocess import compact_report, preprocess_datasets


DATASET_PATH = Path(__file__).parent / "Network_dataset_1.csv"
DATASET_PATTERN = "Network_dataset_*.csv"


def main():
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
