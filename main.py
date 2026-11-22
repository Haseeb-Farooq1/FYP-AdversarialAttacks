
import argparse

from Model.pipeline import run as train_model

from pre_processing.preprocess_shared import PROJECT_ROOT, run_pipeline as run_preprocessing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train each IoT dataset independently")
    parser.add_argument(
        "--dataset", choices=["Raw_DATASET", "Raw_DATASET2"],
        help="process and train only this dataset; omit to run both separately",
    )
    args = parser.parse_args()
    datasets = [args.dataset] if args.dataset else ["Raw_DATASET", "Raw_DATASET2"]
    for dataset in datasets:
        run_preprocessing(
            PROJECT_ROOT / dataset,
            PROJECT_ROOT / "data_split" / dataset,
        )
        train_model([dataset])
