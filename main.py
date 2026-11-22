
from Model.pipeline import run as train_model

from pre_processing.preprocess_shared import PROJECT_ROOT, run_pipeline as run_preprocessing


if __name__ == "__main__":
    dataset = "Raw_DATASET"
    run_preprocessing(
        PROJECT_ROOT / dataset,
        PROJECT_ROOT / "data_split" / dataset,
    )
    train_model([dataset])
