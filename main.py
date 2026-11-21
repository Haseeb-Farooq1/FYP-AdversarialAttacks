
from Model.pipeline import run as train_model

from pre_processing.preprocess_shared import run_pipeline as run_preprocessing


if __name__ == "__main__":
    # Processing datasets 1-23 has already been completed.
    # To preprocess again, uncomment the import above and the call below.
    run_preprocessing()
    train_model()
