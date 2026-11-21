"""Train a simple LightGBM model on prepared train/validation/test CSVs."""

from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_ROOT / "data_split"
TRAIN_DIR = SPLIT_DIR / "train"
VALIDATION_DIR = SPLIT_DIR / "validation"
TEST_DIR = SPLIT_DIR / "test"
TARGET_COLUMN = "label"
SEED = 42
FOLDS = 5


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load numeric prepared features, excluding the target and descriptive label."""
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"{path.name}: missing target column {TARGET_COLUMN!r}")

    y = frame[TARGET_COLUMN].astype("int8")
    X = frame.drop(columns=[TARGET_COLUMN, "type"], errors="ignore")
    if X.empty:
        raise ValueError(f"{path.name}: no feature columns found")
    return X, y


def build_model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.1,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def train_file(train_path: Path, validation_path: Path, test_path: Path) -> LGBMClassifier:
    X, y = load_dataset(train_path)
    X_validation, y_validation = load_dataset(validation_path)
    test_columns = pd.read_csv(test_path, nrows=0).columns.tolist()
    test_features = [column for column in test_columns if column not in {TARGET_COLUMN, "type"}]
    if list(X.columns) != test_features or list(X.columns) != list(X_validation.columns):
        raise ValueError(f"{train_path.name}: train/validation/test feature columns do not match")

    # Fit one model per training fold. Fold metrics are intentionally omitted.
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (fit_rows, _validation_rows) in enumerate(splitter.split(X, y), start=1):
        model = build_model()
        model.fit(X.iloc[fit_rows], y.iloc[fit_rows])
        print(f"{train_path.name}: completed training fold {fold}/{FOLDS}")

    # Fit the returned model on train plus validation. The test split remains
    # untouched, and no evaluation or output artifacts are generated.
    X_final = pd.concat([X, X_validation], ignore_index=True)
    y_final = pd.concat([y, y_validation], ignore_index=True)
    final_model = build_model()
    final_model.fit(X_final, y_final)
    print(f"{train_path.name}: model training complete")

    X_test, y_test = load_dataset(test_path)
    predictions = final_model.predict(X_test)
    f1 = f1_score(y_test, predictions, average="binary", zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    print(f"{train_path.name}: test F1 score={f1:.6f}")
    print(f"{train_path.name}: confusion matrix (rows=true, columns=predicted; labels=[0, 1])")
    print(matrix)
    print(f"{train_path.name}: classification report")
    print(classification_report(y_test, predictions, labels=[0, 1], zero_division=0))
    return final_model


def run() -> list[LGBMClassifier]:
    train_files = sorted(TRAIN_DIR.glob("*.csv"), key=lambda path: path.name.casefold())
    validation_files = {path.name.casefold(): path for path in VALIDATION_DIR.glob("*.csv")}
    test_files = {path.name.casefold(): path for path in TEST_DIR.glob("*.csv")}
    if not train_files:
        raise FileNotFoundError(f"No prepared training CSV files found in {TRAIN_DIR}")
    train_names = {path.name.casefold() for path in train_files}
    if train_names != set(validation_files) or train_names != set(test_files):
        raise ValueError("Train, validation, and test directories must contain matching CSV filenames")

    models = []
    for train_path in train_files:
        print(f"\nTraining {train_path.name}")
        name = train_path.name.casefold()
        models.append(train_file(train_path, validation_files[name], test_files[name]))
    return models
