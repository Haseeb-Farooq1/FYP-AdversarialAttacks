"""Split raw CSVs first, fit Zainab's preprocessing on train, and transform all splits."""

from contextlib import ExitStack
from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pre_processing.preprocess import (
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLUMNS,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    NUMERICAL_COLUMNS,
    PROJECT_ROOT,
    clean_features,
    validate_targets,
    verify_schema,
)


RAW_DATA_DIR = PROJECT_ROOT / "Raw_DATASET"
DATA_SPLIT_DIR = PROJECT_ROOT / "data_split"
SPLIT_NAMES = ("train", "validation", "test")
SEED = 42
CHUNK_SIZE = 50_000
PORT_COLUMNS = {"src_port", "dst_port"}
LOG_COLUMNS = [column for column in NUMERICAL_COLUMNS if column not in PORT_COLUMNS]
NF_TARGET_COLUMNS = {"Label", "Attack"}
NF_EXCLUDED_COLUMNS = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack"}


def read_chunks(path: Path, columns=None):
    columns = columns or FEATURE_COLUMNS + ["label", "type"]
    with pd.read_csv(
        path,
        usecols=columns,
        dtype="string",
        na_filter=False,
        chunksize=CHUNK_SIZE,
        on_bad_lines="error",
    ) as reader:
        yield from reader


def clean_checked(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    try:
        validate_targets(raw)
        return clean_features(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{source}: {exc}") from exc


def split_raw_file(path: Path, raw_split_dir: Path) -> dict[str, Path]:
    verify_schema(path)
    targets = pd.read_csv(
        path, usecols=["label", "type"], dtype="string", na_filter=False
    )
    labels, _ = validate_targets(targets)
    rows = np.arange(len(targets))
    train_rows, remaining_rows = train_test_split(
        rows, test_size=0.30, random_state=SEED, stratify=labels
    )
    validation_rows, test_rows = train_test_split(
        remaining_rows,
        test_size=0.50,
        random_state=SEED,
        stratify=labels.iloc[remaining_rows],
    )

    assignments = np.full(len(targets), -1, dtype=np.int8)
    assignments[train_rows] = 0
    assignments[validation_rows] = 1
    assignments[test_rows] = 2
    output_paths = {name: raw_split_dir / name / path.name for name in SPLIT_NAMES}
    for output in output_paths.values():
        output.parent.mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in SPLIT_NAMES}
    offset = 0
    with pd.read_csv(path, dtype="string", na_filter=False, chunksize=CHUNK_SIZE,
                     on_bad_lines="error") as reader:
        for chunk in reader:
            end = offset + len(chunk)
            chunk_assignments = assignments[offset:end]
            for split_index, name in enumerate(SPLIT_NAMES):
                selected = chunk.loc[chunk_assignments == split_index]
                if not selected.empty:
                    selected.to_csv(
                        output_paths[name], index=False,
                        mode="a" if counts[name] else "w",
                        header=counts[name] == 0,
                    )
                    counts[name] += len(selected)
            offset = end

    if offset != len(targets) or (assignments < 0).any():
        raise ValueError(f"{path.name}: raw split did not preserve every row")
    print(
        f"[split] {path.name}: train={counts['train']:,}, "
        f"validation={counts['validation']:,}, test={counts['test']:,}",
        flush=True,
    )
    return output_paths


class SharedPreprocessor:
    """Zainab's numeric, categorical, and Boolean transforms fitted on train only."""

    def __init__(self):
        self.medians = None
        self.mappings = {}
        self.scaler = StandardScaler()

    def numeric_before_scaling(self, clean: pd.DataFrame) -> np.ndarray:
        values = clean[NUMERICAL_COLUMNS].to_numpy(dtype=np.float64, copy=True)
        values = np.where(np.isnan(values), self.medians, values)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite numeric value after training-median imputation")
        for column in LOG_COLUMNS:
            index = NUMERICAL_COLUMNS.index(column)
            if (values[:, index] < 0).any():
                raise ValueError(f"{column}: negative value before log1p")
            values[:, index] = np.log1p(values[:, index])
        return values

    def fit(self, train_paths: list[Path], work_dir: Path) -> None:
        category_values = {column: set() for column in CATEGORICAL_COLUMNS}
        value_paths = {column: work_dir / f"{column}.bin" for column in NUMERICAL_COLUMNS}
        counts = {column: 0 for column in NUMERICAL_COLUMNS}

        with ExitStack() as stack:
            handles = {
                column: stack.enter_context(value_paths[column].open("wb"))
                for column in NUMERICAL_COLUMNS
            }
            for path in train_paths:
                print(f"[preprocess] learning from training rows in {path.name}", flush=True)
                for chunk in read_chunks(path):
                    clean = clean_checked(chunk, path.name)
                    numeric = clean[NUMERICAL_COLUMNS]
                    for column in NUMERICAL_COLUMNS:
                        values = numeric[column].to_numpy(dtype=np.float64)
                        observed = values[np.isfinite(values)]
                        if observed.size:
                            observed.tofile(handles[column])
                            counts[column] += observed.size
                    for column in CATEGORICAL_COLUMNS:
                        category_values[column].update(clean[column].unique())

        if any(count == 0 for count in counts.values()):
            missing = [column for column, count in counts.items() if count == 0]
            raise ValueError(f"Training data has no observed values for {missing}")

        self.medians = np.empty(len(NUMERICAL_COLUMNS), dtype=np.float64)
        for index, column in enumerate(NUMERICAL_COLUMNS):
            count = counts[column]
            values = np.memmap(value_paths[column], dtype="float64", mode="r+", shape=(count,))
            try:
                low, high = (count - 1) // 2, count // 2
                values.partition((low, high))
                self.medians[index] = values[low] / 2 + values[high] / 2
                values.flush()
            finally:
                values._mmap.close()

        self.mappings = {
            column: {value: index + 2 for index, value in enumerate(sorted(values - {""}))}
            for column, values in category_values.items()
        }
        for path in train_paths:
            for chunk in read_chunks(path):
                clean = clean_checked(chunk, path.name)
                self.scaler.partial_fit(self.numeric_before_scaling(clean))
        print("[preprocess] fitted medians, category mappings, and scaler on train only", flush=True)

    def transform(self, clean: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=clean.index)
        output[NUMERICAL_COLUMNS] = self.scaler.transform(self.numeric_before_scaling(clean))
        for column in CATEGORICAL_COLUMNS:
            output[column] = clean[column].map(self.mappings[column]).fillna(1).astype("int32")
            output.loc[clean[column] == "", column] = 0
        for column in BOOLEAN_COLUMNS:
            output[column] = clean[column].astype("int8")
        output = output[FEATURE_COLUMNS]
        if not np.isfinite(output.to_numpy(dtype=np.float64)).all():
            raise ValueError("Nonfinite transformed feature")
        return output


def write_processed(paths: list[Path], output_dir: Path, processor: SharedPreprocessor) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        output = output_dir / path.name
        rows_written = 0
        for chunk in read_chunks(path):
            clean = clean_checked(chunk, path.name)
            labels, types = validate_targets(chunk)
            transformed = processor.transform(clean)
            transformed["label"] = labels.to_numpy()
            transformed["type"] = types.to_numpy()
            transformed.to_csv(
                output, index=False, mode="a" if rows_written else "w",
                header=rows_written == 0,
            )
            rows_written += len(chunk)
        print(f"[preprocess] wrote {path.name}: {rows_written:,} rows", flush=True)


def run_nf_unsw_pipeline(path: Path, output_dir: Path) -> Path:
    """Prepare the numeric-flow NF-UNSW-NB15 CSV for the shared model stage."""
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if not NF_TARGET_COLUMNS.issubset(columns):
        raise ValueError(f"{path.name}: missing NF-UNSW target columns {NF_TARGET_COLUMNS}")

    feature_columns = [column for column in columns if column not in NF_EXCLUDED_COLUMNS]
    if not feature_columns:
        raise ValueError(f"{path.name}: no NF-UNSW numeric feature columns found")

    targets = pd.read_csv(path, usecols=["Label"], dtype={"Label": "int8"})["Label"]
    if not targets.isin([0, 1]).all():
        raise ValueError(f"{path.name}: Label must contain only 0 and 1")
    rows = np.arange(len(targets))
    train_rows, remaining_rows = train_test_split(
        rows, test_size=0.30, random_state=SEED, stratify=targets
    )
    validation_rows, test_rows = train_test_split(
        remaining_rows, test_size=0.50, random_state=SEED,
        stratify=targets.iloc[remaining_rows],
    )
    assignments = np.full(len(targets), -1, dtype=np.int8)
    assignments[train_rows] = 0
    assignments[validation_rows] = 1
    assignments[test_rows] = 2

    with tempfile.TemporaryDirectory(prefix=".nf-pipeline-stage-", dir=PROJECT_ROOT) as temp_name:
        stage = Path(temp_name) / "data_split"
        output_paths = {
            split: stage / split / path.name for split in SPLIT_NAMES
        }
        for output in output_paths.values():
            output.parent.mkdir(parents=True, exist_ok=True)

        counts = {split: 0 for split in SPLIT_NAMES}
        offset = 0
        usecols = feature_columns + ["Label", "Attack"]
        with pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE,
                         na_filter=False, on_bad_lines="error") as reader:
            for chunk in reader:
                end = offset + len(chunk)
                chunk_assignments = assignments[offset:end]
                features = chunk[feature_columns].apply(pd.to_numeric, errors="coerce")
                features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                prepared = features
                prepared["label"] = chunk["Label"].astype("int8").to_numpy()
                prepared["type"] = chunk["Attack"].astype("string").str.strip().to_numpy()
                for split_index, split in enumerate(SPLIT_NAMES):
                    selected = prepared.loc[chunk_assignments == split_index]
                    if not selected.empty:
                        selected.to_csv(
                            output_paths[split], index=False,
                            mode="a" if counts[split] else "w",
                            header=counts[split] == 0,
                        )
                        counts[split] += len(selected)
                offset = end

        backup = Path(temp_name) / "previous_data_split"
        moved_old = False
        if output_dir.exists():
            os.replace(output_dir, backup)
            moved_old = True
        try:
            os.replace(stage, output_dir)
        except Exception:
            if moved_old and backup.exists():
                os.replace(backup, output_dir)
            raise

    print(
        f"[pipeline] NF-UNSW-NB15 prepared: train={counts['train']:,}, "
        f"validation={counts['validation']:,}, test={counts['test']:,}",
        flush=True,
    )
    return output_dir


def run_pipeline(raw_dir: Path = RAW_DATA_DIR, output_dir: Path = DATA_SPLIT_DIR) -> Path:
    raw_paths = sorted(Path(raw_dir).glob("*.csv"), key=lambda path: path.name.casefold())
    if not raw_paths:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}")
    print(f"[pipeline] found {len(raw_paths)} raw CSV files in {raw_dir}", flush=True)

    first_columns = set(pd.read_csv(raw_paths[0], nrows=0).columns)
    if NF_TARGET_COLUMNS.issubset(first_columns):
        if len(raw_paths) != 1:
            raise ValueError("NF-UNSW-NB15 mode expects one raw CSV file")
        return run_nf_unsw_pipeline(raw_paths[0], Path(output_dir))

    output_dir = Path(output_dir)
    with tempfile.TemporaryDirectory(prefix=".pipeline-stage-", dir=PROJECT_ROOT) as temp_name:
        stage = Path(temp_name)
        raw_split_dir = stage / "raw_splits"
        processed_stage = stage / "data_split"
        raw_paths_by_split = {name: [] for name in SPLIT_NAMES}

        for path in raw_paths:
            split_paths = split_raw_file(path, raw_split_dir)
            for name, split_path in split_paths.items():
                raw_paths_by_split[name].append(split_path)

        processor = SharedPreprocessor()
        with tempfile.TemporaryDirectory(prefix=".fit-work-", dir=stage) as work_name:
            processor.fit(raw_paths_by_split["train"], Path(work_name))
            for split_name in SPLIT_NAMES:
                print(f"[preprocess] transforming {split_name} split", flush=True)
                write_processed(
                    raw_paths_by_split[split_name],
                    processed_stage / split_name,
                    processor,
                )

        backup = stage / "previous_data_split"
        moved_old = False
        if output_dir.exists():
            os.replace(output_dir, backup)
            moved_old = True
        try:
            os.replace(processed_stage, output_dir)
        except Exception:
            if moved_old and backup.exists():
                os.replace(backup, output_dir)
            raise

    print(f"[pipeline] prepared train/validation/test CSVs in {output_dir}", flush=True)
    return output_dir
