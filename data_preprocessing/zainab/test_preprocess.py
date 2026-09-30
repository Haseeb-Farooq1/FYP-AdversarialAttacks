"""Five essential tests, including a small end-to-end seven-file run."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .preprocess import (
    ATTACK_TYPES, BOOLEAN_COLUMNS, CATEGORICAL_COLUMNS, EXPECTED_COLUMNS,
    FEATURE_COLUMNS, FILENAMES, NUMERICAL_COLUMNS, CompactPreprocessor,
    clean_features, make_split_assignments, run_pipeline,
)


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.raw_dir = cls.root / "raw"
        cls.output_dir = cls.root / "processed"
        cls.artifact_dir = cls.root / "artifacts"
        cls.raw_dir.mkdir()
        type_codes = np.array([ATTACK_TYPES.index("normal" if i % 2 == 0 else "password")
                               for i in range(700)], dtype=np.uint8)
        held_out_index = int(np.flatnonzero(make_split_assignments(type_codes) == 2)[0])
        records = []
        for file_index, filename in enumerate(FILENAMES):
            with (cls.raw_dir / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
                writer.writeheader()
                for index in range(100):
                    row = {c: "-" for c in EXPECTED_COLUMNS}
                    row.update({c: str(index) for c in NUMERICAL_COLUMNS})
                    row.update({c: "0" for c in CATEGORICAL_COLUMNS})
                    row.update({c: "F" if index % 2 else "T" for c in BOOLEAN_COLUMNS})
                    row.update(proto=" tcp ", service="-", conn_state="SF",
                               label=str(index % 2), type="normal" if index % 2 == 0 else "password")
                    row["src_bytes"] = " 0.0.0.0 " if index == 0 else str(index + file_index)
                    if file_index * 100 + index == held_out_index:
                        row["proto"] = "heldout_only"
                        row["duration"] = "1000000000"
                    writer.writerow(row)
                    records.append(row)
        cls.raw = pd.DataFrame(records, dtype="string")
        cls.report = run_pipeline(cls.raw_dir, cls.output_dir, cls.artifact_dir, chunk_size=37)
        cls.config = json.loads((cls.artifact_dir / "preprocessing_config.json").read_text())
        cls.outputs = {name: pd.read_csv(cls.output_dir / f"{name}.csv")
                       for name in ("train", "validation", "test")}
        cls.assignments = np.load(cls.output_dir / "split_assignments.npy", allow_pickle=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_targets_are_not_in_x(self):
        self.assertEqual(len(FEATURE_COLUMNS), 23)
        self.assertFalse({"label", "type"}.intersection(FEATURE_COLUMNS))
        for output in self.outputs.values():
            self.assertEqual(list(output.columns), FEATURE_COLUMNS + ["label", "type"])

    def test_src_bytes_remains_numeric_and_rows_are_retained(self):
        self.assertEqual(self.report["invalid_src_bytes_count"], 7)
        self.assertEqual(self.report["original_rows"], self.report["final_rows"])
        self.assertEqual(self.report["final_rows"], 700)
        cleaned = clean_features(self.raw)
        self.assertEqual(int(cleaned.src_bytes.isna().sum()), 7)
        self.assertEqual(cleaned.loc[0, "src_port"], 0)
        for output in self.outputs.values():
            self.assertTrue(pd.api.types.is_numeric_dtype(output.src_bytes))

    def test_train_only_fitting_and_frozen_mapping(self):
        # Reconstruct the actual training records independently of the fit passes.
        training = clean_features(self.raw.loc[self.assignments == 0])
        numeric = training[NUMERICAL_COLUMNS].to_numpy()
        expected_means = np.nanmean(numeric, axis=0)
        learned_means = [self.config["numeric_imputation_means"][c] for c in NUMERICAL_COLUMNS]
        np.testing.assert_allclose(learned_means, expected_means)
        expected_scaler = StandardScaler().fit(np.where(np.isnan(numeric), expected_means, numeric))
        np.testing.assert_allclose(self.config["standard_scaler"]["mean"], expected_scaler.mean_)
        np.testing.assert_allclose(self.config["standard_scaler"]["scale"], expected_scaler.scale_)
        self.assertEqual(self.config["standard_scaler"]["training_rows"], len(training))
        self.assertNotIn("heldout_only", self.config["category_mappings"]["proto"])
        self.assertEqual(self.report["splits"]["test"]["unknown_categories"]["proto"], 1)

        processor = CompactPreprocessor.from_config(self.config)
        before = json.dumps(processor.to_config(), sort_keys=True)
        held_out = self.raw.iloc[:2].copy()
        held_out.loc[:, "src_bytes"] = "1000000000"
        held_out.loc[held_out.index[0], "proto"] = "unseen_protocol"
        held_out.loc[held_out.index[1], "proto"] = "-"
        held_out.loc[:, "dns_AA"] = "-"
        encoded = processor.transform(clean_features(held_out))
        self.assertEqual(encoded.proto.tolist(), [1, 0])
        self.assertEqual(encoded.dns_AA.tolist(), [2, 2])
        self.assertEqual(before, json.dumps(processor.to_config(), sort_keys=True))
        self.assertNotIn("unseen_protocol", processor.mappings["proto"])

    def test_split_is_reproducible_and_stratified(self):
        codes = pd.Categorical(self.raw.type, categories=ATTACK_TYPES).codes.astype(np.uint8)
        first = make_split_assignments(codes, 42)
        np.testing.assert_array_equal(first, make_split_assignments(codes, 42))
        np.testing.assert_array_equal(first, self.assignments)
        self.assertFalse(np.array_equal(first, make_split_assignments(codes, 43)))
        for code in np.unique(codes):
            np.testing.assert_array_equal(np.bincount(first[codes == code]), [245, 52, 53])

    def test_outputs_are_finite_and_codes_are_discrete(self):
        for output in self.outputs.values():
            self.assertFalse(output.isna().any().any())
            self.assertTrue(np.isfinite(output[FEATURE_COLUMNS].to_numpy()).all())
            for column in BOOLEAN_COLUMNS:
                self.assertTrue(output[column].isin([0, 1, 2]).all())
            for column in CATEGORICAL_COLUMNS:
                allowed = {0, 1, *self.config["category_mappings"][column].values()}
                self.assertTrue(output[column].isin(allowed).all())
        invalid = self.raw.iloc[:1].copy()
        invalid.loc[:, "duration"] = "inf"
        with self.assertRaises(ValueError):
            clean_features(invalid)


if __name__ == "__main__":
    unittest.main()
