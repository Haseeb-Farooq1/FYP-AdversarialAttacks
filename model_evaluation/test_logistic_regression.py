"""Small synthetic tests; never read or modify project datasets."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model_evaluation import logistic_regression as lr


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def csv(self, name="data.csv", **columns):
        path = self.root / name
        pd.DataFrame(columns or {"x": [1., 2.], "label": [0, 1]}).to_csv(path, index=False)
        return path

    def test_discovery_excludes_reserved_files(self):
        self.csv("b.csv")
        self.csv("a.CSV")
        for name in lr.NON_MODEL_FILES:
            self.csv(name)
        self.assertEqual([p.name for p in lr.discover_files(self.root)], ["a.CSV", "b.csv"])

    def test_empty_discovery_fails(self):
        with self.assertRaises(ValueError):
            lr.discover_files(self.root)

    def test_targets_excluded_and_type_optional(self):
        a = self.csv("a.csv", x=[1, 2], label=[0, 1], type=["normal", "attack"])
        b = self.csv("b.csv")
        features, y, counts = lr.inspect_files([a, b])
        self.assertEqual(features, ["x"])
        self.assertEqual(counts, [2, 2])
        np.testing.assert_array_equal(y, [0, 1, 0, 1])

    def test_incompatible_schema_fails(self):
        a = self.csv()
        b = self.csv("other.csv", different=[1, 2], label=[0, 1])
        with self.assertRaisesRegex(ValueError, "schema"):
            lr.inspect_files([a, b])

    def test_feature_order_mismatch_fails(self):
        a = self.csv("a.csv", x=[1, 2], z=[3, 4], label=[0, 1])
        b = self.csv("b.csv", z=[3, 4], x=[1, 2], label=[0, 1])
        with self.assertRaisesRegex(ValueError, "schema"):
            lr.inspect_files([a, b])

    def test_missing_or_invalid_labels_fail(self):
        for columns in ({"x": [1, 2]}, {"x": [1, 2], "label": [0, 2]},
                        {"x": [1, 2], "label": [0, np.nan]}):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                lr.inspect_files([self.csv(**columns)])

    def test_duplicate_headers_fail(self):
        path = self.root / "data.csv"
        path.write_text("x,x,label\n1,2,0\n3,4,1\n")
        with self.assertRaises(ValueError):
            lr.inspect_files([path])

    def test_nonnumeric_predictor_fails(self):
        with self.assertRaisesRegex(ValueError, "nonnumeric"):
            lr.inspect_files([self.csv(x=["bad", "value"], label=[0, 1])])

    def test_nan_and_infinity_fail(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, "NaN/infinite"):
                lr.inspect_files([self.csv(x=[1, bad], label=[0, 1])])

    def test_sampling_reproducible_without_replacement(self):
        y = np.array([0] * 101 + [1] * 899, dtype=np.uint8)
        first = lr.sample_indices(y, 333)
        np.testing.assert_array_equal(first, lr.sample_indices(y, 333))
        self.assertEqual(len(np.unique(first)), 333)
        self.assertLessEqual(abs(np.count_nonzero(y[first] == 0) - 333 * .101), .5)

    def test_full_mode_and_invalid_sample_sizes(self):
        y = np.array([0, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(lr.sample_indices(y, None), [0, 1, 2])
        for size in (0, 1, 4):
            with self.assertRaises(ValueError):
                lr.sample_indices(y, size)

    def test_selected_loader_preserves_values_and_order(self):
        a = self.csv("a.csv", x=[1., 2.], label=[0, 1])
        b = self.csv("b.csv", x=[3., 4.], label=[0, 1])
        X = lr.load_selected([a, b], ["x"], np.array([0, 3]), [2, 2])
        np.testing.assert_array_equal(X, [[1.], [4.]])

    def test_exact_grouping_conflicts_and_reproducible_split(self):
        X = np.repeat(np.arange(100, dtype=float), 2).reshape(-1, 1)
        y = np.repeat(np.arange(100) % 2, 2).astype(np.uint8)
        y[0] = 1  # One identical-X group with conflicting labels.
        X = np.vstack([X, [[np.nextafter(99., np.inf)]]])  # Close is not equal.
        y = np.append(y, np.uint8(1))
        train, test, stats = lr.grouped_holdout(X, y)
        train2, test2, _ = lr.grouped_holdout(X, y)
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(test, test2)
        self.assertEqual(stats["distinct_predictor_groups"], 101)
        self.assertEqual(stats["duplicate_groups"], 100)
        self.assertEqual(stats["repeated_rows_beyond_first"], 100)
        self.assertEqual(stats["conflicting_label_groups"], 1)
        self.assertEqual(stats["rows_in_conflicting_label_groups"], 2)
        self.assertEqual(stats["crossing_groups"], 0)
        self.assertFalse(set(X[train, 0]) & set(X[test, 0]))
        self.assertEqual(len(train) + len(test), len(y))
        self.assertEqual(y[0], 1)

    def test_metrics(self):
        metrics = lr.calculate_metrics([0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 0, 1, 1, 1])
        self.assertEqual(metrics["confusion_matrix"], [[2, 1], [1, 3]])
        self.assertAlmostEqual(metrics["accuracy"], 5/7)
        self.assertAlmostEqual(metrics["balanced_accuracy"], (2/3 + 3/4)/2)
        self.assertAlmostEqual(metrics["attack_precision"], .75)
        self.assertAlmostEqual(metrics["macro_f1"], (2/3 + .75)/2)
        self.assertAlmostEqual(metrics["normal_false_positive_rate"], 1/3)
        self.assertAlmostEqual(metrics["attack_false_negative_rate"], .25)

    def test_serialization(self):
        result = {"metrics": lr.calculate_metrics([0, 1], [0, 1]), "seed": 42}
        path = self.root / "results" / "result.json"
        lr.save_result(result, path)
        self.assertEqual(json.loads(path.read_text()), result)

    def test_small_end_to_end_experiment(self):
        self.csv(x=np.linspace(-2, 2, 200), label=[0] * 80 + [1] * 120)
        result = lr.run(self.root, 100)
        self.assertEqual(result["rows_used"], 100)
        self.assertEqual(result["duplicates"]["crossing_groups"], 0)
        self.assertEqual(result["train_rows"] + result["test_rows"], 100)
        self.assertTrue(result["convergence"]["converged"])
        path = self.root / "result.json"
        lr.save_result(result, path)
        self.assertEqual(json.loads(path.read_text())["model"], "LogisticRegression")


if __name__ == "__main__":
    unittest.main()
