"""Synthetic-only shared pipeline tests; never read the real dataset CSVs."""

import csv
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .preprocess import (
    BOOLEAN_COLUMNS, CATEGORICAL_COLUMNS, EXPECTED_COLUMNS, FEATURE_COLUMNS,
    NUMERICAL_COLUMNS, PROJECT_ROOT, clean_features,
)
from .preprocess_shared import (
    LOG_COLUMNS, PORT_COLUMNS, SharedPreprocessor, checked_clean, exact_disk_median,
    input_paths, run_shared, main, source_reference,
)


class SharedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Workspace-local fixtures also work when Windows system Temp is sandboxed.
        cls.temp = tempfile.TemporaryDirectory(prefix=".shared-test-", dir=PROJECT_ROOT / "dataset")
        cls.root = Path(cls.temp.name)
        cls.raw_dir = cls.root / "raw"
        cls.raw_dir.mkdir()
        cls.paths = input_paths(cls.raw_dir)
        cls.records = []
        cls.hashes = {}
        for file_index, path in enumerate(cls.paths):
            rows = []
            for i in range(8):
                row = dict.fromkeys(EXPECTED_COLUMNS, "-")
                row.update({c: str(i + file_index * 10) for c in NUMERICAL_COLUMNS})
                row.update({c: "0" for c in CATEGORICAL_COLUMNS})
                row.update({c: ["F", "T", "-"][i % 3] for c in BOOLEAN_COLUMNS})
                row.update(ts=f"{file_index}.{i}", src_ip="192.0.2.1", dst_ip="192.0.2.2",
                           proto=[" tcp ", "udp", "icmp"][i % 3],
                           service="http" if i % 2 else "-", conn_state="SF",
                           label=str(i % 2), type="mitm" if i % 2 else "normal")
                row["duration"] = str((i + file_index) / 10)
                row["http_request_body_len"] = "0"
                if i == 1:
                    row["src_bytes"] = "0.0.0.0"
                if i == 2:
                    row["dst_bytes"] = "-"
                if i == 6:
                    row["src_bytes"] = "1000000000"
                rows.append(row)
            # Exact duplicate within every source and across sources; retain all.
            rows[-1] = rows[0].copy()
            if file_index:
                rows[0] = cls.records[0][0].copy()
            cls.records.append(rows)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            cls.hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        cls.out = cls.root / "outputs"
        cls.art = cls.root / "artifacts"
        cls.report = run_shared(cls.paths, cls.out, cls.art, cls.root / "report.md", chunk_size=3, write_metadata=True)
        cls.config = json.loads((cls.art / "preprocessing_config.json").read_text())
        cls.processor = SharedPreprocessor.from_config(cls.config)
        cls.raw = pd.DataFrame([r for rows in cls.records for r in rows], dtype="string")
        cls.clean = clean_features(cls.raw)
        cls.outputs = [pd.read_csv(cls.out / p.name) for p in cls.paths]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_feature_contract_and_no_splits(self):
        expected = "src_port dst_port proto service duration src_bytes dst_bytes conn_state missed_bytes src_pkts src_ip_bytes dst_pkts dst_ip_bytes dns_qclass dns_qtype dns_rcode dns_AA dns_RD dns_RA dns_rejected http_request_body_len http_response_body_len http_status_code".split()
        self.assertEqual(FEATURE_COLUMNS, expected)
        self.assertEqual(self.config["feature_columns"], expected)
        self.assertEqual(len(expected), 23)
        self.assertFalse(set(expected) & {"label", "type", "ts", "src_ip", "dst_ip", "source_file", "source_row"})
        for out in self.outputs:
            self.assertEqual(list(out), expected + ["label", "type"])
        self.assertEqual({p.name for p in self.out.iterdir()}, {p.name for p in self.paths} | {"metadata.csv"})
        self.assertFalse(self.config["split_performed"])
        self.assertNotIn("class_weights", self.config)

    def test_shared_medians_and_scaler_against_independent_calculation(self):
        a = self.clean[NUMERICAL_COLUMNS].to_numpy()
        medians = np.nanmedian(a, axis=0)
        np.testing.assert_array_equal(self.processor.medians, medians)
        a = np.where(np.isnan(a), medians, a)
        for c in LOG_COLUMNS:
            i = NUMERICAL_COLUMNS.index(c)
            a[:, i] = np.log1p(a[:, i])
        expected = StandardScaler().fit(a)
        np.testing.assert_allclose(self.processor.scaler.mean_, expected.mean_, atol=1e-12)
        np.testing.assert_allclose(self.processor.scaler.scale_, expected.scale_, atol=1e-12)
        combined = pd.concat(self.outputs, ignore_index=True)
        np.testing.assert_allclose(combined[NUMERICAL_COLUMNS], expected.transform(a), atol=1e-12)
        self.assertEqual(self.config["standard_scaler"]["fitted_rows"], len(a))
        self.assertNotIn("training_rows", self.config["standard_scaler"])

    def test_raw_preservation_targets_and_accounting(self):
        for path, original, output in zip(self.paths, self.records, self.outputs):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), self.hashes[path.name])
            self.assertEqual(len(output), len(original))
            self.assertEqual(output.label.tolist(), [int(r["label"]) for r in original])
            self.assertEqual(output.type.tolist(), [r["type"] for r in original])
        self.assertEqual(self.report["original_rows"], self.report["final_rows"])
        self.assertEqual(self.report["final_rows"], 56)
        self.assertEqual(self.report["invalid_src_bytes_count"], 7)
        self.assertEqual(self.clean.src_bytes.isna().sum(), 7)
        self.assertEqual(self.clean.src_bytes.iloc[0], 0)
        self.assertTrue(pd.api.types.is_numeric_dtype(self.outputs[0].src_bytes))

    def test_output_finiteness_and_boolean_codes(self):
        for out, rows in zip(self.outputs, self.records):
            self.assertFalse(out.isna().any().any())
            self.assertTrue(np.isfinite(out[FEATURE_COLUMNS].to_numpy()).all())
            for c in BOOLEAN_COLUMNS:
                self.assertEqual(out[c].tolist(), [{"F": 0, "T": 1, "-": 2}[r[c]] for r in rows])

    def test_shared_categories_missing_unknown_and_frozen_reload(self):
        before = json.dumps(self.processor.to_config(), sort_keys=True)
        for c in CATEGORICAL_COLUMNS:
            values = sorted(set(self.clean[c]) - {""})
            self.assertEqual(self.processor.mappings[c], {v: i for i, v in enumerate(values, 2)})
        for raw, output in zip(self.records, self.outputs):
            frame = checked_clean(pd.DataFrame(raw, dtype="string"), "fixture")
            np.testing.assert_allclose(self.processor.transform(frame), output[FEATURE_COLUMNS], atol=1e-12)
            self.assertEqual(output.service.iloc[1], self.processor.mappings["service"]["http"])
        probe = self.clean.iloc[:2].copy()
        probe.loc[probe.index[0], "service"] = "never_seen"
        probe.loc[probe.index[1], "service"] = ""
        encoded = self.processor.transform(probe)
        self.assertEqual(encoded.service.tolist(), [1, 0])
        self.assertEqual(self.processor.decode_categories(encoded).service.tolist(), ["<unknown>", "<missing>"])
        encoded["service"] = encoded["service"].astype("float64")
        encoded.loc[encoded.index[0], "service"] = 1.5
        with self.assertRaisesRegex(ValueError, "invalid discrete"):
            self.processor.decode_categories(encoded)
        self.assertEqual(json.dumps(self.processor.to_config(), sort_keys=True), before)

    def test_duplicate_counts_against_raw_tuples(self):
        tuples = [tuple(row[c] for c in EXPECTED_COLUMNS) for rows in self.records for row in rows]
        counts = Counter(tuples)
        d = self.report["duplicates"]
        self.assertEqual(d["duplicate_excess_rows"], len(tuples) - len(counts))
        self.assertAlmostEqual(d["duplicate_percentage"], 100 * (len(tuples) - len(counts)) / len(tuples))
        self.assertEqual(d["rows_in_duplicate_groups"], sum(n for n in counts.values() if n > 1))
        locations = {}
        for file_index, rows in enumerate(self.records):
            for row in rows:
                locations.setdefault(tuple(row[c] for c in EXPECTED_COLUMNS), set()).add(file_index)
        cross = [row for row, files in locations.items() if len(files) > 1]
        self.assertEqual(d["cross_file_duplicate_groups"], len(cross))
        self.assertEqual(d["rows_in_cross_file_groups"], sum(counts[row] for row in cross))
        by_type = Counter()
        for row, n in counts.items():
            by_type[row[-1]] += n - 1
        self.assertEqual(d["duplicate_excess_by_type"], dict(by_type))
        self.assertEqual(d["removed_rows"], 0)
        self.assertTrue(d["retained"])
        for stats, rows in zip(self.report["files"], self.records):
            local = {tuple(row[c] for c in EXPECTED_COLUMNS) for row in rows}
            self.assertEqual(stats["within_file_duplicate_excess"], len(rows) - len(local))
        self.assertEqual(sum(f["global_duplicate_excess_attributed_to_file"] for f in self.report["files"]), d["duplicate_excess_rows"])

    def test_inverse_and_no_log_ports(self):
        self.assertEqual(LOG_COLUMNS, [c for c in NUMERICAL_COLUMNS if c not in PORT_COLUMNS])
        self.assertEqual(len(LOG_COLUMNS), 10)
        processed = self.processor.transform(self.clean)
        recovered = self.processor.inverse_numerical(processed)
        original = self.clean[NUMERICAL_COLUMNS].to_numpy()
        expected = np.where(np.isnan(original), self.processor.medians, original)
        np.testing.assert_allclose(recovered, expected, rtol=1e-10, atol=1e-10)
        before_scaling = self.processor.numeric_before_scaling(self.clean)
        np.testing.assert_array_equal(before_scaling[:, :2], expected[:, :2])

    def test_negative_boolean_and_nonfinite_diagnostics(self):
        for c in LOG_COLUMNS:
            with self.subTest(feature=c):
                raw = self.raw.iloc[:1].copy()
                raw.loc[:, c] = "-1"
                with self.assertRaisesRegex(ValueError, f"{c}.*negative"):
                    checked_clean(raw, "negative_fixture.csv")
                clean = self.clean.iloc[:1].copy()
                clean.loc[:, c] = -1
                with self.assertRaisesRegex(ValueError, f"{c}.*negative.*log1p"):
                    self.processor.numeric_before_scaling(clean)
        raw = self.raw.iloc[:1].copy()
        raw.loc[:, "dns_AA"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "dns_AA.*unexpected"):
            checked_clean(raw, "bad_boolean.csv")
        raw = self.raw.iloc[:1].copy()
        raw.loc[:, "duration"] = "inf"
        with self.assertRaisesRegex(ValueError, "duration.*infinite"):
            checked_clean(raw, "infinite.csv")

    def test_metadata_and_fit_manifest(self):
        self.assertTrue(self.report["metadata_written"])
        self.assertEqual(self.report["metadata_rows"], 56)
        metadata = pd.read_csv(self.out / "metadata.csv", dtype={"ts": "string"})
        self.assertEqual(list(metadata), ["source_file", "source_row", "ts"])
        self.assertEqual(len(metadata), 56)
        for path, rows in zip(self.paths, self.records):
            frame = metadata.loc[metadata.source_file == path.name]
            self.assertEqual(frame.source_row.tolist(), list(range(1, 9)))
            self.assertEqual(frame.ts.tolist(), [r["ts"] for r in rows])
        digest = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.assertTrue(all(f["shared_config_sha256"] == digest for f in self.report["files"]))
        feature_meta = json.loads((self.art / "feature_metadata.json").read_text())
        self.assertEqual([f["feature"] for f in feature_meta["features"]], FEATURE_COLUMNS)
        self.assertTrue(all(f["adversarial_mutability"] == "review_required" for f in feature_meta["features"]))
        self.assertEqual(set(p.name for p in self.art.iterdir()), {"preprocessing_config.json", "preprocessing_report.json", "feature_metadata.json"})

    def test_exact_median_even_odd_constant_and_missing(self):
        for values in ([9., 0., 2., 100.], [4., 1., 8.], [0., 0., 0.]):
            p = self.root / "median-test.bin"
            np.array(values, dtype="float64").tofile(p)
            self.assertEqual(exact_disk_median(p), float(np.median(values)))
        p = self.root / "empty.bin"
        p.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "entirely missing"):
            exact_disk_median(p)

    def test_input_parameterization_and_overwrite_protection(self):
        full = input_paths(self.raw_dir, 1, 23)
        self.assertEqual(len(full), 23)
        self.assertEqual(full[-1].name, "Network_dataset_23(in).csv")
        self.assertEqual(full[0].name, "Network_dataset_1.csv")
        with self.assertRaises(ValueError):
            input_paths(self.raw_dir, 23, 17)
        with self.assertRaises(FileExistsError):
            run_shared(self.paths, self.out, self.art, self.root / "report.md")

    def test_fatal_input_stops_before_exports_and_config(self):
        for column, value, diagnostic in [("duration", "-1", "negative"),
                                          ("dns_AA", "not_a_boolean", "unexpected Boolean")]:
            with self.subTest(column=column):
                bad = self.root / f"bad_{column}.csv"
                row = self.records[0][0].copy()
                row[column] = value
                with bad.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
                    writer.writeheader()
                    writer.writerow(row)
                out, art = self.root / f"bad_out_{column}", self.root / f"bad_art_{column}"
                with self.assertRaisesRegex(ValueError, diagnostic):
                    run_shared([bad], out, art, self.root / f"bad_{column}.md", chunk_size=2)
                self.assertFalse(list(out.iterdir()))
                self.assertFalse(art.exists())

    def test_reject_changed_saved_feature_order(self):
        config = json.loads(json.dumps(self.config))
        config["feature_columns"] = list(reversed(config["feature_columns"]))
        with self.assertRaisesRegex(ValueError, "feature order"):
            SharedPreprocessor.from_config(config)

    def test_deterministic_shared_fit_across_file_order_and_chunks(self):
        out = self.root / "reverse_outputs"
        art = self.root / "reverse_artifacts"
        report = run_shared(list(reversed(self.paths)), out, art, self.root / "reverse.md", chunk_size=5)
        self.assertFalse(report["metadata_written"])
        self.assertEqual(report["metadata_rows"], 0)
        self.assertEqual({p.name for p in out.iterdir()}, {p.name for p in self.paths})
        config = json.loads((art / "preprocessing_config.json").read_text())
        self.assertEqual(config["category_mappings"], self.config["category_mappings"])
        self.assertEqual(config["numeric_imputation_medians"], self.config["numeric_imputation_medians"])
        np.testing.assert_allclose(config["standard_scaler"]["mean"], self.config["standard_scaler"]["mean"], atol=1e-12)
        for path, expected in zip(self.paths, self.outputs):
            actual = pd.read_csv(out / path.name)
            np.testing.assert_allclose(actual[FEATURE_COLUMNS], expected[FEATURE_COLUMNS], atol=1e-12)
            self.assertEqual(list(actual), FEATURE_COLUMNS + ["label", "type"])
            self.assertEqual(actual[["label", "type"]].to_dict("list"), expected[["label", "type"]].to_dict("list"))

    def test_metadata_cli_flag_and_portable_paths(self):
        for flags, enabled in [([], False), (["--write-metadata"], True)]:
            with patch("sys.argv", ["preprocess_shared", *flags]), patch(
                "data_preprocessing.zainab.preprocess_shared.run_shared"
            ) as run:
                main()
                self.assertEqual(run.call_args.kwargs["write_metadata"], enabled)
        for item in self.config["source_files"]:
            self.assertFalse(Path(item["source_path"]).is_absolute())
            self.assertEqual(item["source_path"], source_reference(Path(item["source_path"])))
        self.assertEqual(source_reference(PROJECT_ROOT.parent / "external.csv"), "external.csv")


if __name__ == "__main__":
    unittest.main()
