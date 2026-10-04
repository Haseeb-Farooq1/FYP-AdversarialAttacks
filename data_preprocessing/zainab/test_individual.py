"""Small fixtures verifying independent per-file fitting and frozen reloads."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess import (
    EXPECTED_COLUMNS, FEATURE_COLUMNS, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS,
    BOOLEAN_COLUMNS, CompactPreprocessor, clean_features,
)
from .preprocess_individual import process_file


class IndependentFileTests(unittest.TestCase):
    def test_independent_fitting_retention_and_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for number, values, protocols in [(17, [0, 2, 4], ['tcp', 'udp', 'tcp']),
                                               (18, [100, 102, 104], ['icmp', 'tcp', 'tcp'])]:
                source = root / f'Network_dataset_{number}.csv'
                rows = []
                for index, value in enumerate(values):
                    row = dict.fromkeys(EXPECTED_COLUMNS, '-')
                    row.update({c: str(value) for c in NUMERICAL_COLUMNS})
                    row.update({c: '0' for c in CATEGORICAL_COLUMNS})
                    row.update({c: ['F', 'T', '-'][index] for c in BOOLEAN_COLUMNS})
                    row.update(proto=protocols[index], label='0', type='normal')
                    if index == 1:
                        row['src_bytes'] = '0.0.0.0'
                    rows.append(row)
                with source.open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                before = source.read_bytes()
                result = process_file(source, root/'output', root/'configs', chunk_size=2)
                self.assertEqual(source.read_bytes(), before)
                self.assertEqual(result['final_rows'], 3)
                self.assertEqual(result['invalid_src_bytes_count'], 1)
                output = pd.read_csv(root/'output'/source.name)
                self.assertEqual(list(output), FEATURE_COLUMNS + ['label', 'type'])
                self.assertTrue(np.isfinite(output[FEATURE_COLUMNS]).all().all())
                self.assertEqual(output.dns_AA.tolist(), [0, 1, 2])
                config = json.loads((root/'configs'/f'{source.stem}_config.json').read_text())
                self.assertFalse(config['split_performed'])
                self.assertNotIn('training_rows', config['standard_scaler'])
                self.assertEqual(config['standard_scaler']['fitted_rows'], 3)
                self.assertEqual(config['numeric_imputation_means']['src_bytes'], np.mean(values))
                processor = CompactPreprocessor.from_config(config)
                cleaned = clean_features(pd.DataFrame(rows, dtype='string'))
                np.testing.assert_allclose(processor.transform(cleaned), output[FEATURE_COLUMNS], atol=1e-12)
                unknown = cleaned.iloc[:1].copy()
                unknown.loc[:, 'proto'] = 'unseen'
                self.assertEqual(processor.transform(unknown).proto.iloc[0], 1)
                if number == 17:
                    first = config
                else:
                    self.assertNotEqual(first['standard_scaler']['mean'], config['standard_scaler']['mean'])
                    self.assertNotEqual(first['category_mappings']['proto']['tcp'], config['category_mappings']['proto']['tcp'])
            self.assertEqual(sorted(p.name for p in (root/'output').iterdir()),
                             ['Network_dataset_17.csv', 'Network_dataset_18.csv'])


if __name__ == '__main__':
    unittest.main()
