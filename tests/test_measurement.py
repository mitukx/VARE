import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.audit_scheduler import audit_measurement
from vare.runner import ROOT, encode, write_manifest


class MeasurementTests(unittest.TestCase):
    def test_retained_report_reconstructs(self):
        result = audit_measurement(ROOT / 'results/cpu-scheduler-v1')
        self.assertEqual(80, result['historical_evaluations'])
        self.assertEqual(16, result['reliability_cases'])

    def test_headline_edit_fails_even_with_new_manifest(self):
        with tempfile.TemporaryDirectory(prefix='vare-report-test-') as scratch:
            output = Path(scratch) / 'bundle'
            shutil.copytree(ROOT / 'results/cpu-scheduler-v1', output)
            path = output / 'summary.json'
            summary = json.loads(path.read_text())
            summary['median_paired_speedup'] = 1000
            path.write_bytes(encode(summary))
            (output / 'manifest.json').unlink()
            write_manifest(output)
            with self.assertRaises(ValueError):
                audit_measurement(output)
