import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gsm8k_sequence_task import resolve_development_rank_ranges  # noqa: E402


class SequenceProtocolRangeTests(unittest.TestCase):
    def read_protocol(self, name):
        return json.loads((ROOT / "protocols" / name).read_text(encoding="utf-8"))

    def test_legacy_v1_uses_its_frozen_range(self):
        spec = self.read_protocol("cpu_lm_gsm8k_sequence_dpo_development_v1.json")
        self.assertEqual(resolve_development_rank_ranges(spec), ((608, 672), (672, 736)))

    def test_legacy_v2_uses_its_declared_range_not_v1_default(self):
        spec = self.read_protocol("cpu_lm_gsm8k_sequence_dpo_development_v2.json")
        self.assertEqual(resolve_development_rank_ranges(spec), ((1376, 1440), (1440, 1504)))

    def test_new_protocol_requires_both_explicit_ranges(self):
        spec = {
            "protocol_id": "vare-test",
            "dataset": {
                "development_training_examples": 2,
                "development_validation_examples": 3,
                "development_training_rank_range": [10, 12],
            },
        }
        with self.assertRaisesRegex(ValueError, "both training and validation"):
            resolve_development_rank_ranges(spec)

    def test_new_protocol_rejects_overlapping_ranges(self):
        spec = {
            "protocol_id": "vare-test",
            "dataset": {
                "development_training_examples": 2,
                "development_validation_examples": 2,
                "development_training_rank_range": [10, 12],
                "development_validation_rank_range": [11, 13],
            },
        }
        with self.assertRaisesRegex(ValueError, "overlap"):
            resolve_development_rank_ranges(spec)


if __name__ == "__main__":
    unittest.main()
