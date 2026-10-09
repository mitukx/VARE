import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cpu_hh_length_normalized_dpo_development_v1 import length_normalized_relative_margin


class LengthNormalizedDPOObjectiveTests(unittest.TestCase):
    def test_divides_sequence_margin_by_mean_response_target_tokens(self):
        self.assertAlmostEqual(length_normalized_relative_margin(12.0, 5, 3), 3.0)

    def test_eos_inclusive_target_lengths_are_required(self):
        self.assertAlmostEqual(length_normalized_relative_margin(8.0, 2, 6), 2.0)

    def test_empty_response_targets_are_rejected(self):
        for counts in ((0, 2), (2, 0), (0, 0)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                length_normalized_relative_margin(1.0, *counts)


if __name__ == "__main__":
    unittest.main()
