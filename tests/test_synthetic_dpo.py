import unittest

from scripts.run_synthetic_dpo import evaluate, finite_difference_check, verify_protocol


class SyntheticDPOTests(unittest.TestCase):
    def test_protocol_matches_frozen_lock(self):
        spec, digest = verify_protocol()
        self.assertEqual(spec["protocol_id"], "vare-synthetic-contextual-dpo-cpu-v1")
        self.assertEqual(len(digest), 64)

    def test_analytical_gradient_matches_finite_differences(self):
        spec, _ = verify_protocol()
        result = finite_difference_check(spec)
        self.assertTrue(result["passed"], result)

    def test_reference_policy_ties_receive_half_credit(self):
        theta = [[0.0] * 8 for _ in range(4)]
        example = {"x": [1.0] + [0.0] * 7, "chosen": 0, "rejected": 1}
        result = evaluate(theta, [example])
        self.assertEqual(result["heldout_preference_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
