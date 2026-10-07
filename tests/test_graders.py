from pathlib import Path
import tempfile
import unittest

from benchmarks.historical.trl_grpo_accumulation_scale.evaluator.grade import (
    _execute_normalizer,
    _extract_normalizer,
)
from benchmarks.historical.rvl_behavior_policy_parity.evaluator.grade import (
    FIXTURE_CASES,
    _Model as RVLModel,
    _Tensor as RVLTensor,
)


VALID_BRANCH = '''
class GRPOTrainer:
    def _compute_loss(self):
        if self.loss_type in ["cispo", "dapo", "vespo"]:
            normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes
            if mode == "train":
                normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation
            loss = per_token_loss / normalizer
'''


class GraderIntegrityTests(unittest.TestCase):
    def test_trl_grader_rejects_later_normalizer_overwrite(self):
        mutated = VALID_BRANCH.replace(
            '            loss = per_token_loss / normalizer\n',
            '            normalizer = normalizer * 2\n            loss = per_token_loss / normalizer\n',
        )
        self.assertNotEqual(VALID_BRANCH, mutated)
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(mutated, encoding="utf-8")
            assignment, gate = _extract_normalizer(source, "main_dapo_cispo_vespo")
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(_execute_normalizer(source, "main_dapo_cispo_vespo", {
                "mode": "train", "items": 12, "world_size": 1,
                "current_accumulation_steps": 2, "steps_per_generation": 4,
            }))

    def test_trl_grader_keeps_accepting_locked_normalizer_contract(self):
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(VALID_BRANCH, encoding="utf-8")
            observed = _execute_normalizer(source, "main_dapo_cispo_vespo", {
                "mode": "train", "items": 12, "world_size": 1,
                "current_accumulation_steps": 2, "steps_per_generation": 4,
            })
            self.assertEqual(6.0, observed)

    def test_rvl_fixture_exposes_non_neutral_pretrained_typical_p(self):
        model = RVLModel(FIXTURE_CASES)
        prompt = FIXTURE_CASES[0]["prompt_ids"]
        model.generate(input_ids=RVLTensor([prompt]), typical_p=model.generation_config.typical_p)
        self.assertEqual(0.72, model.effective_settings["typical_p"])
        model.generate(input_ids=RVLTensor([prompt]), typical_p=1.0)
        self.assertEqual(1.0, model.effective_settings["typical_p"])


if __name__ == "__main__":
    unittest.main()
