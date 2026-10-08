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
            loss = (per_token_loss * mask).sum() / normalizer
        return loss
'''


class GraderIntegrityTests(unittest.TestCase):
    def test_trl_grader_rejects_later_normalizer_overwrite(self):
        mutated = VALID_BRANCH.replace(
            '            loss = (per_token_loss * mask).sum() / normalizer\n',
            '            normalizer = normalizer * 2\n            loss = (per_token_loss * mask).sum() / normalizer\n',
        )
        self.assertNotEqual(VALID_BRANCH, mutated)
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(mutated, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer(source, "main_dapo_cispo_vespo")
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)
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

    def test_trl_v4_grader_rejects_post_division_loss_reassignment(self):
        mutated = VALID_BRANCH.replace(
            '            loss = (per_token_loss * mask).sum() / normalizer\n',
            '            loss = (per_token_loss * mask).sum() / normalizer\n            loss = loss * 2\n',
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(mutated, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer(source, "main_dapo_cispo_vespo")
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)
            self.assertIsNone(_execute_normalizer(source, "main_dapo_cispo_vespo", {
                "mode": "train", "items": 12, "world_size": 1,
                "current_accumulation_steps": 2, "steps_per_generation": 4,
            }))

    def test_trl_v5_grader_rejects_final_loss_reassignment(self):
        mutated = VALID_BRANCH + "        loss = loss * 2\n        return loss\n"
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(mutated, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer(source, "main_dapo_cispo_vespo")
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v5_grader_allows_pinned_downstream_loss_adjustments(self):
        source_text = VALID_BRANCH + (
            '        if self._entropy_bonus_enabled:\n'
            '            loss = loss - apply_coef * entropy_loss\n'
            '        if self.aux_loss_enabled:\n'
            '            loss = loss + self.router_aux_loss_coef * aux_loss / normalizer\n'
            '        return loss\n'
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer(source, "main_dapo_cispo_vespo")
            self.assertIsNotNone(assignment)
            self.assertIsNotNone(gate)
            self.assertIsNotNone(denominator)

    def test_trl_v6_grader_rejects_inplace_normalizer_mutation(self):
        mutated = VALID_BRANCH.replace(
            '                normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation\n',
            '                normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation\n'
            '                normalizer.data.mul_(2)\n',
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(mutated, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer(source, "main_dapo_cispo_vespo")
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v6_grader_rejects_inplace_loss_mutation(self):
        mutated = VALID_BRANCH.replace(
            '            loss = (per_token_loss * mask).sum() / normalizer\n',
            '            loss = (per_token_loss * mask).sum() / normalizer\n            loss.data.mul_(2)\n',
        ) + '        return loss\n'
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(mutated, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer(source, "main_dapo_cispo_vespo")
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v6_grader_rejects_out_parameter_mutation(self):
        mutated = VALID_BRANCH.replace(
            '            loss = (per_token_loss * mask).sum() / normalizer\n',
            '            loss = (per_token_loss * mask).sum() / normalizer\n            torch.mul(loss, 2, out=loss)\n',
        ) + '        return loss\n'
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(mutated, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer(source, "main_dapo_cispo_vespo")
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v7_grader_rejects_inplace_mutation_through_loss_alias(self):
        mutated = VALID_BRANCH.replace(
            '            loss = (per_token_loss * mask).sum() / normalizer\n',
            '            loss = (per_token_loss * mask).sum() / normalizer\n            loss_alias = loss\n            loss_alias.data.mul_(2)\n',
        ) + '        return loss\n'
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(mutated, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer(source, "main_dapo_cispo_vespo")
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v7_grader_rejects_inplace_mutation_through_normalizer_alias(self):
        mutated = VALID_BRANCH.replace(
            '                normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation\n',
            '                normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation\n'
            '                normalizer_alias = normalizer\n'
            '                normalizer_alias.data.mul_(2)\n',
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(mutated, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer(source, "main_dapo_cispo_vespo")
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_grader_rejects_loss_divided_by_scaled_normalizer(self):
        mutated = VALID_BRANCH.replace(
            "loss = (per_token_loss * mask).sum() / normalizer",
            "loss = (per_token_loss * mask).sum() / (normalizer * 2)",
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(mutated, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer(source, "main_dapo_cispo_vespo")
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v8_rejects_zeroed_loss_numerator(self):
        mutated = VALID_BRANCH.replace(
            "loss = (per_token_loss * mask).sum() / normalizer",
            "loss = (per_token_loss * mask * 0.0).sum() / normalizer",
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(mutated, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer(source, "main_dapo_cispo_vespo")
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v8_rejects_unreachable_correct_branch_after_early_return(self):
        mutated = VALID_BRANCH.replace(
            "    def _compute_loss(self):\n",
            "    def _compute_loss(self):\n        return None\n",
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(mutated, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer(source, "main_dapo_cispo_vespo")
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_rvl_fixture_exposes_non_neutral_pretrained_typical_p(self):
        model = RVLModel(FIXTURE_CASES)
        prompt = FIXTURE_CASES[0]["prompt_ids"]
        model.generate(input_ids=RVLTensor([prompt]), typical_p=model.generation_config.typical_p)
        self.assertEqual(0.72, model.effective_settings["typical_p"])
        model.generate(input_ids=RVLTensor([prompt]), typical_p=1.0)
        self.assertEqual(1.0, model.effective_settings["typical_p"])

    def test_rvl_fixture_applies_and_observes_inherited_token_suppression(self):
        model = RVLModel(FIXTURE_CASES)
        prompt = FIXTURE_CASES[0]["prompt_ids"]
        model.generate(
            input_ids=RVLTensor([prompt]),
            suppress_tokens=model.generation_config.suppress_tokens,
            no_repeat_ngram_size=model.generation_config.no_repeat_ngram_size,
        )
        self.assertEqual([2], model.effective_settings["suppress_tokens"])
        self.assertEqual(2, model.effective_settings["no_repeat_ngram_size"])
        self.assertEqual(float("-inf"), model._transition[0])

        model.generate(input_ids=RVLTensor([prompt]), suppress_tokens=None, no_repeat_ngram_size=0)
        self.assertIsNone(model.effective_settings["suppress_tokens"])
        self.assertEqual(0, model.effective_settings["no_repeat_ngram_size"])
        self.assertNotEqual(float("-inf"), model._transition[0])


if __name__ == "__main__":
    unittest.main()
