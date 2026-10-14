from pathlib import Path
import tempfile
import unittest

from benchmarks.historical.trl_grpo_accumulation_scale.evaluator.grade import (
    _execute_normalizer,
    _extract_normalizer,
)
from benchmarks.historical.trl_grpo_accumulation_scale_v9.evaluator.grade import (
    _extract_normalizer as _extract_normalizer_v9,
)
from benchmarks.historical.trl_grpo_accumulation_scale_v10.evaluator.grade import (
    _extract_normalizer as _extract_normalizer_v10,
)
from benchmarks.historical.trl_grpo_accumulation_scale_v11.evaluator.grade import (
    _extract_normalizer as _extract_normalizer_v11,
)
from benchmarks.historical.rvl_behavior_policy_parity.evaluator.grade import (
    FIXTURE_CASES,
    _Model as RVLModel,
    _Tensor as RVLTensor,
)
from benchmarks.historical.rvl_bad_words_neutrality.evaluator.grade import (
    FIXTURE_CASES as BAD_WORD_FIXTURE_CASES,
    _Model as BadWordsModel,
    _Tensor as BadWordsTensor,
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

    def test_trl_v9_accepts_non_mutating_tuple_alias_of_loss(self):
        source_text = VALID_BRANCH.replace(
            '            loss = (per_token_loss * mask).sum() / normalizer\n',
            '            loss = (per_token_loss * mask).sum() / normalizer\n'
            '            loss_alias, = (loss,)\n',
        ) + '        return loss\n'
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v9(source, "main_dapo_cispo_vespo")
            self.assertIsNotNone(assignment)
            self.assertIsNotNone(gate)
            self.assertIsNotNone(denominator)

    def test_trl_v9_rejects_inplace_mutation_through_tuple_destructured_loss_alias(self):
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            mutations = (
                ("tuple", "loss_alias, = (loss,)", "loss_alias"),
                ("list", "loss_alias, = [loss]", "loss_alias"),
                ("chained", "loss_alias = second_alias = loss", "second_alias"),
            )
            for label, alias, mutated_name in mutations:
                with self.subTest(alias_kind=label):
                    source_text = VALID_BRANCH.replace(
                        '            loss = (per_token_loss * mask).sum() / normalizer\n',
                        '            loss = (per_token_loss * mask).sum() / normalizer\n'
                        f'            {alias}\n'
                        f'            {mutated_name}.data.mul_(2)\n',
                    ) + '        return loss\n'
                    source.write_text(source_text, encoding="utf-8")
                    assignment, gate, denominator = _extract_normalizer_v9(
                        source, "main_dapo_cispo_vespo"
                    )
                    self.assertIsNone(assignment)
                    self.assertIsNone(gate)
                    self.assertIsNone(denominator)

    def test_trl_v10_accepts_non_mutating_tuple_alias_of_loss(self):
        source_text = VALID_BRANCH.replace(
            '            loss = (per_token_loss * mask).sum() / normalizer\n',
            '            loss = (per_token_loss * mask).sum() / normalizer\n'
            '            loss_alias, = (loss,)\n',
        ) + '        return loss\n'
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v10(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNotNone(assignment)
            self.assertIsNotNone(gate)
            self.assertIsNotNone(denominator)

    def test_trl_v10_rejects_storage_writes_through_tuple_loss_alias(self):
        mutations = (
            ("subscript", "loss_alias[...] = 0"),
            ("nested-subscript", "loss_alias[0], = [0]"),
            ("attribute", "loss_alias.data = 0"),
            ("augmented", "loss_alias += 1"),
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            for label, mutation in mutations:
                with self.subTest(write_kind=label):
                    source_text = VALID_BRANCH.replace(
                        '            loss = (per_token_loss * mask).sum() / normalizer\n',
                        '            loss = (per_token_loss * mask).sum() / normalizer\n'
                        '            loss_alias, = (loss,)\n'
                        f'            {mutation}\n',
                    ) + '        return loss\n'
                    source.write_text(source_text, encoding="utf-8")
                    assignment, gate, denominator = _extract_normalizer_v10(
                        source, "main_dapo_cispo_vespo"
                    )
                    self.assertIsNone(assignment)
                    self.assertIsNone(gate)
                    self.assertIsNone(denominator)

    def test_trl_v11_rejects_writes_through_derived_loss_views(self):
        mutations = (
            ("data", "loss_alias = loss.data\n            loss_alias[...] = 0"),
            ("view", "loss_alias = loss.view(-1)\n            loss_alias[0] = 0"),
            ("call-result", "loss_alias = loss.detach()\n            loss_alias[...] = 0"),
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            for label, mutation in mutations:
                with self.subTest(alias_kind=label):
                    source_text = VALID_BRANCH.replace(
                        '            loss = (per_token_loss * mask).sum() / normalizer\n',
                        '            loss = (per_token_loss * mask).sum() / normalizer\n'
                        f'            {mutation}\n',
                    ) + '        return loss\n'
                    source.write_text(source_text, encoding="utf-8")
                    assignment, gate, denominator = _extract_normalizer_v11(
                        source, "main_dapo_cispo_vespo"
                    )
                    self.assertIsNone(assignment)
                    self.assertIsNone(gate)
                    self.assertIsNone(denominator)

    def test_trl_v11_rejects_operator_setitem_through_loss_alias(self):
        mutations = (
            "import operator\n            operator.setitem(loss_alias, Ellipsis, 0)",
            "from operator import setitem as put_item\n            put_item(loss_alias, Ellipsis, 0)",
            "import operator\n            put_item = operator.setitem\n            put_item(loss_alias, Ellipsis, 0)",
            "torch.Tensor.__setitem__(loss_alias, Ellipsis, 0)",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                source_text = VALID_BRANCH.replace(
                    '            loss = (per_token_loss * mask).sum() / normalizer\n',
                    '            loss = (per_token_loss * mask).sum() / normalizer\n'
                    '            loss_alias, = (loss,)\n'
                    f'            {mutation}\n',
                ) + '        return loss\n'
                with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
                    source = Path(temporary) / "candidate.py"
                    source.write_text(source_text, encoding="utf-8")
                    assignment, gate, denominator = _extract_normalizer_v11(
                        source, "main_dapo_cispo_vespo"
                    )
                    self.assertIsNone(assignment)
                    self.assertIsNone(gate)
                    self.assertIsNone(denominator)

    def test_trl_v11_rejects_module_level_mutator_api_alias(self):
        source_text = (
            "import operator\n"
            "put_item = operator.setitem\n"
            + VALID_BRANCH.replace(
                '            loss = (per_token_loss * mask).sum() / normalizer\n',
                '            loss = (per_token_loss * mask).sum() / normalizer\n'
                '            loss_alias = loss\n'
                '            put_item(loss_alias, Ellipsis, 0)\n',
            )
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_control_flow_module_mutator_api_alias(self):
        source_text = (
            "import operator\n"
            "if True:\n"
            "    put_item = operator.setitem\n"
            + VALID_BRANCH.replace(
                '            loss = (per_token_loss * mask).sum() / normalizer\n',
                '            loss = (per_token_loss * mask).sum() / normalizer\n'
                '            loss_alias = loss\n'
                '            put_item(loss_alias, Ellipsis, 0)\n',
            )
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_conditional_expression_module_mutator_alias(self):
        source_text = (
            "import operator\n"
            "put_item = operator.setitem if True else operator.delitem\n"
            + VALID_BRANCH.replace(
                '            loss = (per_token_loss * mask).sum() / normalizer\n',
                '            loss = (per_token_loss * mask).sum() / normalizer\n'
                '            loss_alias = loss\n'
                '            put_item(loss_alias, Ellipsis, 0)\n',
            )
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_boolean_expression_module_mutator_alias(self):
        source_text = (
            "import operator\n"
            "put_item = operator.setitem or operator.delitem\n"
            + VALID_BRANCH.replace(
                '            loss = (per_token_loss * mask).sum() / normalizer\n',
                '            loss = (per_token_loss * mask).sum() / normalizer\n'
                '            loss_alias = loss\n'
                '            put_item(loss_alias, Ellipsis, 0)\n',
            )
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_getattr_module_mutator_alias(self):
        source_text = (
            "import operator\n"
            'put_item = getattr(operator, "setitem")\n'
            + VALID_BRANCH.replace(
                '            loss = (per_token_loss * mask).sum() / normalizer\n',
                '            loss = (per_token_loss * mask).sum() / normalizer\n'
                '            loss_alias = loss\n'
                '            put_item(loss_alias, Ellipsis, 0)\n',
            )
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_module_dict_mutator_alias(self):
        source_text = (
            "import operator\n"
            'put_item = operator.__dict__["setitem"]\n'
            + VALID_BRANCH.replace(
                '            loss = (per_token_loss * mask).sum() / normalizer\n',
                '            loss = (per_token_loss * mask).sum() / normalizer\n'
                '            loss_alias = loss\n'
                '            put_item(loss_alias, Ellipsis, 0)\n',
            )
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_module_dict_get_mutator_alias(self):
        source_text = (
            "import operator\n"
            'put_item = operator.__dict__.get("setitem")\n'
            + VALID_BRANCH.replace(
                '            loss = (per_token_loss * mask).sum() / normalizer\n',
                '            loss = (per_token_loss * mask).sum() / normalizer\n'
                '            loss_alias = loss\n'
                '            put_item(loss_alias, Ellipsis, 0)\n',
            )
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_qualified_getattr_module_mutator_alias(self):
        source_text = (
            "import operator\n"
            "import builtins\n"
            'put_item = builtins.getattr(operator, "setitem")\n'
            + VALID_BRANCH.replace(
                '            loss = (per_token_loss * mask).sum() / normalizer\n',
                '            loss = (per_token_loss * mask).sum() / normalizer\n'
                '            loss_alias = loss\n'
                '            put_item(loss_alias, Ellipsis, 0)\n',
            )
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_tuple_valued_out_alias(self):
        source_text = VALID_BRANCH.replace(
            '            loss = (per_token_loss * mask).sum() / normalizer\n',
            '            loss = (per_token_loss * mask).sum() / normalizer\n'
            '            loss_alias = loss\n'
            '            torch.frexp(torch.ones_like(loss), out=(loss_alias, torch.empty_like(loss, dtype=torch.int32)))\n',
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_out_call_result_through_container_alias(self):
        source_text = VALID_BRANCH.replace(
            '            loss = (per_token_loss * mask).sum() / normalizer\n',
            '            loss = (per_token_loss * mask).sum() / normalizer\n'
            '            holder = {"loss": loss}\n'
            '            torch.add(torch.ones_like(loss), 1.0, out=holder.get("loss"))\n',
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_dict_derived_loss_alias_store(self):
        source_text = VALID_BRANCH.replace(
            '            loss = (per_token_loss * mask).sum() / normalizer\n',
            '            loss = (per_token_loss * mask).sum() / normalizer\n'
            '            loss_alias = {"root": loss}["root"]\n'
            '            loss_alias[...] = 0\n',
        ) + '        return loss\n'
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_starred_tuple_alias_store(self):
        source_text = VALID_BRANCH.replace(
            '            loss = (per_token_loss * mask).sum() / normalizer\n',
            '            loss = (per_token_loss * mask).sum() / normalizer\n'
            '            loss_alias, *rest = (loss,)\n'
            '            loss_alias.data.mul_(2)\n',
        ) + '        return loss\n'
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_pattern_capture_alias_stores(self):
        mutations = (
            '        match [loss]:\n'
            '            case [loss_alias]:\n'
            '                loss_alias[...] = 0\n',
            '        match [normalizer]:\n'
            '            case [normalizer_alias]:\n'
            '                normalizer_alias[...] = 0\n',
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                source_text = VALID_BRANCH.replace('        return loss\n', mutation + '        return loss\n')
                with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
                    source = Path(temporary) / "candidate.py"
                    source.write_text(source_text, encoding="utf-8")
                    assignment, gate, denominator = _extract_normalizer_v11(
                        source, "main_dapo_cispo_vespo"
                    )
                    self.assertIsNone(assignment)
                    self.assertIsNone(gate)
                    self.assertIsNone(denominator)

    def test_trl_v11_rejects_comprehension_and_generator_alias_stores(self):
        mutations = (
            'loss_alias = next(item for item in (loss,))\n'
            '        loss_alias[...] = 0',
            'loss_alias = [item for item in (loss,)][0]\n'
            '        loss_alias[...] = 0',
            'loss_alias = {"root": item for item in (loss,)}["root"]\n'
            '        loss_alias[...] = 0',
            'normalizer_alias = next(item for item in (normalizer,))\n'
            '        normalizer_alias[...] = 0',
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    source_text = VALID_BRANCH.replace(
                        '            loss = (per_token_loss * mask).sum() / normalizer\n',
                        '            loss = (per_token_loss * mask).sum() / normalizer\n'
                        f'            {mutation}\n',
                    ) + '        return loss\n'
                    source.write_text(source_text, encoding="utf-8")
                    assignment, gate, denominator = _extract_normalizer_v11(
                        source, "main_dapo_cispo_vespo"
                    )
                    self.assertIsNone(assignment)
                    self.assertIsNone(gate)
                    self.assertIsNone(denominator)

    def test_trl_v11_allows_unrelated_pattern_capture_and_comprehension(self):
        source_text = VALID_BRANCH.replace(
            '        return loss\n',
            '        values = [item for item in (1,)]\n'
            '        match [mode]:\n'
            '            case [loss_alias]:\n'
            '                pass\n'
            '        return loss\n',
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNotNone(assignment)
            self.assertIsNotNone(gate)
            self.assertIsNotNone(denominator)

    def test_trl_v11_rejects_storage_write_through_normalizer_alias(self):
        source_text = VALID_BRANCH.replace(
            '            normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes\n',
            '            normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes\n'
            '            normalizer_alias, = (normalizer,)\n'
            '            normalizer_alias[...] = 0\n',
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_alias_store_in_for_target(self):
        source_text = VALID_BRANCH.replace(
            '            loss = (per_token_loss * mask).sum() / normalizer\n',
            '            loss = (per_token_loss * mask).sum() / normalizer\n'
            '            loss_alias, = (loss,)\n'
            '            for loss_alias[...] in [0]:\n'
            '                pass\n',
        ) + '        return loss\n'
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_store_through_with_bound_loss_alias(self):
        source_text = VALID_BRANCH.replace(
            '            loss = (per_token_loss * mask).sum() / normalizer\n',
            '            loss = (per_token_loss * mask).sum() / normalizer\n'
            '            from contextlib import nullcontext\n'
            '            with nullcontext(loss) as loss_alias:\n'
            '                loss_alias[...] = 0\n',
        ) + '        return loss\n'
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_alias_store_inside_allowed_entropy_adjustment(self):
        source_text = VALID_BRANCH.replace(
            '        return loss\n',
            '        if self._entropy_bonus_enabled:\n'
            '            loss_alias, = (loss,)\n'
            '            loss_alias[...] = 0\n'
            '            loss = loss - apply_coef * entropy_loss\n'
            '        return loss\n',
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_operator_setitem_inside_allowed_entropy_adjustment(self):
        source_text = VALID_BRANCH.replace(
            '        return loss\n',
            '        if self._entropy_bonus_enabled:\n'
            '            loss_alias, = (loss,)\n'
            '            import operator\n'
            '            operator.setitem(loss_alias, Ellipsis, 0)\n'
            '            loss = loss - apply_coef * entropy_loss\n'
            '        return loss\n',
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_normalizer_alias_write_inside_allowed_loss_adjustment(self):
        source_text = VALID_BRANCH.replace(
            '        return loss\n',
            '        if self._entropy_bonus_enabled:\n'
            '            normalizer_alias, = (normalizer,)\n'
            '            normalizer_alias[...] = 0\n'
            '            loss = loss - apply_coef * entropy_loss\n'
            '        return loss\n',
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_rejects_extra_write_inside_pinned_auxiliary_adjustment(self):
        source_text = VALID_BRANCH.replace(
            '        return loss\n',
            '        if self.aux_loss_enabled:\n'
            '            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0\n'
            '            normalizer_alias, = (normalizer,)\n'
            '            normalizer_alias[...] = 0\n'
            '            loss = loss + self.router_aux_loss_coef * aux_loss / normalizer\n'
            '        return loss\n',
        )
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
            self.assertIsNone(assignment)
            self.assertIsNone(gate)
            self.assertIsNone(denominator)

    def test_trl_v11_conservatively_rejects_mutated_clone_derived_from_loss(self):
        source_text = VALID_BRANCH.replace(
            '            loss = (per_token_loss * mask).sum() / normalizer\n',
            '            loss = (per_token_loss * mask).sum() / normalizer\n'
            '            loss_alias = loss.clone()\n'
            '            loss_alias[...] = 0\n',
        ) + '        return loss\n'
        with tempfile.TemporaryDirectory(prefix="vare-grader-probe-") as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(source_text, encoding="utf-8")
            assignment, gate, denominator = _extract_normalizer_v11(
                source, "main_dapo_cispo_vespo"
            )
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

    def test_bad_words_fixture_masks_and_observes_inherited_single_token(self):
        model = BadWordsModel(BAD_WORD_FIXTURE_CASES)
        prompt = BAD_WORD_FIXTURE_CASES[0]["prompt_ids"]
        model.generate(input_ids=BadWordsTensor([prompt]),
                       bad_words_ids=model.generation_config.bad_words_ids)
        self.assertEqual([[2]], model.effective_settings["bad_words_ids"])
        self.assertEqual(float("-inf"), model._transition[0])

        model.generate(input_ids=BadWordsTensor([prompt]), bad_words_ids=None,
                       suppress_tokens=None, no_repeat_ngram_size=0,
                       repetition_penalty=1.0, top_k=0, top_p=1.0, typical_p=1.0)
        self.assertIsNone(model.effective_settings["bad_words_ids"])
        self.assertNotEqual(float("-inf"), model._transition[0])


if __name__ == "__main__":
    unittest.main()
