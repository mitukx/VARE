"""Draft action formats and the pass@8 training gate. No model is loaded."""

from __future__ import annotations

import pytest

from vare.mlsys_coding_actions import (
    classify_action,
    format_failure_rate,
    training_gate_decision,
)

DIFF = """--- a/widget.py
+++ b/widget.py
@@ -1 +1 @@
-return 0
+return 1
"""


def test_three_action_formats_and_a_separate_failure_rate():
    tool = '<tool_call>{"name": "apply_patch", "arguments": {"patch": "x"}}</tool_call>'
    fenced = "```python\ndef add(x):\n    return x + 1\n```"
    bare_json = '{"name": "apply_patch", "arguments": {"patch": "x"}}'
    prose = "I think the function should return x + 1."
    assert classify_action(tool) == "json_tool_call"
    assert classify_action(bare_json) == "json_tool_call"
    assert classify_action(DIFF) == "unified_diff"
    assert classify_action("```diff\n" + DIFF + "```") == "unified_diff"
    assert classify_action(fenced) == "fenced_file"
    assert classify_action(prose) == "format_failure"
    assert format_failure_rate([tool, fenced, DIFF, prose, "   "]) == pytest.approx(0.4)


def test_training_gate_is_pass_at_8_not_greedy_success():
    constant_fail = [0.0] * 8
    one_hit = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    all_correct = [1.0] * 8
    opened = training_gate_decision(
        correct_samples_per_task=[1, 1, 0, 0, 0],
        group_rewards=[one_hit, one_hit, constant_fail, constant_fail, constant_fail],
    )
    assert opened["passed"] is True
    assert opened["tasks_with_a_correct_sample"] == 2
    assert opened["nonzero_within_group_reward_variance"] is True
    greedy_shape = training_gate_decision(
        correct_samples_per_task=[0, 0, 0, 0, 0],
        group_rewards=[constant_fail] * 5,
    )
    assert greedy_shape["passed"] is False
    saturated = training_gate_decision(
        correct_samples_per_task=[8, 8, 0, 0, 0],
        group_rewards=[
            all_correct,
            all_correct,
            constant_fail,
            constant_fail,
            constant_fail,
        ],
    )
    assert saturated["nonzero_within_group_reward_variance"] is False
    assert saturated["passed"] is False
