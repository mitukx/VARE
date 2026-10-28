"""Action formats and the draft pass@8 training gate.

The gate is a draft rule for teammate review. It does not run a model.
``format_failure_rate`` is its own metric: a sample that is not one of the
three accepted formats is not relabeled as a wrong answer inside that rate.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

FORMATS = ("json_tool_call", "unified_diff", "fenced_file", "format_failure")
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FENCE = re.compile(r"```(?:[A-Za-z0-9_+-]*)?\n(.*?)```", re.DOTALL)


def _json_object(text: str) -> dict | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _is_unified_diff(text: str) -> bool:
    if "diff --git " in text:
        return True
    lines = text.splitlines()
    has_old = any(line.startswith("--- ") for line in lines)
    has_new = any(line.startswith("+++ ") for line in lines)
    has_hunk = any(line.startswith("@@") for line in lines)
    return has_old and has_new and has_hunk


def _is_json_tool_call(text: str) -> bool:
    match = _TOOL_CALL.search(text)
    if match is not None and _json_object(match.group(1)) is not None:
        return True
    payload = _json_object(text.strip())
    if payload is None:
        return False
    name = payload.get("name", payload.get("tool"))
    return name == "apply_patch" or (
        isinstance(payload.get("name"), str) and "arguments" in payload
    )


def _is_fenced_file(text: str) -> bool:
    match = _FENCE.search(text)
    if match is None:
        return False
    body = match.group(1).strip()
    if not body or _is_unified_diff(body):
        return False
    return any(line.strip() for line in body.splitlines())


def classify_action(text: str) -> str:
    """Return one of ``FORMATS``.

    A ``<tool_call>`` JSON object, or a single JSON object naming
    ``apply_patch``, is ``json_tool_call``. A unified diff, fenced or bare,
    is ``unified_diff``. Any other fenced block is ``fenced_file``.
    Everything else is ``format_failure``.
    """
    if _is_json_tool_call(text):
        return "json_tool_call"
    if _is_unified_diff(text):
        return "unified_diff"
    if _is_fenced_file(text):
        return "fenced_file"
    return "format_failure"


def format_failure_rate(texts: Sequence[str]) -> float:
    """Fraction of samples that match none of the three accepted formats."""
    if not texts:
        raise ValueError("format_failure_rate needs at least one sample")
    failures = sum(classify_action(text) == "format_failure" for text in texts)
    return failures / len(texts)


def training_gate_decision(
    *,
    correct_samples_per_task: Sequence[int],
    group_rewards: Sequence[Sequence[float]],
    samples_per_task: int = 8,
    minimum_tasks_with_a_hit: int = 2,
) -> dict[str, object]:
    """Draft pass@8 gate on the five train tasks.

    A task contributes a hit when at least one of its temperature samples is
    correct. The gate also requires nonzero reward variance inside at least
    one group of ``samples_per_task``. A group whose rewards are all equal,
    including a group of eight format failures, has variance 0 and does not
    by itself open the gate. Format failures are not counted here; record
    them with ``format_failure_rate``.
    """
    if len(correct_samples_per_task) != 5:
        raise ValueError("the draft gate is defined on 5 train tasks")
    if any(count < 0 or count > samples_per_task for count in correct_samples_per_task):
        raise ValueError("correct-sample counts must lie in 0..samples_per_task")
    if len(group_rewards) != 5:
        raise ValueError("expected one reward group per train task")
    variances: list[float] = []
    for group in group_rewards:
        if len(group) != samples_per_task:
            raise ValueError("each train group has samples_per_task rewards")
        mean = sum(group) / len(group)
        variances.append(sum((value - mean) ** 2 for value in group) / len(group))
    tasks_with_hit = sum(count >= 1 for count in correct_samples_per_task)
    nonzero = any(variance > 0.0 for variance in variances)
    return {
        "group_reward_variances": variances,
        "minimum_tasks_with_a_hit": minimum_tasks_with_a_hit,
        "nonzero_within_group_reward_variance": nonzero,
        "passed": tasks_with_hit >= minimum_tasks_with_a_hit and nonzero,
        "samples_per_task": samples_per_task,
        "tasks_with_a_correct_sample": tasks_with_hit,
    }
