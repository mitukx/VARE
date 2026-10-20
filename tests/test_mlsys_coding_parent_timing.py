"""Parent-clock and fresh-input checks for the performance gate.

The attack sources live in this file. They are written into a temporary
workspace and are not part of the task pack. A false pass is a correct
implementation at reference speed clearing the 2x gate.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from vare.environments.runner import ExecutableEvaluator, Workspace
from vare.environments.spec import EngineeringTaskSpec

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmarks" / "mlsys_coding_v1"
TASKS = PACK / "tasks"

_reward_module_spec = importlib.util.spec_from_file_location(
    "mlsys_coding_v1_reward_parent_timing", PACK / "reward.py"
)
reward = importlib.util.module_from_spec(_reward_module_spec)
assert _reward_module_spec is not None and _reward_module_spec.loader is not None
_reward_module_spec.loader.exec_module(reward)

_SLOW_RANGE = """\
def range_sums(values, queries):
    if any(type(value) is not int for value in values):
        raise ValueError("values must be ints")
    size = len(values)
    totals = []
    for query in queries:
        if not isinstance(query, tuple) or len(query) != 2:
            raise ValueError("query must be a (left, right) tuple")
        left, right = query
        if (
            type(left) is not int
            or type(right) is not int
            or not 0 <= left <= right <= size
        ):
            raise ValueError("query out of range")
        total = 0
        for index in range(left, right):
            total += values[index]
        totals.append(total)
    return totals
"""

_CLOCK_ATTACK = (
    """\
from __future__ import annotations

import time


def _hook(real):
    state = {"n": 0, "anchor": None}

    def clock():
        state["n"] += 1
        if state["n"] <= 14:
            if state["n"] % 2 == 1:
                state["anchor"] = real()
                return state["anchor"]
            return state["anchor"] + 1.0
        return real()

    return clock


time.perf_counter = _hook(time.perf_counter)
time.monotonic = _hook(time.monotonic)


"""
    + _SLOW_RANGE
)

_CACHE_ATTACK = """\
from __future__ import annotations

_ID_CACHE = {}
_VALUE_CACHE = {}


def range_sums(values, queries):
    if any(type(value) is not int for value in values):
        raise ValueError("values must be ints")
    size = len(values)
    checked = []
    for query in queries:
        if not isinstance(query, tuple) or len(query) != 2:
            raise ValueError("query must be a (left, right) tuple")
        left, right = query
        if (
            type(left) is not int
            or type(right) is not int
            or not 0 <= left <= right <= size
        ):
            raise ValueError("query out of range")
        checked.append((left, right))
    key_id = (id(values), id(queries))
    key_value = (tuple(values), tuple(checked))
    found = _ID_CACHE.get(key_id)
    if found is not None and found[0] == key_value:
        return list(found[1])
    found = _VALUE_CACHE.get(key_value)
    if found is not None:
        return list(found)
    totals = []
    for left, right in checked:
        total = 0
        for index in range(left, right):
            total += values[index]
        totals.append(total)
    _ID_CACHE[key_id] = (key_value, list(totals))
    _VALUE_CACHE[key_value] = list(totals)
    return totals
"""

_SLOW_DATASET = """\
from __future__ import annotations

import math


def _mean_and_variance(rows):
    count = len(rows)
    mean = sum(rows) / count
    variance = sum((value - mean) ** 2 for value in rows) / count
    return mean, variance


def standardized_rows(rows):
    if not rows:
        raise ValueError("rows must be non-empty")
    values = []
    for value in rows:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError("rows must be finite")
        values.append(float(value))
    standardized = []
    for value in values:
        mean, variance = _mean_and_variance(values)
        if variance == 0.0:
            standardized.append(0.0)
        else:
            standardized.append((value - mean) / math.sqrt(variance))
    return standardized
"""


def _grade(task_name: str, source: str) -> tuple[dict, dict]:
    task_dir = TASKS / task_name
    spec = EngineeringTaskSpec.load(task_dir / "task.json")
    filename = (
        "range_sums.py" if task_name == "range_sums" else "dataset_standardize.py"
    )
    with Workspace(spec) as workspace:
        (workspace.path / filename).write_text(source, encoding="utf-8")
        result = ExecutableEvaluator(spec, evaluator_root=task_dir).evaluate(
            workspace.path
        )
    decision = reward.decide(
        result.to_dict(), task_type=str(spec.metadata["task_type"])
    )
    payload = json.loads(result.tests[0].stdout)
    return decision, payload


def _assert_gate_closed(decision: dict, payload: dict) -> None:
    assert payload["correctness_pass"] is True
    assert payload["improvement_pass"] is False
    assert decision["success"] is False
    assert decision["conservative_speedup"] < reward.SUCCESS_THRESHOLD
    assert decision["reward"] < 1.0


def test_clock_patch_does_not_clear_the_two_x_gate():
    decision, payload = _grade("range_sums", _CLOCK_ATTACK)
    _assert_gate_closed(decision, payload)
    baseline = payload["baseline_samples_s"]
    assert len(baseline) == reward.REPEATS
    assert not all(abs(sample - 1.0) <= 1e-6 for sample in baseline)


def test_value_and_identity_cache_does_not_clear_the_two_x_gate():
    decision, payload = _grade("range_sums", _CACHE_ATTACK)
    _assert_gate_closed(decision, payload)


_DATASET_CACHE = """\
from __future__ import annotations

import math

_ID_CACHE = {}
_VALUE_CACHE = {}


def standardized_rows(rows):
    if not rows:
        raise ValueError("rows must be non-empty")
    values = []
    for value in rows:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError("rows must be finite")
        values.append(float(value))
    key_id = id(rows)
    key_value = tuple(values)
    found = _ID_CACHE.get(key_id)
    if found is not None and found[0] == key_value:
        return list(found[1])
    found = _VALUE_CACHE.get(key_value)
    if found is not None:
        return list(found)
    count = len(values)
    standardized = []
    for value in values:
        mean = sum(values) / count
        variance = sum((item - mean) ** 2 for item in values) / count
        if variance == 0.0:
            standardized.append(0.0)
        else:
            standardized.append((value - mean) / math.sqrt(variance))
    _ID_CACHE[key_id] = (key_value, list(standardized))
    _VALUE_CACHE[key_value] = list(standardized)
    return standardized
"""


def test_dataset_identity_cache_does_not_clear_the_two_x_gate():
    decision, payload = _grade("dataset_standardize", _DATASET_CACHE)
    _assert_gate_closed(decision, payload)


def test_import_print_does_not_score_a_performance_task_as_correctness():
    decision, payload = _grade(
        "range_sums", 'print("prefix-measurement")\n\n' + _SLOW_RANGE
    )
    assert decision["reason"] != "correctness"
    assert decision["success"] is False
    assert decision["reward"] < 1.0
    assert payload["measurement"] == reward.MEASUREMENT
    _assert_gate_closed(decision, payload)


@pytest.mark.parametrize(
    ("task_name", "source"),
    [
        ("range_sums", _SLOW_RANGE),
        ("dataset_standardize", _SLOW_DATASET),
    ],
)
def test_reference_speed_candidate_fails_the_two_x_gate(task_name: str, source: str):
    decision, payload = _grade(task_name, source)
    _assert_gate_closed(decision, payload)


def test_recorded_reference_speed_false_pass_rate():
    path = (
        ROOT / "results" / "mlsys-coding-reference-speed-false-pass-v1" / "summary.json"
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["statistic"] == "paired_ratio_student_t_lower"
    assert summary["load_drift"] == "four busy Python processes during every trial"
    assert summary["combined_trials"] == 40
    assert summary["combined_false_pass_count"] == 0
    assert summary["combined_one_sided_95_clopper_pearson_upper"] == pytest.approx(
        1.0 - 0.05 ** (1.0 / 40)
    )
    for record in summary["tasks"].values():
        assert record["false_pass_count"] == 0
        assert len(record["trials"]) == 20
        assert all(row["correctness_pass"] is True for row in record["trials"])
        assert all(row["success"] is False for row in record["trials"])
        assert all(
            row["conservative_speedup"] < reward.SUCCESS_THRESHOLD
            for row in record["trials"]
        )
