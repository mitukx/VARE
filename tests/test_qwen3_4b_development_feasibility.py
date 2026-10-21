from __future__ import annotations

from pathlib import Path

from scripts.run_qwen3_4b_mlsys_development_feasibility_v1 import (
    K,
    SEEDS,
    TASKS,
    _apply_and_grade,
    _binary_variance,
)


def test_development_feasibility_task_and_sampling_allocation_is_fixed():
    assert len(TASKS) == 3
    assert K == 4
    assert all(len(SEEDS[task]) == K for task in TASKS)
    assert len({seed for seeds in SEEDS.values() for seed in seeds}) == 12


def test_binary_reward_variance_uses_population_definition():
    assert _binary_variance([True, True, True, True]) == 0.0
    assert _binary_variance([False, False, False, False]) == 0.0
    assert _binary_variance([True, False, True, False]) == 0.25
    assert _binary_variance([]) == 0.0


def test_candidate_output_with_prose_is_rejected_as_format_failure(tmp_path: Path):
    result = _apply_and_grade(
        "unrecognized-task",
        {},
        tmp_path,
        "Here is my patch:\ndiff --git a/a.py b/a.py\n",
        tmp_path,
    )
    assert result["passed"] is False
    assert result["failure"] == "format_failure"
