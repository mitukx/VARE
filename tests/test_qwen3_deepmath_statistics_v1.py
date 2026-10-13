from __future__ import annotations

import pytest

from scripts.qwen3_deepmath_statistics_v1 import paired_task_seed_bootstrap


def test_constant_paired_effect_has_degenerate_intervals():
    candidate = [[0.6] * 20 for _ in range(5)]
    control = [[0.5] * 20 for _ in range(5)]
    result = paired_task_seed_bootstrap(candidate, control, seed=13, draws=200)
    assert result["mean_candidate_minus_control"] == pytest.approx(0.1)
    assert result["task_only_95_percentile_interval"] == pytest.approx([0.1, 0.1])
    assert result["seed_only_95_percentile_interval"] == pytest.approx([0.1, 0.1])
    assert result["hierarchical_task_seed_95_percentile_interval"] == pytest.approx([0.1, 0.1])


def test_bootstrap_is_reproducible_and_keeps_seed_deltas_paired():
    candidate = [[0.8, 0.2, 0.8], [0.2, 0.8, 0.2], [0.8, 0.8, 0.8], [0.2, 0.2, 0.2], [0.8, 0.2, 0.8]]
    control = [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]
    first = paired_task_seed_bootstrap(candidate, control, seed=17, draws=500)
    second = paired_task_seed_bootstrap(candidate, control, seed=17, draws=500)
    assert first == second
    assert first["seed_deltas"] == pytest.approx([0.4, 0.2, 0.6, 0.0, 0.4])
    assert first["task_only_95_percentile_interval"] != first["seed_only_95_percentile_interval"]


def test_bootstrap_rejects_unpaired_or_empty_matrices():
    with pytest.raises(ValueError):
        paired_task_seed_bootstrap([], [], seed=1, draws=10)
    with pytest.raises(ValueError):
        paired_task_seed_bootstrap([[1, 0]], [[0]], seed=1, draws=10)
