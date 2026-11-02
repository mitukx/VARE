import inspect

import pytest

from vare import grpo_mlx


@pytest.mark.parametrize(
    ("max_abs_log_ratio", "tolerance", "expected"),
    [(0.0, 0.0, True), (1e-4, 1e-3, True), (1e-3, 1e-3, True), (1.1e-3, 1e-3, False)],
)
def test_behavior_ratio_tolerance_boundary(max_abs_log_ratio, tolerance, expected):
    assert (
        grpo_mlx.behavior_ratio_within_tolerance(max_abs_log_ratio, tolerance)
        is expected
    )


@pytest.mark.parametrize(
    ("max_abs_log_ratio", "tolerance"),
    [
        (float("nan"), 1e-3),
        (float("inf"), 1e-3),
        (-1e-5, 1e-3),
        (0.0, float("nan")),
        (0.0, -1.0),
    ],
)
def test_behavior_ratio_tolerance_rejects_invalid_values(max_abs_log_ratio, tolerance):
    with pytest.raises(ValueError):
        grpo_mlx.behavior_ratio_within_tolerance(max_abs_log_ratio, tolerance)


def test_grpo_train_step_exposes_an_opt_in_pre_update_alignment_gate():
    parameter = inspect.signature(grpo_mlx.train_step).parameters[
        "max_abs_log_ratio_tolerance"
    ]
    assert parameter.default is None
