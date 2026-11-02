import inspect
import sys
from types import ModuleType, SimpleNamespace

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


def test_grpo_train_step_enables_pre_update_alignment_gate_by_default():
    parameter = inspect.signature(grpo_mlx.train_step).parameters["max_abs_log_ratio_tolerance"]
    assert parameter.default == grpo_mlx.DEFAULT_MAX_ABS_LOG_RATIO_TOLERANCE


def test_recorded_behavior_logprobs_must_match_recomputation():
    assert grpo_mlx.validate_behavior_logprob_alignment(
        [-0.2, -1.0], [-0.2, -1.0005], tolerance=1e-3,
    ) == pytest.approx(5e-4)
    with pytest.raises(ValueError, match="disagree"):
        grpo_mlx.validate_behavior_logprob_alignment(
            [-0.2, -1.0], [-0.2, -1.01], tolerance=1e-3,
        )
    with pytest.raises(ValueError, match="equal non-zero lengths"):
        grpo_mlx.validate_behavior_logprob_alignment([-0.2], [-0.2, -1.0])
    with pytest.raises(ValueError, match="finite numbers"):
        grpo_mlx.validate_behavior_logprob_alignment([float("nan")], [0.0])


def test_compute_behavior_logprobs_checks_and_preserves_recorded_values(monkeypatch):
    mlx = ModuleType("mlx")
    mlx.__path__ = []
    mx = ModuleType("mlx.core")
    mx.float32 = object()
    mx.zeros = lambda *_args, **_kwargs: object()
    mx.eval = lambda *_args: None
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    class FakeLogprobs:
        def __getitem__(self, _index):
            return SimpleNamespace(tolist=lambda: [-0.2])

    def fake_value_and_grad(*_args, **_kwargs):
        return lambda *_values: ((None, (None, FakeLogprobs())), object())

    monkeypatch.setattr(grpo_mlx, "_value_and_grad", fake_value_and_grad)
    monkeypatch.setattr(
        grpo_mlx,
        "sample_tensors",
        lambda *_args, **_kwargs: (object(), SimpleNamespace(shape=(1, 1)), 0),
    )
    sample = grpo_mlx.Sample(
        "task", "policy", 0, [1], [2], 1.0, "grader-v1", 0, [-0.2005],
    )

    grpo_mlx.compute_behavior_logprobs(
        object(), [sample], grpo_mlx.GRPOConfig(), max_abs_log_ratio_tolerance=1e-3,
    )

    assert sample.behavior_logprobs == [-0.2005]
    sample.behavior_logprobs = [-0.21]
    with pytest.raises(ValueError, match="disagree"):
        grpo_mlx.compute_behavior_logprobs(
            object(), [sample], grpo_mlx.GRPOConfig(), max_abs_log_ratio_tolerance=1e-3,
        )
    assert sample.behavior_logprobs == [-0.21]


def test_train_step_blocks_misaligned_behavior_before_optimizer_update(monkeypatch):
    mlx = ModuleType("mlx")
    mlx.__path__ = []
    mx = ModuleType("mlx.core")
    mx.eval = lambda *_args: None
    optim = ModuleType("mlx.optimizers")
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx.optimizers", optim)

    class FakeOptimizer:
        def __init__(self):
            self.state = {"step": 0}

        def update(self, model, _grads):
            model.value += 1
            self.state["step"] += 1

    model = SimpleNamespace(value=0, parameters=lambda: {})
    optimizer = FakeOptimizer()
    samples = [
        grpo_mlx.Sample("task", "policy", 0, [1], [2], 1.0, "grader-v1", 0, [-0.5]),
        grpo_mlx.Sample("task", "policy", 0, [1], [3], 0.0, "grader-v1", 1, [-0.5]),
    ]
    monkeypatch.setattr(grpo_mlx, "trainable_digest", lambda current: str(current.value))
    monkeypatch.setattr(
        grpo_mlx,
        "policy_gradient",
        lambda *_args, **_kwargs: (
            object(),
            {"loss": 1.0, "clip_fraction": 0.0, "ratio_mean": 1.0, "ratio_max": 1.0,
             "approx_kl": 0.0, "max_abs_log_ratio": 1e-2},
        ),
    )

    record = grpo_mlx.train_step(
        model, optimizer, samples, grpo_mlx.GRPOConfig(group_size=2, max_grad_norm=None),
        policy_checkpoint_id="policy", policy_step=0,
    )

    assert record["update_applied"] is False
    assert record["reason"] == "behavior_logprob_ratio_out_of_tolerance"
    assert record["params_changed"] is False
    assert model.value == 0
    assert optimizer.state["step"] == 0
