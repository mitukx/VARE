from __future__ import annotations

import itertools
import math

import pytest

from scripts.boolq_rloo_task import (
    balanced_accuracy,
    binary_nll,
    hash_rank_indices,
    validate_disjoint_hashes,
)


class PromptOnlyView:
    def __init__(self, rows, columns):
        assert set(columns) == {"question", "passage"}
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {"question": row["question"], "passage": row["passage"]}


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.requested_full_rows = []

    def __len__(self):
        return len(self.rows)

    def select_columns(self, columns):
        return PromptOnlyView(self.rows, columns)

    def __getitem__(self, index):
        self.requested_full_rows.append(index)
        return self.rows[index]


def test_hash_ranking_requests_prompt_columns_without_answer_labels():
    rows = [
        {"question": "q2", "passage": "p", "answer": True},
        {"question": "q1", "passage": "p", "answer": False},
        {"question": "q3", "passage": "p", "answer": True},
    ]
    dataset = FakeDataset(rows)
    indices = hash_rank_indices(dataset, 0, 3)
    assert len(indices) == 3
    assert dataset.requested_full_rows == []


def test_hash_ranking_rejects_invalid_intervals():
    dataset = FakeDataset([{"question": "q", "passage": "p", "answer": True}])
    with pytest.raises(ValueError):
        hash_rank_indices(dataset, 1, 1)
    with pytest.raises(ValueError):
        hash_rank_indices(dataset, 0, 2)


def test_cohort_disjointness_fails_closed_on_reused_prompt():
    a = [{"question_sha256": "same"}]
    with pytest.raises(ValueError, match="overlaps"):
        validate_disjoint_hashes({"train": a, "development": a})


def test_balanced_accuracy_and_nll():
    assert balanced_accuracy([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.5)
    assert balanced_accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert binary_nll([0.8, 0.2], [1, 0]) == pytest.approx(-math.log(0.8))
    with pytest.raises(ValueError):
        balanced_accuracy([1], [1])


def test_stratified_paired_bootstrap_is_deterministic_and_paired():
    from scripts.boolq_rloo_task import stratified_paired_bootstrap

    a = [[1, 1], [0, 1], [1, 0], [0, 0]]
    b = [[0, 0], [0, 0], [1, 1], [0, 0]]
    labels = [1, 1, 0, 0]
    left = stratified_paired_bootstrap(a, b, labels, resamples=500, seed=17)
    right = stratified_paired_bootstrap(a, b, labels, resamples=500, seed=17)
    assert left == right
    assert left["difference"] == pytest.approx(0.25)


def test_auditor_bootstrap_matches_runner_with_seed_averaged_per_prompt_rows():
    pytest.importorskip("numpy")
    from scripts.audit_cpu_lm_boolq_verifier_rloo_development_v1 import paired_bootstrap
    from scripts.boolq_rloo_task import stratified_paired_bootstrap

    labels = [0, 1, 0, 1, 0, 1]
    a_by_seed = [
        [1, 1, 0, 1, 1, 0],
        [1, 0, 1, 1, 0, 1],
        [0, 1, 1, 0, 1, 1],
    ]
    b_by_seed = [
        [0, 1, 0, 1, 1, 0],
        [1, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 1],
    ]
    a = [list(values) for values in zip(*a_by_seed)]
    b = [list(values) for values in zip(*b_by_seed)]
    expected = stratified_paired_bootstrap(a, b, labels, resamples=500, seed=29)
    actual = paired_bootstrap(a, b, labels, resamples=500, seed=29)
    assert all(math.isclose(actual[key], expected[key], rel_tol=0.0, abs_tol=1e-12) for key in expected)


def test_exact_expected_reward_gradient_matches_exact_enumerated_rloo_expectation():
    torch = pytest.importorskip("torch")
    from scripts.rloo_binary_objectives import exact_expected_reward_loss, rloo_policy_loss

    logit_value = 0.37
    gold = 1.0
    weight = 1.4
    k = 4
    logits = torch.tensor([logit_value], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([gold], dtype=torch.float64)
    weights = torch.tensor([weight], dtype=torch.float64)
    base = torch.tensor([logit_value], dtype=torch.float64)
    exact, _ = exact_expected_reward_loss(logits, labels, weights, base, 0.0)
    exact_grad = torch.autograd.grad(exact, logits)[0].item()

    p = torch.sigmoid(torch.tensor(logit_value, dtype=torch.float64)).item()
    expected_gradient = 0.0
    for group in itertools.product((0, 1), repeat=k):
        actions = torch.tensor([group], dtype=torch.float64)
        probability = math.prod(p if action == 1 else 1 - p for action in group)
        sample_logits = torch.tensor([logit_value], dtype=torch.float64, requires_grad=True)
        loss, _ = rloo_policy_loss(sample_logits, actions, labels, weights, base, 0.0)
        expected_gradient += probability * torch.autograd.grad(loss, sample_logits)[0].item()
    assert expected_gradient == pytest.approx(exact_grad, abs=1e-10)


def test_rloo_equal_reward_group_has_zero_policy_gradient():
    torch = pytest.importorskip("torch")
    from scripts.rloo_binary_objectives import rloo_policy_loss

    logits = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    actions = torch.tensor([[1, 1, 1, 1]], dtype=torch.float64)
    labels = torch.tensor([1.0], dtype=torch.float64)
    weights = torch.tensor([1.0], dtype=torch.float64)
    base = logits.detach().clone()
    loss, details = rloo_policy_loss(logits, actions, labels, weights, base, 0.0)
    grad = torch.autograd.grad(loss, logits)[0]
    assert grad.item() == pytest.approx(0.0, abs=1e-12)
    assert details["zero_advantage_group_rate"].item() == pytest.approx(1.0)


def test_rloo_advantages_are_detached_and_unstandardized():
    torch = pytest.importorskip("torch")
    from scripts.rloo_binary_objectives import rloo_policy_loss

    logits = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
    actions = torch.tensor([[1, 0, 1, 0]], dtype=torch.float64)
    labels = torch.tensor([1.0], dtype=torch.float64)
    weights = torch.tensor([2.0], dtype=torch.float64)
    base = torch.tensor([0.1], dtype=torch.float64)
    loss, details = rloo_policy_loss(logits, actions, labels, weights, base, 0.0)
    grad = torch.autograd.grad(loss, logits)[0]
    assert math.isfinite(grad.item())
    assert details["mixed_reward_group_rate"].item() == pytest.approx(1.0)
    assert details["zero_advantage_group_rate"].item() == pytest.approx(0.0)


def test_bernoulli_kl_is_zero_at_reference_and_positive_after_shift():
    torch = pytest.importorskip("torch")
    from scripts.rloo_binary_objectives import bernoulli_kl_from_logits

    reference = torch.tensor([0.3, -0.2], dtype=torch.float64)
    same = bernoulli_kl_from_logits(reference, reference)
    shifted = bernoulli_kl_from_logits(reference + 0.5, reference)
    assert torch.allclose(same, torch.zeros_like(same), atol=1e-12)
    assert torch.all(shifted > 0)


def test_runtime_expected_reward_gradient_self_check_covers_both_classes_and_weights():
    pytest.importorskip("torch")
    from scripts.rloo_binary_objectives import expected_reward_gradient_self_check

    result = expected_reward_gradient_self_check()
    assert result["pass"] is True
    assert result["enumerated_cases"] == 12
    assert result["max_abs_gradient_error"] <= result["tolerance"]


def test_independent_auditor_replays_all_four_optimizer_arms_on_tiny_fixture(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from scripts.audit_cpu_lm_boolq_verifier_rloo_development_v1 import replay_optimizers, verify_rollouts
    from scripts.run_cpu_lm_boolq_verifier_rloo_development_v1 import make_minibatches, train_arm
    from pathlib import Path
    import json
    import time

    spec = json.loads(
        Path("protocols/cpu_lm_boolq_verifier_rloo_development_v1.json").read_text(encoding="utf-8")
    )
    spec["learner"]["optimizer"].update({"maximum_updates": 2, "checkpoint_updates": [0, 1, 2], "seeds": [8101]})
    spec["learner"]["optimizer"]["minibatch_size"] = 4
    spec["compute_limits"]["threads"] = 1
    train_x = np.asarray([[0.0, 1.0], [1.0, -1.0], [-1.0, 0.5], [0.5, 0.2]], dtype=np.float32)
    train_margin = np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    train_y = np.asarray([1, 0, 1, 0], dtype=np.int64)
    weights = {0: 1.0, 1: 1.0}
    schedule = make_minibatches(4, 2, 4, 8101)
    rollouts = []
    history = {}
    for arm in ("scalar_calibration", "context_sft", "exact_expected_reward", "rloo_k4"):
        history[arm] = {"8101": train_arm(arm, 8101, train_x, train_margin, train_y, weights, schedule, spec, rollouts)}
    rollout_map = {(str(row["seed"]), int(row["update"])): row for row in rollouts}
    rollout_file = tmp_path / "rloo_rollouts.json"
    rollout_file.write_text(json.dumps(rollouts), encoding="utf-8")
    count, _ = verify_rollouts(tmp_path, spec, train_y, weights, {"8101": schedule}, {})
    assert count == 2
    replay_optimizers(
        spec, train_x, train_margin, train_y, weights, {"8101": schedule}, rollout_map,
        history, deadline=time.monotonic() + 30,
    )
    corrupted = json.loads(json.dumps(rollouts))
    corrupted[0]["rewards"][0][0] += 1.0
    rollout_file.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="verifier reward differs"):
        verify_rollouts(tmp_path, spec, train_y, weights, {"8101": schedule}, {})
