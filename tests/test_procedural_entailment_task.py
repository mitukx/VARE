import json

from scripts.procedural_entailment_task import generate_split, verify_split
from scripts.run_cpu_procedural_entailment_dpo_development_v1 import load_protocol


def test_procedural_splits_are_deterministic_balanced_and_oracle_checked():
    kwargs = {"count": 64, "positive_count": 32, "n_nodes": 12, "edge_probability": 0.1}
    first = generate_split(seed=21262013, style=1, **kwargs)
    second = generate_split(seed=21262013, style=1, **kwargs)
    assert first == second
    verify_split(first, expected_count=64, expected_positive_count=32)
    assert all(len(row["graph"]["reachable_nodes"]) < 12 for row in first)
    assert all(all(len(edge) == 2 for edge in row["graph"]["edges"]) for row in first)


def test_generated_splits_use_distinct_prompt_hashes_and_surface_forms():
    train = generate_split(seed=20261011, count=64, style=0, positive_count=32,
                           n_nodes=10, edge_probability=0.14)
    development = generate_split(seed=20261012, count=64, style=1, positive_count=32,
                                 n_nodes=10, edge_probability=0.14)
    assert not ({row["prompt_sha256"] for row in train} &
                {row["prompt_sha256"] for row in development})
    assert train[0]["prompt"] != development[0]["prompt"]
    assert all(train[0]["prompt"] != row["prompt"] for row in train[1:])


def test_locked_protocol_excludes_pilots_and_reserves_confirmation():
    spec, digest = load_protocol()
    assert digest
    assert spec["data"]["pilot_exclusion"].find("20261009") >= 0
    assert spec["data"]["pilot_exclusion"].find("20261010") >= 0
    assert spec["data"]["confirmation_reservation"]["balanced"]["count"] == 512
    assert spec["data"]["confirmation_reservation"]["prior_shift"]["count"] == 512


def test_two_action_dpo_loss_pushes_correct_action_delta_upward():
    import pytest
    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional

    delta = torch.tensor(0.0, requires_grad=True)
    beta = 1.0
    # Positive means Yes is the verifier-correct action.
    loss = functional.softplus(-beta * 1.0 * delta)
    loss.backward()
    assert delta.grad.item() < 0
    # Gradient descent therefore increases the correct Yes-minus-No log-ratio.
    delta_no = torch.tensor(0.0, requires_grad=True)
    loss_no = functional.softplus(-beta * -1.0 * delta_no)
    loss_no.backward()
    assert delta_no.grad.item() > 0


def test_scalar_and_contextual_training_take_one_update():
    import pytest
    torch = pytest.importorskip("torch")
    from scripts.run_cpu_procedural_entailment_dpo_development_v1 import train_arm

    train_rows = [{"label": "Yes" if i % 2 == 0 else "No"} for i in range(16)]
    dev_rows = [{"label": "Yes" if i % 2 == 0 else "No"} for i in range(8)]
    train_features = torch.arange(48, dtype=torch.float32).reshape(16, 3) / 10
    dev_features = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10
    train_margins = torch.zeros(16)
    dev_margins = torch.zeros(8)
    standardizer = {"mean": train_features.mean(0),
                    "scale": train_features.std(0, unbiased=False).clamp_min(1e-5)}
    spec = {"learner": {"learning_rate": 0.01, "weight_decay": 0.0,
                         "minibatch_size": 16, "maximum_updates": 1,
                         "checkpoints_updates": [0, 1], "gradient_clip": 1.0,
                         "beta": 1.0}}
    for arm in ("calibration_only", "context_sft", "context_dpo"):
        result = train_arm(arm, train_features, train_margins, train_rows,
                           dev_features, dev_margins, dev_rows, standardizer, 7301, spec)
        assert set(result["checkpoints"]) == {"0", "1"}
        assert result["checkpoints"]["1"]["updates"] == 1
