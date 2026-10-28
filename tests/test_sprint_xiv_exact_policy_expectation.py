from scripts.sprint_xiv.exact_policy_expectation import (
    compare_distributions,
    summarize_distribution,
    terminal_distribution,
)


def test_uniform_policy_enumerates_all_terminal_sequences_and_mass():
    labels = ["A", "B", "C", "D"]
    nodes = {
        "".join(prefix): [0.0] * 5
        for depth in range(5)
        for prefix in __import__("itertools").product(labels, repeat=depth)
    }
    outcomes = terminal_distribution(
        nodes,
        labels,
        [0.0] * 5,
        eos_offset=0.0,
        temperature=1.0,
        minimum_actions=2,
        maximum_actions=5,
        task={"target_pair": "AB", "forbidden_pair": "CD"},
    )
    assert len(outcomes) == 1360
    assert abs(sum(outcomes.values()) - 1.0) < 1e-12
    assert sum(p for key, p in outcomes.items() if key.endswith("|maximum_actions")) > 0


def test_reward_moments_and_distribution_divergence():
    task = {"target_pair": "AB", "forbidden_pair": "CD"}
    outcomes = {"AB|eos": 0.5, "CD|eos": 0.5}
    summary = summarize_distribution(outcomes, task)
    assert summary["expected_trusted_reward_v1"] == 0.4375
    assert summary["probability_trusted_reward_equals_1"] == 0.0
    assert summary["probability_exact_task_success"] == 0.5
    assert compare_distributions(outcomes, outcomes) == {
        "trajectory_total_variation": 0.0,
        "trajectory_kl_reference_to_candidate_nats": 0.0,
        "trajectory_kl_candidate_to_reference_nats": 0.0,
    }


def test_trusted_full_reward_requires_target_pair_and_partial_component():
    task = {"target_pair": "AB", "forbidden_pair": "CD"}
    summary = summarize_distribution({"ABAB|eos": 1.0}, task)
    assert summary["expected_trusted_reward_v1"] == 1.0
    assert summary["probability_trusted_reward_equals_1"] == 1.0
    assert summary["probability_exact_task_success"] == 1.0


def test_nonmatching_distribution_tv_is_exact():
    metrics = compare_distributions({"a": 0.75, "b": 0.25}, {"a": 0.25, "b": 0.75})
    assert metrics["trajectory_total_variation"] == 0.5
    assert metrics["trajectory_kl_candidate_to_reference_nats"] > 0
