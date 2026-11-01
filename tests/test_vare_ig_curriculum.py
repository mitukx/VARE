from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vare.learning_pipeline import informative_curriculum as curriculum
from vare.learning_pipeline import vare_ig_tasks as tasks
from scripts.qwen3_vare_ig_experiment import (
    _case_wall_time_summary,
    _concatenate_staged_rollouts,
)
from scripts.resume_vare_ig_preupdate_batch import (
    _assert_saved_response_matches_rollout,
    _clear_mlx_cache_then_guard,
)
from scripts.run_vare_ig_durable_training import make_training_sample_record


def test_vare_ig_task_pack_is_new_fixed_size_and_split_clean():
    tasks.validate_task_pack()
    assert len(tasks.TRAIN_TASKS) == 12
    assert len(tasks.EXPLORATION_TASKS) == 10
    assert len(tasks.DEVELOPMENT_TASKS) == 4
    assert len({task.task_id for task in tasks.all_tasks()}) == 26
    assert all(len(task.cases) == 5 for task in tasks.all_tasks())
    assert set(tasks.FAMILY_BY_TASK.values()) == {"parsing", "graph", "stream", "configuration"}
    assert set(tasks.FAMILY_BY_EXPLORATION_TASK.values()) == set(tasks.FAMILY_BY_TASK.values())
    assert not set(tasks.FAMILY_BY_TASK).intersection(tasks.FAMILY_BY_EXPLORATION_TASK)


def test_task_hashes_cover_contract_and_parent_owned_expected_values():
    task = tasks.TRAIN_TASKS[0]
    before = task.sha256()
    changed = type(task)(task.task_id, task.split, task.prompt, task.cases[:-1])
    assert changed.sha256() != before
    assert all(case.expected is not None or case.case_id for case in task.cases)


def test_preupdate_recovery_binds_response_text_to_graded_rollout_digest():
    response = "def solve(value):\n    return value\n"
    digest = hashlib.sha256(response.encode("utf-8")).hexdigest()
    sample = {"response_text": response}
    rollout = {"text": response, "response_sha256": digest}

    _assert_saved_response_matches_rollout(sample, rollout)

    with pytest.raises(RuntimeError, match="serialized sample response differs"):
        _assert_saved_response_matches_rollout({"response_text": response + "# altered\n"}, rollout)

    with pytest.raises(RuntimeError, match="rollout response digest"):
        _assert_saved_response_matches_rollout(sample, {"text": response, "response_sha256": "bad"})


def test_preupdate_recovery_clears_evaluated_cache_before_resource_guard():
    events = []

    class FakeMx:
        def clear_cache(self):
            events.append("clear")

    _clear_mlx_cache_then_guard(FakeMx(), lambda: events.append("guard"))
    assert events == ["clear", "guard"]


def test_durable_training_launcher_persists_exact_token_ids_and_reward_provenance():
    response = "def solve(value): return value"
    digest = hashlib.sha256(response.encode("utf-8")).hexdigest()
    generated = {
        "text": response,
        "prompt_ids": [11, 12],
        "completion_ids": [13, 14, 15],
        "sampler_logprobs": [-0.1, -0.2, -0.3],
    }
    rollout = {
        "task_id": "task-1", "task_sha256": "task-sha", "phase": "train-step-1",
        "policy_checkpoint_id": "policy-1", "behavior_step": 1,
        "sample_index": 2, "reward": 1.0, "response_sha256": digest,
        "prompt_tokens": 2, "completion_tokens": 3, "candidate_source_sha256": "source-sha",
        "finish": "stop", "grade_status": "all_cases_passed", "trusted_contract": True,
        "grader": {"verifier_id": "oracle-v1", "case_results": [{"passed": True}],
                   "case_set_sha256": "cases-sha"},
    }

    row = make_training_sample_record(generated, rollout)
    assert row["prompt_ids"] == generated["prompt_ids"]
    assert row["completion_ids"] == generated["completion_ids"]
    assert row["response_sha256"] == digest
    assert row["candidate_source_sha256"] == "source-sha"
    assert row["verifier"] == "oracle-v1"
    assert row["behavior_logprobs"] is None


def test_independent_bernoulli_informative_probability_is_only_diagnostic_formula():
    formula = curriculum.independent_bernoulli_informative_probability
    assert formula(0.0, 4) == 0.0
    assert formula(1.0, 4) == 0.0
    assert formula(0.5, 4) == pytest.approx(0.875)
    assert formula(0.25, 4) == pytest.approx(1 - 0.25**4 - 0.75**4)
    with pytest.raises(ValueError):
        formula(1.1, 4)


def test_group_rate_tracks_trusted_mixed_reward_groups_and_pass_rates():
    rate = curriculum.GroupRate()
    assert not rate.observe([1, 1, 1, 1])
    assert rate.observe([1, 0, 1, 0])
    assert rate.observed_groups == 2
    assert rate.informative_groups == 1
    assert rate.expected_informative_group_probability == pytest.approx(0.5)
    assert rate.observed_pass_probability == pytest.approx(0.75)
    for invalid in ([0, 0.5], [True, False], []):
        with pytest.raises(ValueError):
            rate.observe(invalid)


def test_uniform_selection_is_without_replacement_and_probabilities_sum_to_one():
    ids = {task.task_id: family for task, family in
           ((task, tasks.FAMILY_BY_TASK[task.task_id]) for task in tasks.TRAIN_TASKS)}
    rates = {family: curriculum.GroupRate() for family in set(ids.values())}
    costs = {family: 2.0 for family in rates}
    scheduler = curriculum.InformativeGroupScheduler(
        task_families=ids, family_rates=rates, family_cost_seconds=costs, seed=7, mode="uniform"
    )
    selected = []
    for _ in range(6):
        selected.append(scheduler.next_task())
        record = scheduler.history[-1]
        assert sum(record["conditional_probabilities"].values()) == pytest.approx(1.0)
        assert record["selection_probability"] == pytest.approx(1.0 / (len(ids) - len(selected) + 1))
        scheduler.observe(selected[-1], [1, 0, 1, 0])
    assert len(set(selected)) == len(selected)


def test_adaptive_selection_uses_observed_informativeness_per_measured_cost():
    family_map = {"easy-a": "easy", "learnable-a": "learnable", "learnable-b": "learnable"}
    rates = {
        "easy": curriculum.GroupRate(informative_groups=0, observed_groups=4),
        "learnable": curriculum.GroupRate(informative_groups=4, observed_groups=4),
    }
    costs = {"easy": 1.0, "learnable": 2.0}
    scheduler = curriculum.InformativeGroupScheduler(
        task_families=family_map, family_rates=rates, family_cost_seconds=costs,
        seed=1, mode="informative_per_cost"
    )
    scheduler.next_task()
    record = scheduler.history[0]
    assert record["conditional_probabilities"]["learnable-a"] == pytest.approx(5 / 12)
    assert record["conditional_probabilities"]["easy-a"] == pytest.approx(1 / 6)
    scheduler.observe(record["selected_task_id"], [1, 1, 1, 1])
    assert scheduler.history[0]["observed_informative_group"] is False
    assert "family_alpha_after_observation" in scheduler.history[0]


def test_adaptive_probabilities_update_before_the_next_training_group():
    family_map = {"a1": "a", "a2": "a", "b1": "b", "b2": "b"}
    rates = {"a": curriculum.GroupRate(), "b": curriculum.GroupRate()}
    costs = {"a": 1.0, "b": 1.0}

    def after_first_group(rewards):
        scheduler = curriculum.InformativeGroupScheduler(
            task_families=family_map, family_rates=rates, family_cost_seconds=costs,
            seed=29, mode="informative_per_cost",
        )
        selected = scheduler.next_task()
        family = family_map[selected]
        scheduler.observe(selected, rewards)
        scheduler.next_task()
        probabilities = scheduler.history[1]["conditional_probabilities"]
        return sum(probability for task_id, probability in probabilities.items()
                   if family_map[task_id] == family)

    assert after_first_group([1, 0, 1, 0]) == pytest.approx(0.4)
    assert after_first_group([1, 1, 1, 1]) == pytest.approx(0.25)


def test_family_rates_reject_observations_outside_the_locked_split():
    with pytest.raises(ValueError):
        curriculum.family_group_rates([("heldout-secret", [1, 0, 1, 0])], {"train": "graph"})


def test_staged_rollouts_concatenate_as_bytes_without_text_binary_type_error(tmp_path: Path):
    first = tmp_path / "group-00.jsonl"
    second = tmp_path / "group-01.jsonl"
    output = tmp_path / "rollouts.jsonl"
    first_bytes = b'{"group":0,"text":"alpha"}\n'
    second_bytes = b'{"group":1,"text":"beta"}\n'
    first.write_bytes(first_bytes)
    second.write_bytes(second_bytes)

    _concatenate_staged_rollouts([first, second], output)

    assert output.read_bytes() == first_bytes + second_bytes
    with pytest.raises(FileExistsError):
        _concatenate_staged_rollouts([first], output)


def test_case_wall_time_summary_counts_missing_values_without_treating_them_as_zero():
    summary = _case_wall_time_summary([
        {"case_results": [
            {"wall_s": 0.25}, {"wall_s": 1}, {"wall_s": None},
            {"wall_s": True}, {"wall_s": -1}, {"wall_s": "0.5"},
            {"wall_s": float("nan")}, {"wall_s": 10**1000},
        ]},
        {"case_results": None},
    ])

    assert summary == {
        "measured_seconds": 1.25,
        "measured_case_count": 2,
        "unmeasured_case_count": 7,
    }
