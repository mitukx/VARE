from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "sprint_xiii" / "online_actor.py"
spec = importlib.util.spec_from_file_location("sprint_xiii_online_actor", SCRIPT)
online = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(online)

AUDIT_SCRIPT = Path(__file__).parents[1] / "scripts" / "sprint_xiii" / "audit_online_actor.py"
audit_spec = importlib.util.spec_from_file_location(
    "sprint_xiii_audit_online_actor", AUDIT_SCRIPT
)
audit = importlib.util.module_from_spec(audit_spec)
assert audit_spec and audit_spec.loader
audit_spec.loader.exec_module(audit)

FREEZER_SCRIPT = Path(__file__).parents[1] / "scripts" / "sprint_xiii" / "freeze_online_actor_protocol.py"
freezer_spec = importlib.util.spec_from_file_location(
    "sprint_xiii_freeze_online_actor_protocol", FREEZER_SCRIPT
)
freezer = importlib.util.module_from_spec(freezer_spec)
assert freezer_spec and freezer_spec.loader
freezer_spec.loader.exec_module(freezer)


TASK = {"target_pair": "AB", "forbidden_pair": "CD"}


def test_proxy_verifiers_are_distinct_and_trusted_rule_checks_full_route() -> None:
    assert online.verifier_score("ABCD", TASK, 0) == 0.5
    assert online.verifier_score("ABCD", TASK, 1) == 0.875
    assert audit.trusted("ABCD", TASK) == 0.0
    assert audit.trusted("XAB", TASK) == 1.0
    assert not hasattr(online, "trusted_score")
    assert online.verifier_score("DDDDAB", TASK, 0) == 0.0
    assert online.verifier_score("DDDDAB", TASK, 1) == 0.875
    assert online.verifier_score("ACDAB", TASK, 1) == 0.0


def test_calibration_enumerates_the_frozen_variable_length_action_space() -> None:
    p = {
        "run_id": "synthetic-calibration-test",
        "action_labels": ["A", "B", "C", "D"],
        "generation": {"maximum_actions": 4},
        "tasks": [
            {
                "task_id": "t",
                "split": "train",
                "target_pair": "AB",
                "forbidden_pair": "CD",
            },
            {
                "task_id": "e",
                "split": "evaluation",
                "target_pair": "AC",
                "forbidden_pair": "BD",
            },
        ],
    }
    p["task_manifest"] = {"tasks": p["tasks"]}
    p["training_tasks"] = ["t"]
    p["evaluation"] = {"task_ids": ["e"]}
    result = audit.calibrate(p)
    assert result["v0"]["n"] == 2 * (4**2 + 4**3 + 4**4)
    assert result["v1"]["n"] == result["v0"]["n"]
    assert result["v0_v1_disagreement"] > 0
    assert result["v0"]["false_positive"] > 0
    assert result["v1"]["false_positive"] > 0


def test_frozen_protocol_requires_runner_tasks_to_match_manifest() -> None:
    tasks = [
        {"task_id": "train-1", "split": "train"},
        {"task_id": "eval-1", "split": "evaluation"},
    ]
    valid = {
        "run_id": "run-test",
        "tasks": tasks,
        "task_manifest": {"tasks": tasks},
        "training_tasks": ["train-1"],
        "evaluation": {"task_ids": ["eval-1"]},
    }
    online.validate_protocol_contract(valid)
    audit.validate_protocol_contract(valid)
    freezer.validate_protocol_task_contract(
        valid
    )
    for validator in (
        online.validate_protocol_contract,
        audit.validate_protocol_contract,
        freezer.validate_protocol_task_contract,
    ):
        try:
            validator(
                {
                    "tasks": tasks,
                    "task_manifest": {"tasks": tasks},
                    "training_tasks": ["train-1"],
                    "evaluation": {"task_ids": ["eval-1"]},
                }
            )
        except (RuntimeError, ValueError) as error:
            assert "run_id" in str(error)
        else:
            raise AssertionError("protocol missing run_id must be rejected")
        invalid_split = dict(valid)
        invalid_split["training_tasks"] = ["not-in-train-split"]
        try:
            validator(invalid_split)
        except (RuntimeError, ValueError) as error:
            assert "training_tasks" in str(error)
        else:
            raise AssertionError("protocol task split mismatch must be rejected")
    try:
        freezer.validate_protocol_task_contract(
            {
                "run_id": "run-test",
                "task_manifest": {"tasks": tasks},
            }
        )
    except ValueError as error:
        assert str(error) == "protocol tasks must exactly match task_manifest.tasks"
    else:
        raise AssertionError("protocol missing runner tasks must be rejected")


class _Selection(list):
    def tolist(self):
        return list(self)


class _Vector(list):
    def __getitem__(self, index):
        if isinstance(index, list):
            return _Selection([super(_Vector, self).__getitem__(item) for item in index])
        return super().__getitem__(index)


def test_audit_v2_rejects_the_three_historical_mutant_classes() -> None:
    commit = {
        "transaction_id": "tx-2",
        "checkpoint_sha256": "a" * 64,
        "policy_version": 2,
    }
    assert audit.active_matches_latest_commit(
        {
            "transaction_id": "tx-2",
            "checkpoint_sha256": "a" * 64,
            "policy_version": 2,
        },
        commit,
    )
    assert not audit.active_matches_latest_commit(
        {
            "transaction_id": "tx-0",
            "checkpoint_sha256": "b" * 64,
            "policy_version": 0,
        },
        commit,
    )

    state_bias = _Vector([0.0, 0.25, -0.5, 0.75])
    attempt = {"bias_before_support": [0.0, 0.25, 0.75]}
    assert audit.gradient_origin_matches(state_bias, attempt, [0, 1, 3], 1e-7)
    attempt["bias_before_support"] = [0.125, 0.375, 0.875]
    assert not audit.gradient_origin_matches(state_bias, attempt, [0, 1, 3], 1e-7)

    outcomes = {
        "task-a": {"mean_trusted_reward": 0.25},
        "task-b": {"mean_trusted_reward": 0.75},
    }
    assert audit.evaluation_aggregation_matches(outcomes, 0.5)
    assert not audit.evaluation_aggregation_matches(outcomes, 0.75)


def test_journal_detects_mutation_and_keeps_an_append_only_chain(
    tmp_path: Path,
) -> None:
    online.append_journal(tmp_path, {"kind": "first", "value": 1})
    online.append_journal(tmp_path, {"kind": "second", "value": 2})
    rows = [
        __import__("json").loads(line)
        for line in (tmp_path / "journal.jsonl").read_text().splitlines()
    ]
    assert rows[1]["previous"] == rows[0]["sha256"]
    rows[0]["value"] = 9
    assert rows[0]["sha256"] != online.sha(
        online.canonical(
            {key: value for key, value in rows[0].items() if key != "sha256"}
        )
    )


def test_resource_token_accounting_covers_group_and_evaluation_outputs(
    tmp_path: Path,
) -> None:
    import json

    group_dir = tmp_path / "cases" / "c" / "groups"
    group_dir.mkdir(parents=True)
    (group_dir / "g.json").write_text(
        json.dumps(
            {
                "prompt_ids": [1, 2, 3],
                "generations": [{"generated_token_count": 2}, {"generated_token_count": 3}],
            }
        )
    )
    baseline = tmp_path / "baseline" / "seed-1"
    baseline.mkdir(parents=True)
    (baseline / "evaluation.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "prompt_ids": [1, 2],
                        "generations": [{"generated_token_count": 4}],
                    }
                ]
            }
        )
    )
    assert online.completed_generation_tokens(tmp_path) == 9
    assert online.completed_prompt_tokens(tmp_path) == 8
