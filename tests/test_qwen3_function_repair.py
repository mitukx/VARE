"""Contract and prompt-boundary tests for the parent-scored Qwen3 pilot."""
from __future__ import annotations

import json
from pathlib import Path

from scripts import qwen3_function_repair_grpo as runner
from vare import function_repair_tasks as tasks


def test_task_ids_and_splits_are_unique_and_frozen():
    all_tasks = tasks.all_tasks()
    ids = [task.task_id for task in all_tasks]
    assert len(ids) == len(set(ids))
    assert len(tasks.TRAIN_TASKS) == 6
    assert len(tasks.DEVELOPMENT_TASKS) == 4
    assert all(task.split == "train" for task in tasks.TRAIN_TASKS)
    assert all(task.split == "development" for task in tasks.DEVELOPMENT_TASKS)
    assert all(task.cases and len({case.case_id for case in task.cases}) == len(task.cases) for task in all_tasks)


def test_parent_json_comparison_handles_null_and_keeps_boolean_types_distinct():
    matches = runner._json_value_matches
    assert matches(None, None)
    assert matches({"a": [None, True, 1]}, {"a": [None, True, 1.0]})
    assert not matches(None, [])
    assert not matches(False, None)
    assert not matches(True, 1)
    assert not matches({"a": None}, {"a": "None"})


def test_parent_oracle_scores_json_null_through_the_isolated_worker():
    source = """def solve(values, k):
    if k < 1 or k > len(values):
        return None
    best_sum = sum(values[:k])
    best_start = 0
    current = best_sum
    for end in range(k, len(values)):
        current += values[end] - values[end - k]
        if current > best_sum:
            best_sum, best_start = current, end - k + 1
    return [best_sum, best_start]
"""
    report = runner.score_source(tasks.TRAIN_TASKS[1], source)
    assert report["trusted_contract"] is True
    assert report["passed"] is True
    assert report["reward"] == 1.0
    assert len(report["cases"]) == len(tasks.TRAIN_TASKS[1].cases)


def test_model_prompt_contains_contract_but_not_parent_owned_cases():
    task = tasks.TRAIN_TASKS[0]
    messages = runner.task_prompt(task)
    assert messages[0]["role"] == "system"
    assert "do not include imports" in messages[0]["content"]
    assert task.prompt in messages[1]["content"]
    assert "case_id" not in messages[1]["content"]
    assert not any(f'"case_id": "{case.case_id}"' in messages[1]["content"] for case in task.cases)
    assert json.dumps(task.cases[1].expected, ensure_ascii=False) not in messages[1]["content"]
    assert json.dumps(task.cases[4].expected, ensure_ascii=False) not in messages[1]["content"]


def test_response_extraction_is_deterministic_and_fail_closed():
    source, status = runner.extract_source("```python\ndef solve(x):\n    return x\n```\n")
    assert status == "source_extracted" and source == "def solve(x):\n    return x\n"
    assert runner.extract_source("I cannot write code.") == (None, "solve_definition_missing")
    assert runner.extract_source("```python\ndef solve(x): return x\n```\n```python\ndef solve(x): return x\n```")[1] == (
        "code_block_missing_or_ambiguous"
    )


def test_rollout_seed_is_stable_and_task_specific():
    a = runner.seed_for(20261101, "train-step-0", "task-a", 0)
    assert a == runner.seed_for(20261101, "train-step-0", "task-a", 0)
    assert a != runner.seed_for(20261101, "train-step-0", "task-b", 0)
    assert a != runner.seed_for(20261101, "train-step-0", "task-a", 1)


def test_followup_protocol_locks_seed_sources_and_task_contracts(tmp_path: Path):
    train = {task.task_id: task.sha256() for task in tasks.TRAIN_TASKS}
    development = {task.task_id: task.sha256() for task in tasks.DEVELOPMENT_TASKS}
    protocol = tmp_path / "followup.json"
    payload = {
        "planned_run_id": "seed-followup",
        "preregistered_before_model_inference": True,
        "sampling": {"seed": 20261102},
        "source_hashes": runner.source_hashes(),
        "task_contracts": {
            "train": {"task_hashes": train},
            "development": {"task_hashes": development},
        },
    }
    protocol.write_text(json.dumps(payload), encoding="utf-8")
    assert runner.protocol_payload(protocol, seed=20261102)["planned_run_id"] == "seed-followup"
    assert runner.protocol_hash(protocol) == runner.sha256_file(protocol)
    try:
        runner.protocol_payload(protocol, seed=20261101)
    except RuntimeError as exc:
        assert "seed" in str(exc)
    else:
        raise AssertionError("a CLI seed outside the frozen protocol must fail closed")


def test_followup_protocol_rejects_changed_source_hashes(tmp_path: Path):
    protocol = tmp_path / "changed-source.json"
    protocol.write_text(json.dumps({
        "planned_run_id": "seed-followup",
        "preregistered_before_model_inference": True,
        "sampling": {"seed": 20261102},
        "source_hashes": {"scripts/qwen3_function_repair_grpo.py": "0" * 64},
        "task_contracts": {},
    }), encoding="utf-8")
    try:
        runner.protocol_payload(protocol, seed=20261102)
    except RuntimeError as exc:
        assert "source hashes" in str(exc)
    else:
        raise AssertionError("changed source hashes must fail closed")


def test_sampler_configuration_matches_exact_ratio_objective():
    runner.grpo_mlx.check_exact_ratio_sampling(temperature=runner.GRPO_CONFIG.temperature, top_p=1.0,
                                               top_k=0, min_p=0.0)
    assert runner.GRPO_CONFIG.group_size == runner.GROUP_SIZE == 4


def test_macos_available_memory_fallback_uses_free_inactive_and_speculative_pages():
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   100
Pages active:                                  40
Pages inactive:                               200
Pages speculative:                             5
"""
    assert runner._vm_available_memory_bytes(vm_stat) == 16384 * 305
    assert runner._vm_available_memory_bytes("unrecognized vm_stat output") is None


def test_adapter_export_uses_mlx_lm_inference_schema(tmp_path: Path):
    assert runner.adapter_config() == {
        "fine_tune_type": "lora",
        "num_layers": 8,
        "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0},
    }
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapters.safetensors").write_bytes(b"fixture adapter bytes")
    export = tmp_path / "export"
    digest = runner.export_adapter(checkpoint, export)
    assert (export / "adapters.safetensors").read_bytes() == b"fixture adapter bytes"
    assert json.loads((export / "adapter_config.json").read_text()) == runner.adapter_config()
    assert len(digest) == 64
