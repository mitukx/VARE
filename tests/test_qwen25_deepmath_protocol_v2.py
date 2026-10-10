import json
import time

import pytest

from scripts.gpu_gate_journal_v2 import GateJournal, completion_token_count, run_prompt_groups
from scripts.qwen25_deepmath_inputs_v2 import (
    GRPO_GENERATION_DEFAULTS,
    base_gate_generation_config,
    prompt_messages,
    sft_example,
    tokenize_generation_prompt,
)

PROTOCOL = json.loads(open("protocols/qwen25_deepmath_grpo_math500_v2.lock.json", encoding="utf-8").read())


def test_grpo_gate_sft_and_evaluation_share_user_only_messages():
    problem = "Find x: x + 1 = 2"
    assert prompt_messages(problem) == [{"role": "user", "content": problem}]
    assert sft_example(problem, "1") == {
        "prompt": [{"role": "user", "content": problem}],
        "completion": [{"role": "assistant", "content": r"\boxed{1}"}],
    }

    class Tokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            self.seen = conversation, kwargs
            return {"input_ids": [101, 102]}

    tokenizer = Tokenizer()
    assert tokenize_generation_prompt(tokenizer, problem) == {"input_ids": [101, 102]}
    conversation, kwargs = tokenizer.seen
    assert conversation == prompt_messages(problem)
    assert kwargs == {"tokenize": True, "add_generation_prompt": True, "return_dict": True}


def test_frozen_conditions_align_gate_grpo_sft_and_evaluation():
    contract = PROTOCOL["prompt_and_generation_contract"]
    grpo = PROTOCOL["conditional_followup_if_gate_passes"]["planned_comparison"]["grpo"]
    sft = PROTOCOL["conditional_followup_if_gate_passes"]["planned_comparison"]["sft"]
    evaluation = PROTOCOL["conditional_followup_if_gate_passes"]["planned_comparison"]["evaluation"]
    assert contract["system_prompt"] is None
    assert contract["base_gate_sampling"]["temperature"] == grpo["temperature"]
    assert contract["base_gate_sampling"]["top_p"] == grpo["top_p"]
    assert contract["base_gate_sampling"]["max_new_tokens"] == grpo["max_completion_length"]
    assert contract["base_gate_sampling"]["top_k"] == grpo["top_k"] == 0
    assert contract["base_gate_sampling"]["min_p"] is grpo["min_p"] is None
    assert contract["base_gate_sampling"]["repetition_penalty"] == grpo["repetition_penalty"] == 1.0
    assert {key: contract["base_gate_sampling"][key] for key in GRPO_GENERATION_DEFAULTS} == GRPO_GENERATION_DEFAULTS
    assert {key: grpo[key] for key in GRPO_GENERATION_DEFAULTS} == GRPO_GENERATION_DEFAULTS
    assert base_gate_generation_config(max_new_tokens=37, pad_token_id=4, bos_token_id=1, eos_token_id=2) == {
        **GRPO_GENERATION_DEFAULTS,
        "max_new_tokens": 37,
        "pad_token_id": 4,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    assert grpo["max_prompt_length"] == contract["max_prompt_tokens"] == 512
    assert sft["system_prompt"] is None and sft["completion_only_loss"] is True
    assert evaluation["prompt_messages"] == "same one-user-message format; no system prompt"
    assert evaluation["do_sample"] is False
    assert PROTOCOL["protocol_revision"]["v1_immutable"] is True


def test_generated_token_count_uses_token_ids_and_stops_after_eos():
    assert completion_token_count([11, 12, 99, 99, 99], 99) == 3
    assert completion_token_count([11, 12, 13], 99) == 3
    assert completion_token_count([], 99) == 0


def _group(_index, _row, cap):
    ids = [[1], [2, 3], [4], [5, 6]]
    assert all(len(tokens) <= cap for tokens in ids)
    return {"completion_token_ids": ids, "metric": "cpu-fixture"}


def test_checkpoint_resume_keeps_completed_prompt_and_token_budget(tmp_path):
    output = tmp_path / "gate.json"
    identity = {"protocol": "test", "weights": "hash-a"}
    journal = GateJournal(output, identity)

    class OutOfMemoryError(Exception):
        pass

    calls = []

    def fail_second(index, row, cap):
        calls.append(index)
        if index == 1:
            raise OutOfMemoryError("simulated CPU exception with CUDA OOM name")
        return _group(index, row, cap)

    with pytest.raises(OutOfMemoryError):
        run_prompt_groups(rows=[{"i": 0}, {"i": 1}], journal=journal, generate=fail_second,
                          completions_per_prompt=4, max_new_tokens_per_completion=8,
                          max_generated_tokens=30, max_wall_seconds=60, eos_token_id=99)
    state = json.loads(journal.state_path.read_text())
    assert state["decision"] == "oom"
    assert state["completed_prompt_groups"] == 1
    assert len(journal.partial_path.read_text().splitlines()) == 1

    resumed = GateJournal(output, identity, resume=True)
    assert run_prompt_groups(rows=[{"i": 0}, {"i": 1}], journal=resumed, generate=_group,
                             completions_per_prompt=4, max_new_tokens_per_completion=8,
                             max_generated_tokens=30, max_wall_seconds=60, eos_token_id=99) == "complete"
    assert sorted(resumed.records) == [0, 1]
    assert resumed.records[0]["generated_tokens"] == 6
    assert resumed.records[1]["generated_tokens"] == 6

    result = {"decision": "pass", "marker": "fixture"}
    resumed.finalize(result)
    final = json.loads(output.read_text())
    assert len(final["prompt_groups"]) == 2
    assert final["prompt_groups"][0]["generated_tokens"] == 6
    assert json.loads(resumed.state_path.read_text())["decision"] == "complete"


def test_resume_rejects_changed_protocol_or_weights(tmp_path):
    output = tmp_path / "gate.json"
    journal = GateJournal(output, {"weights": "hash-a"})
    journal.stop("oom", "fixture")
    with pytest.raises(ValueError, match="identity differs"):
        GateJournal(output, {"weights": "hash-b"}, resume=True)


def test_wall_timeout_persists_state_and_can_resume(tmp_path):
    output = tmp_path / "gate.json"
    journal = GateJournal(output, {"protocol": "timeout"})

    def slow(_index, _row, _cap):
        time.sleep(2)
        return _group(0, {}, _cap)

    assert run_prompt_groups(rows=[{}], journal=journal, generate=slow, completions_per_prompt=4,
                             max_new_tokens_per_completion=8, max_generated_tokens=30,
                             max_wall_seconds=1, eos_token_id=99) == "wall_time"
    state = json.loads(journal.state_path.read_text())
    assert state["decision"] == "wall_time"
    assert state["completed_prompt_groups"] == 0
    resumed = GateJournal(output, {"protocol": "timeout"}, resume=True)
    assert run_prompt_groups(rows=[{}], journal=resumed, generate=_group, completions_per_prompt=4,
                             max_new_tokens_per_completion=8, max_generated_tokens=30,
                             max_wall_seconds=30, eos_token_id=99) == "complete"
