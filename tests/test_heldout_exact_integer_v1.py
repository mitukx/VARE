from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vare.heldout_exact_integer import (
    N_TASKS,
    SAMPLES_PER_TASK,
    canonical_task_pack,
    comparison_refusal,
    completed_samples,
    evaluate_expression,
    generate_tasks,
    grade_output,
    load_task_pack,
    parse_final_integer,
    pass_at_k,
    read_jsonl,
    run_gate,
    task_pack_sha256,
    verify_task_pack,
    wilson_interval,
)

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmarks/heldout/exact_integer_v1/tasks.jsonl"
PROTOCOL = ROOT / "benchmarks/heldout/exact_integer_v1/protocol.json"


def test_generator_is_deterministic_and_nonleaking():
    first = generate_tasks()
    second = generate_tasks()
    assert [task.to_dict() for task in first] == [task.to_dict() for task in second]
    assert len(first) == N_TASKS >= 50
    assert len({task.task_id for task in first}) == N_TASKS
    rows = [task.to_dict() for task in first]
    verify_task_pack(rows)
    families = {task.family for task in first}
    assert families == {"add_sub", "mul_add", "div_exact", "paren"}
    assert all(sum(task.family == family for task in first) == 16 for family in families)


def test_protocol_file_hash_is_frozen():
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert digest == "21c959a86423505a68fa53969837b883989857072a599110e08ee21ee5bf5276"


def test_committed_pack_matches_generator_and_protocol_hash():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    blob = PACK.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == protocol["tasks_sha256"]
    rows = load_task_pack(blob.decode("utf-8"))
    verify_task_pack(rows)
    assert canonical_task_pack(generate_tasks()) == blob.decode("utf-8")
    assert task_pack_sha256(generate_tasks()) == protocol["tasks_sha256"]
    assert protocol["frozen_before_model_output"] is True
    assert "policy-update comparison" in protocol["acceptance"]["nondegenerate_base_signal"]


def test_parser_accepts_only_the_final_line():
    assert parse_final_integer("working\n#### 12") == 12
    assert parse_final_integer("#### 12\nextra") is None
    assert parse_final_integer("####12") is None
    assert grade_output("3 * 4 + 2", "#### 14") is True
    assert grade_output("3 * 4 + 2", "#### 15") is False
    assert evaluate_expression("(8 * 3) / 2") == 12


def test_wilson_interval_covers_known_center_and_bounds():
    low, high = wilson_interval(0, 64)
    assert low == 0.0
    assert 0.0 < high < 0.1
    low, high = wilson_interval(32, 64)
    assert low < 0.5 < high
    summary = pass_at_k({f"t{i}": [i == 0] + [False] * (SAMPLES_PER_TASK - 1) for i in range(4)})
    assert summary["decision"] == "nondegenerate_base_signal"
    assert pass_at_k({"t": [False] * SAMPLES_PER_TASK})["decision"] == "stop_zero"
    assert pass_at_k({"t": [True] + [False] * (SAMPLES_PER_TASK - 1)})["decision"] == "stop_saturated"


def test_gate_resumes_without_repeating_completed_samples(tmp_path: Path):
    tasks = [task.to_dict() for task in generate_tasks()]
    calls: list[tuple[str, int]] = []

    def sampler(prompt: str, seed: int) -> str:
        calls.append((prompt, seed))
        value = evaluate_expression(prompt.split(":", 1)[1].split("\n", 1)[0].strip())
        return f"#### {value}"

    digest = task_pack_sha256(generate_tasks())
    first = run_gate(tasks, sampler, tmp_path, pack_sha256=digest, backend="fake", model_id="fake")
    assert first["complete"] is True
    assert first["decision"] == "stop_saturated"
    call_count = len(calls)
    second = run_gate(tasks, sampler, tmp_path, pack_sha256=digest, backend="fake", model_id="fake")
    assert len(calls) == call_count
    assert second["passed_tasks"] == N_TASKS


def test_resume_rejects_a_different_model():
    row = {
        "backend": "mlx",
        "model_id": "one",
        "pack_sha256": "abc",
        "sample_index": 0,
        "task_id": "t",
    }
    with pytest.raises(ValueError, match="model"):
        completed_samples([row], pack_sha256="abc", backend="mlx", model_id="two")


def test_truncated_sample_line_fails_closed(tmp_path: Path):
    path = tmp_path / "samples.jsonl"
    path.write_text('{"a": 1}\n{"broken"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="partial"):
        read_jsonl(path)


def test_mlx_runner_uses_make_sampler():
    text = (ROOT / "scripts/run_heldout_exact_integer_base_gate_v1.py").read_text(encoding="utf-8")
    assert "from mlx_lm.sample_utils import make_sampler" in text
    assert "sampler = make_sampler(temp=TEMPERATURE)" in text
    assert "temp=TEMPERATURE," not in text


def test_update_comparison_is_refused():
    refusal = comparison_refusal()
    assert refusal["status"] == "refused"
    assert "GRPO" in refusal["reason"]
    script = ROOT / "scripts/heldout_exact_integer_update_comparison_v1.py"
    proc = subprocess.run([sys.executable, str(script)], check=False, capture_output=True, text=True)
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["status"] == "refused"
