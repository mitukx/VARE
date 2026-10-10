import json
import hashlib
from pathlib import Path

import pytest

from scripts.gpu_gate_journal_v3 import GateJournal, run_prompt_groups
from scripts.prompt_id_hash_v3 import (
    canonical_prompt_id_payload,
    canonical_prompt_id_sha256,
    verify_prompt_id_sha256,
)


PROTOCOL = json.loads(open("protocols/qwen25_deepmath_grpo_math500_v3.lock.json", encoding="utf-8").read())


def _group(_index, _row, cap):
    ids = [[1, 99, 0], [2, 3], [4], [5, 6]]
    assert all(len(tokens) <= cap for tokens in ids)
    return {"completion_token_ids": ids, "fixture": True}


def _fail_on(index_to_fail):
    def generate(index, row, cap):
        if index == index_to_fail:
            raise SimulatedOutOfMemoryError("fixture OOM")
        return _group(index, row, cap)

    return generate


class SimulatedOutOfMemoryError(RuntimeError):
    pass


def test_prompt_id_hash_serialization_matches_cpu_validator_and_frozen_gate_digest():
    ids = [f"{index:064x}" for index in range(32)]
    payload = canonical_prompt_id_payload(reversed(ids))
    assert payload.endswith(b"\n")
    assert payload.count(b"\n") == 32
    assert payload == ("\n".join(sorted(ids)) + "\n").encode("utf-8")
    assert canonical_prompt_id_sha256(ids) == canonical_prompt_id_sha256(reversed(ids))
    expected = canonical_prompt_id_sha256(ids)
    assert verify_prompt_id_sha256(reversed(ids), expected, expected_count=32) == expected
    with pytest.raises(ValueError, match="mismatch"):
        verify_prompt_id_sha256(ids, "f" * 64, expected_count=32)
    with pytest.raises(ValueError, match="expected 32"):
        verify_prompt_id_sha256(ids[:-1], expected, expected_count=32)
    assert PROTOCOL["training_data"]["sample"]["base_gate_id_serialization"].endswith(
        "including one final LF; SHA-256 the resulting bytes. Shared implementation: scripts/prompt_id_hash_v3.py."
    )
    assert PROTOCOL["training_data"]["sample"]["base_gate_id_list_sha256"] == (
        "0ab6b16559db37f6ff84bb312e18c29046063fac32fae026d5b1265037d03c9a"
    )


def test_protocol_execution_source_manifest_matches_working_tree():
    for name, expected in PROTOCOL["execution_code"]["files_sha256"].items():
        observed = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        assert observed == expected, name


@pytest.mark.parametrize("values", [[], ["A" * 64], ["0" * 64, "0" * 64], ["bad"]])
def test_prompt_id_hash_rejects_noncanonical_sets(values):
    with pytest.raises(ValueError):
        canonical_prompt_id_sha256(values)


def test_cumulative_metrics_survive_two_restarts_and_keep_budget_totals(tmp_path, monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr("scripts.gpu_gate_journal_v3.time.time", lambda: clock["now"])
    output = tmp_path / "gate.json"
    identity = {
        "experiment_budget": {"max_wall_seconds": 7200, "max_generated_tokens": 131072},
        "execution_code": {"git_revision": "commit-A", "sha256": "code-A"},
        "eos_token_id": 99,
    }
    rows = [{"i": index} for index in range(3)]

    first = GateJournal(output, identity)
    first.record_peak_vram(100, 130)
    with pytest.raises(SimulatedOutOfMemoryError):
        run_prompt_groups(rows=rows, journal=first, generate=_fail_on(1), completions_per_prompt=4,
                          max_new_tokens_per_completion=8, max_generated_tokens=131072,
                          max_wall_seconds=7200, eos_token_id=99)
    clock["now"] = 1_010.0
    first.stop("oom", "fixture")

    # Include 15 seconds of downtime since the first process was stopped.
    clock["now"] = 1_025.0
    second = GateJournal(output, identity, resume=True)
    assert second.elapsed() == pytest.approx(25.0)
    second.record_peak_vram(200, 240)
    with pytest.raises(SimulatedOutOfMemoryError):
        run_prompt_groups(rows=rows, journal=second, generate=_fail_on(2), completions_per_prompt=4,
                          max_new_tokens_per_completion=8, max_generated_tokens=131072,
                          max_wall_seconds=7200, eos_token_id=99)
    clock["now"] = 1_040.0
    second.stop("oom", "fixture")

    # Second downtime is also charged; no completed prompt is generated twice.
    clock["now"] = 1_060.0
    third = GateJournal(output, identity, resume=True)
    assert third.elapsed() == pytest.approx(60.0)
    third.record_peak_vram(180, 300)
    assert run_prompt_groups(rows=rows, journal=third, generate=_group, completions_per_prompt=4,
                             max_new_tokens_per_completion=8, max_generated_tokens=131072,
                             max_wall_seconds=7200, eos_token_id=99) == "complete"
    clock["now"] = 1_070.0
    result = {"decision": "pass", "resources": {}}
    third.finalize(result)
    saved = json.loads(output.read_text())
    assert saved["generated_tokens"] == 21  # 7 generated IDs per group, first EOS included once.
    assert saved["elapsed_seconds"] == pytest.approx(70.0)
    assert saved["peak_allocated_vram_bytes"] == 200
    assert saved["peak_reserved_vram_bytes"] == 300
    assert [r["prompt_index"] for r in saved["prompt_groups"]] == [0, 1, 2]
    assert all(r["generated_tokens"] == 7 for r in saved["prompt_groups"])
    assert saved["generated_tokens"] / saved["elapsed_seconds"] == pytest.approx(21 / 70)


@pytest.mark.parametrize(
    "changed_identity",
    [
        {"experiment_budget": {"max_wall_seconds": 3600, "max_generated_tokens": 131072},
         "execution_code": {"git_revision": "commit-A", "sha256": "code-A"}, "eos_token_id": 99},
        {"experiment_budget": {"max_wall_seconds": 7200, "max_generated_tokens": 65536},
         "execution_code": {"git_revision": "commit-A", "sha256": "code-A"}, "eos_token_id": 99},
        {"experiment_budget": {"max_wall_seconds": 7200, "max_generated_tokens": 131072},
         "execution_code": {"git_revision": "commit-B", "sha256": "code-A"}, "eos_token_id": 99},
        {"experiment_budget": {"max_wall_seconds": 7200, "max_generated_tokens": 131072},
         "execution_code": {"git_revision": "commit-A", "sha256": "code-B"}, "eos_token_id": 99},
    ],
)
def test_resume_rejects_changed_budget_or_execution_revision(tmp_path, changed_identity):
    output = tmp_path / "gate.json"
    identity = {
        "experiment_budget": {"max_wall_seconds": 7200, "max_generated_tokens": 131072},
        "execution_code": {"git_revision": "commit-A", "sha256": "code-A"},
        "eos_token_id": 99,
    }
    journal = GateJournal(output, identity)
    journal.stop("oom", "fixture")
    with pytest.raises(ValueError, match="identity differs"):
        GateJournal(output, changed_identity, resume=True)


def test_resume_recounts_saved_token_ids_and_fails_on_tampered_accounting(tmp_path, monkeypatch):
    output = tmp_path / "gate.json"
    identity = {"eos_token_id": 99, "experiment_budget": {"max_wall_seconds": 10, "max_generated_tokens": 100}}
    journal = GateJournal(output, identity)
    run_prompt_groups(rows=[{}], journal=journal, generate=_group, completions_per_prompt=4,
                      max_new_tokens_per_completion=8, max_generated_tokens=100,
                      max_wall_seconds=10, eos_token_id=99)
    journal.stop("interrupted", "fixture")
    lines = journal.partial_path.read_text().splitlines()
    record = json.loads(lines[0])
    record["generated_tokens"] += 1
    journal.partial_path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="accounting mismatch"):
        GateJournal(output, identity, resume=True)
