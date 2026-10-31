"""Offline deterministic E0 controls for evidence binding; no candidate is run."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts/record_coding_agent_baseline.py"
SPEC = importlib.util.spec_from_file_location("record_coding_agent_baseline", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fixture(tmp_path: Path) -> dict:
    root = tmp_path / "historical"
    root.mkdir()
    source_path = "src/pkg/x.py"
    repo = "https://github.com/example/public-task.git"
    commit = "f" * 40
    task_id = "real-task-locked-v1"
    task = {
        "schema_version": 1, "task_id": task_id,
        "source": {"repository": repo, "base_revision": commit,
                   "files": [source_path]},
        "grader": {"path": "benchmarks/historical/test/evaluator/grade.py",
                   "protocol_lock": "benchmarks/historical/test/protocol.lock.json"},
    }
    dump(root / "task.json", task)
    (root / "TASK.md").write_text("Public historical calibration control\n")
    (root / "evaluator").mkdir()
    (root / "evaluator/grade.py").write_text("# Trusted grader fixture; not executed\n")
    file_hashes = {
        "task.json": MODULE.sha256_file(root / "task.json"),
        "TASK.md": MODULE.sha256_file(root / "TASK.md"),
        "grader.py": MODULE.sha256_file(root / "evaluator/grade.py"),
    }
    lock = {"schema_version": 1, "task_id": task_id,
            "protocol_id": "historical-protocol-v1",
            "locked_hashes": {
                "task_json_sha256": file_hashes["task.json"],
                "task_brief_sha256": file_hashes["TASK.md"],
                "evaluator_sha256": file_hashes["grader.py"],
            }}
    dump(root / "protocol.lock.json", lock)
    file_hashes["protocol.lock.json"] = MODULE.sha256_file(root / "protocol.lock.json")
    protocol = {
        "schema_version": 1, "protocol_id": "coding-agent-baseline-test-v1",
        "task": {"task_id": task_id, "repository": repo, "base_revision": commit,
                 "grader_path": task["grader"]["path"],
                 "lock_path": task["grader"]["protocol_lock"],
                 "task_protocol_id": lock["protocol_id"], "hashes": file_hashes},
        "model": {"repository": "test/model", "snapshot": "pinned-hash",
                  "file_hashes": {"config.json": "a" * 64}},
        "seeds": [101],
        "budgets": {"max_turns": 10, "max_new_tokens_per_turn": 256,
                    "max_total_new_tokens": 2560, "wall_seconds": 300,
                    "incomplete_tool_retry_tokens": None},
    }
    diff = ("diff --git a/src/pkg/x.py b/src/pkg/x.py\n"
            "index 0000000..1111111 100644\n"
            "--- a/src/pkg/x.py\n+++ b/src/pkg/x.py\n"
            "@@ -1 +1 @@\n-old\n+new\n")
    agent = {
        "schema_version": 1, "task_id": task_id,
        "task_input_hashes": file_hashes,
        "task_revisions": {"base": commit, "calibration": "e" * 40},
        "workspace_revision": commit,
        "model_repository": "test/model", "model_snapshot": "pinned-hash",
        "model_files": {"config.json": "a" * 64}, "seed": 101,
        "budgets": copy.deepcopy(protocol["budgets"]),
        "tool_turns": 2, "new_tokens": 132,
        "stop_reason": "agent_finished", "elapsed_seconds": 12.3,
        "transcript": [
            {"turn": 1, "model_output": "patch", "tool_calls": [
                {"name": "apply_patch", "arguments": {"patch": "..."},
                 "result": {"applied": True}}]},
            {"turn": 2, "model_output": "finish", "tool_calls": [
                {"name": "finish", "arguments": {"summary": "done"},
                 "result": {"finished": True}}]},
        ],
        "offline_environment_required": True,
        "initial_messages": [{"role": "user", "content": "task"}],
        "runtime": {"python": "3.13", "device": "cpu"},
        "source_hashes_before": {source_path: digest(b"old\n")},
        "source_hashes_after": {source_path: digest(b"new\n")},
        "candidate_diff": diff, "candidate_diff_sha256": digest(diff.encode()),
    }
    grade = {"schema_version": 1, "task_id": task_id,
             "workspace_revision": commit,
             "task_descriptor_sha256": file_hashes["task.json"],
             "evaluator_sha256": file_hashes["grader.py"],
             "candidate_diff_sha256": digest(diff.encode()),
             "source_file_sha256": copy.deepcopy(agent["source_hashes_after"]),
             "passed": True, "failures": [], "checks": {"integration": "ok"}}
    return {"task_root": root, "protocol": protocol, "agent": agent, "grade": grade}


def run(tmp_path: Path, records: dict, index: int = 0) -> tuple[int, dict, Path]:
    files = tmp_path / f"records-{index}"
    dump(files / "protocol.json", records["protocol"])
    dump(files / "agent.json", records["agent"])
    dump(files / "grade.json", records["grade"])
    output = tmp_path / f"evidence-{index}"
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--protocol", str(files / "protocol.json"),
        "--task-root", str(records["task_root"]),
        "--agent-record", str(files / "agent.json"),
        "--grader-record", str(files / "grade.json"), "--output-dir", str(output),
    ], capture_output=True, text=True, check=False)
    assert output.is_dir(), result.stderr
    assert (output / "manifest.json").exists()
    assert (output / "inputs/agent.json").exists()
    return result.returncode, json.loads((output / "summary.json").read_text()), output


def test_independent_grader_success_and_deterministic_manifest(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    status, summary, output = run(tmp_path, data)
    assert status == 0
    assert summary["record_integrity"] == "valid"
    assert summary["task_success"] is True
    assert summary["changed_paths"] == ["src/pkg/x.py"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["files"]["inputs/agent.json"] == MODULE.sha256_file(output / "inputs/agent.json")
    _, summary2, output2 = run(tmp_path, data, index=1)
    assert summary == summary2
    assert (output / "manifest.json").read_bytes() == (output2 / "manifest.json").read_bytes()


def test_configured_retry_cap_requires_matching_turn_evidence(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    data["protocol"]["budgets"]["incomplete_tool_retry_tokens"] = 512
    data["agent"]["budgets"]["incomplete_tool_retry_tokens"] = 512
    patch_call = copy.deepcopy(data["agent"]["transcript"][0]["tool_calls"][0])
    finish_call = copy.deepcopy(data["agent"]["transcript"][1]["tool_calls"][0])
    data["agent"]["tool_turns"] = 3
    data["agent"]["transcript"] = [
        {
            "turn": 1,
            "model_output": '<tool_call>{"name":"apply_patch"',
            "tool_calls": [],
            "tool_call_output_status": "truncated_at_token_limit",
            "automatic_reminder": 1,
            "generation_token_limit": 256,
        },
        {
            "turn": 2, "model_output": "patch",
            "tool_calls": [patch_call], "generation_token_limit": 512,
        },
        {
            "turn": 3, "model_output": "finish",
            "tool_calls": [finish_call], "generation_token_limit": 256,
        },
    ]
    data["agent"]["tool_call_diagnostics"] = {
        "expanded_incomplete_retry_turns": [2],
    }

    status, summary, _ = run(tmp_path, data)

    assert status == 0
    assert summary["record_integrity"] == "valid"

    data["agent"]["tool_call_diagnostics"]["expanded_incomplete_retry_turns"] = []
    status, summary, _ = run(tmp_path, data, index=1)
    assert status == 1
    assert summary["record_integrity"] == "invalid"


def test_valid_negative_grader_is_preserved_not_rejected(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    data["grade"].update(passed=False, failures=[{"case": "regression"}])
    status, summary, _ = run(tmp_path, data)
    assert status == 0 and summary["task_success"] is False
    assert summary["failure_category"] == "independent_grader_reject"


def test_no_op_cannot_pass(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    data["agent"]["candidate_diff"] = ""
    data["agent"]["candidate_diff_sha256"] = digest(b"")
    data["grade"]["candidate_diff_sha256"] = digest(b"")
    data["agent"]["source_hashes_after"] = copy.deepcopy(data["agent"]["source_hashes_before"])
    data["grade"]["source_file_sha256"] = copy.deepcopy(data["agent"]["source_hashes_after"])
    status, summary, _ = run(tmp_path, data)
    assert status == 1
    assert summary["failure_category"] == "invalid_evidence"


@pytest.mark.parametrize("mutation", ["protected_patch", "bad_diff", "source_revision",
                                       "grader_hash", "task_id", "tampered_protocol",
                                       "token_budget", "missing_transcript", "tampered_diff",
                                       "agent_forged_finish", "modified_evaluator", "model_hash",
                                       "grader_source_hash", "tool_result_type", "bad_hunk_counts",
                                       "retry_token_budget", "invalid_retry_cap"])
def test_mutations_fail_closed(tmp_path: Path, mutation: str) -> None:
    data = fixture(tmp_path)
    if mutation == "protected_patch":
        patch = data["agent"]["candidate_diff"].replace("src/pkg/x.py", "evaluator/grade.py")
        data["agent"].update(candidate_diff=patch, candidate_diff_sha256=digest(patch.encode()))
        data["grade"]["candidate_diff_sha256"] = digest(patch.encode())
    elif mutation == "bad_diff":
        data["agent"]["candidate_diff"] = "not a diff\n"
        data["agent"]["candidate_diff_sha256"] = digest(b"not a diff\n")
        data["grade"]["candidate_diff_sha256"] = digest(b"not a diff\n")
    elif mutation == "bad_hunk_counts":
        patch = data["agent"]["candidate_diff"].replace("@@ -1 +1 @@", "@@ -1,4 +1 @@")
        data["agent"].update(candidate_diff=patch, candidate_diff_sha256=digest(patch.encode()))
        data["grade"]["candidate_diff_sha256"] = digest(patch.encode())
    elif mutation == "source_revision":
        data["grade"]["workspace_revision"] = "e" * 40
    elif mutation == "grader_hash":
        data["grade"]["evaluator_sha256"] = "0" * 64
    elif mutation == "task_id":
        data["agent"]["task_id"] = "different-task"
    elif mutation == "tampered_protocol":
        data["protocol"]["task"]["hashes"]["grader.py"] = "0" * 64
    elif mutation == "token_budget":
        data["agent"]["new_tokens"] = 3000
    elif mutation == "retry_token_budget":
        data["protocol"]["budgets"]["incomplete_tool_retry_tokens"] = 512
    elif mutation == "invalid_retry_cap":
        data["protocol"]["budgets"]["incomplete_tool_retry_tokens"] = 256
        data["agent"]["budgets"]["incomplete_tool_retry_tokens"] = 256
    elif mutation == "missing_transcript":
        data["agent"]["transcript"] = []
    elif mutation == "tampered_diff":
        data["grade"]["candidate_diff_sha256"] = "0" * 64
    elif mutation == "agent_forged_finish":
        data["agent"]["transcript"][1]["tool_calls"][0]["result"] = {"finished": False}
    elif mutation == "modified_evaluator":
        (data["task_root"] / "evaluator/grade.py").write_text("# edited!\n")
    elif mutation == "model_hash":
        data["agent"]["model_files"]["config.json"] = "0" * 64
    elif mutation == "grader_source_hash":
        data["grade"]["source_file_sha256"]["src/pkg/x.py"] = "0" * 64
    elif mutation == "tool_result_type":
        data["agent"]["transcript"][0]["tool_calls"][0]["result"] = "broken"
    status, summary, _ = run(tmp_path, data)
    assert status == 1
    assert summary["record_integrity"] == "invalid"
    assert summary["task_success"] is False


def test_timeout_is_retained_as_negative_result(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    data["agent"]["stop_reason"] = "wall_time_exhausted"
    data["agent"]["transcript"][1]["tool_calls"] = []
    data["grade"]["passed"] = False
    data["grade"]["failures"] = ["task timed out"]
    status, summary, _ = run(tmp_path, data)
    assert status == 0
    assert summary["failure_category"] == "incomplete_agent"
    assert summary["stop_reason"] == "wall_time_exhausted"


def test_missing_grade_fails_closed_and_keeps_agent_record(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    files = tmp_path / "records-x"
    dump(files / "protocol.json", data["protocol"])
    dump(files / "agent.json", data["agent"])
    proc = subprocess.run([sys.executable, str(SCRIPT),
        "--protocol", str(files / "protocol.json"), "--task-root", str(data["task_root"]),
        "--agent-record", str(files / "agent.json"),
        "--grader-record", str(files / "missing.json"),
        "--output-dir", str(tmp_path / "failure")], capture_output=True, text=True)
    assert proc.returncode == 1
    assert json.loads((tmp_path / "failure/summary.json").read_text())["task_success"] is False
    assert (tmp_path / "failure/inputs/agent.json").is_file()


def test_exclusive_output_directory(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    assert run(tmp_path, data)[0] == 0
    files = tmp_path / "records-0"
    proc = subprocess.run([sys.executable, str(SCRIPT),
        "--protocol", str(files / "protocol.json"), "--task-root", str(data["task_root"]),
        "--agent-record", str(files / "agent.json"),
        "--grader-record", str(files / "grade.json"),
        "--output-dir", str(tmp_path / "evidence-0")], capture_output=True, text=True)
    assert proc.returncode != 0
