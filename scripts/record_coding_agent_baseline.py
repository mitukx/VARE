#!/usr/bin/env python3
"""Bind an existing coding-agent trajectory to external grader evidence.

This tool NEVER evaluates or executes candidate code. Run the task's unchanged
trusted grader in an appropriately isolated environment first, then pass its
JSON result to this offline consistency auditor. Hashes are not signatures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from verify_agent_workspace import verify_workspace

MAX_INPUT_BYTES = 32 * 1024 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_ALLOWED_STOP_REASONS = {
    "agent_finished", "wall_time_exhausted", "token_budget_exhausted",
    "turn_budget_exhausted", "no_tool_call_after_edit",
    "no_tool_call_retries_exhausted",
}


class EvidenceError(ValueError):
    """An input record cannot support a reproducible result."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise EvidenceError(f"missing or oversized JSON: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=lambda name: (_ for _ in ()).throw(
                               EvidenceError(f"nonfinite JSON value: {name}")))
    except (UnicodeError, ValueError) as exc:
        raise EvidenceError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON root must be an object: {path.name}")
    return value


def expect(ok: bool, message: str) -> None:
    if not ok:
        raise EvidenceError(message)


def positive_int(value: Any, field: str) -> int:
    expect(type(value) is int and value > 0, f"invalid positive integer: {field}")
    return value


def nonnegative_int(value: Any, field: str) -> int:
    expect(type(value) is int and value >= 0, f"invalid nonnegative integer: {field}")
    return value


def checked_sha(value: Any, field: str) -> str:
    expect(isinstance(value, str) and SHA256.fullmatch(value) is not None,
           f"invalid SHA-256: {field}")
    return value


def parse_diff(diff: str, allowed: set[str]) -> list[str]:
    """Only ordinary modifications of allowlisted, existing source paths."""
    expect(isinstance(diff, str) and len(diff.encode("utf-8")) <= 2 * 1024 * 1024,
           "candidate diff missing or too large")
    if not diff:
        return []
    sections = re.split(r"(?=^diff --git )", diff, flags=re.MULTILINE)
    expect(sections and sections[0] == "", "malformed diff header")
    paths: list[str] = []
    for section in sections[1:]:
        lines = section.splitlines()
        expect(bool(lines), "empty diff section")
        marker = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", lines[0])
        expect(marker is not None, "malformed diff --git header")
        assert marker is not None
        a, b = marker.groups()
        expect(a == b and a in allowed and ".." not in Path(a).parts,
               "diff modifies non-allowlisted or protected path")
        expect(not any(line.startswith(("new file mode", "deleted file mode", "rename ",
                                            "copy ", "GIT binary patch", "Binary files "))
                       for line in lines), "diff contains create/delete/rename/binary operation")
        expect(f"--- a/{a}" in lines and f"+++ b/{a}" in lines,
               "diff source/destination headers missing or mismatched")
        expect(not any(line.startswith(("old mode ", "new mode ")) for line in lines),
               "diff changes file permissions")
        hunks = [i for i, line in enumerate(lines) if line.startswith("@@ ")]
        expect(bool(hunks), "diff lacks hunks")
        expect(lines.count(f"--- a/{a}") == 1 and lines.count(f"+++ b/{a}") == 1,
               "duplicate or ambiguous diff file headers")
        for position, hunk_start in enumerate(hunks):
            header = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*", lines[hunk_start])
            expect(header is not None, "malformed unified diff hunk header")
            assert header is not None
            expected_old = int(header.group(2) or 1)
            expected_new = int(header.group(4) or 1)
            observed_old = observed_new = 0
            end = hunks[position + 1] if position + 1 < len(hunks) else len(lines)
            for line in lines[hunk_start + 1:end]:
                if line.startswith("\\ No newline at end of file"):
                    continue
                expect(bool(line) and line[0] in (" ", "+", "-"),
                       "malformed unified diff hunk line")
                observed_old += line[0] != "+"
                observed_new += line[0] != "-"
            expect((observed_old, observed_new) == (expected_old, expected_new),
                   "unified diff hunk line counts do not match header")
        paths.append(a)
    expect(len(paths) == len(set(paths)), "duplicate diff path")
    return paths


def inspect(protocol: dict[str, Any], task: dict[str, Any], lock: dict[str, Any],
            agent: dict[str, Any], grade: dict[str, Any], files: dict[str, Path]) -> dict[str, Any]:
    expect(protocol.get("schema_version") == 1 and
           isinstance(protocol.get("protocol_id"), str), "unsupported protocol")
    cfg = protocol.get("task", {})
    expect(type(cfg) is dict and type(task.get("source")) is dict, "invalid task configuration")
    task_id = cfg.get("task_id")
    expected_repo = cfg.get("repository")
    base = cfg.get("base_revision")
    expect(type(base) is str and COMMIT.fullmatch(base) is not None, "invalid pinned revision")
    expect(task.get("task_id") == lock.get("task_id") == agent.get("task_id") ==
           grade.get("task_id") == task_id, "task identity mismatch")
    expect(task["source"].get("repository") == expected_repo and
           task["source"].get("base_revision") == base, "source identity mismatch")
    expect(task.get("grader", {}).get("path") == cfg.get("grader_path"),
           "grader selection mismatch")
    expect(task.get("grader", {}).get("protocol_lock") == cfg.get("lock_path"),
           "grader lock path mismatch")
    expected_hashes = cfg.get("hashes", {})
    actual_hashes = {
        "task.json": sha256_file(files["task.json"]),
        "TASK.md": sha256_file(files["TASK.md"]),
        "protocol.lock.json": sha256_file(files["protocol.lock.json"]),
        "grader.py": sha256_file(files["grader.py"]),
    }
    for name, actual in actual_hashes.items():
        expect(checked_sha(expected_hashes.get(name), f"protocol.{name}") == actual,
               f"frozen protocol hash mismatch: {name}")
    locked = lock.get("locked_hashes", {})
    for key, name in (("task_json_sha256", "task.json"),
                      ("task_brief_sha256", "TASK.md"),
                      ("evaluator_sha256", "grader.py")):
        expect(locked.get(key) == actual_hashes[name], f"task protocol lock mismatch: {name}")
    expect(lock.get("protocol_id") == cfg.get("task_protocol_id"), "task lock ID mismatch")
    input_hashes = agent.get("task_input_hashes", {})
    expect(isinstance(input_hashes, dict), "agent task input hashes missing")
    for name, actual in actual_hashes.items():
        expect(input_hashes.get(name) == actual, f"agent input hash mismatch: {name}")
    expect(agent.get("schema_version") == 1 and grade.get("schema_version") == 1,
           "unsupported run or grader schema")
    expect(agent.get("task_revisions", {}).get("base") == base and
           agent.get("workspace_revision") == base and grade.get("workspace_revision") == base,
           "candidate source revision mismatch")
    expect(grade.get("task_descriptor_sha256") == actual_hashes["task.json"] and
           grade.get("evaluator_sha256") == actual_hashes["grader.py"],
           "grader provenance mismatch")
    model = protocol.get("model", {})
    expect(agent.get("model_repository") == model.get("repository") and
           agent.get("model_snapshot") == model.get("snapshot"), "model identity mismatch")
    expect(isinstance(agent.get("model_files"), dict) and bool(agent["model_files"]),
           "model file hashes missing")
    for name, digest in model.get("file_hashes", {}).items():
        expect(agent["model_files"].get(name) == checked_sha(digest, name),
               f"model snapshot hash mismatch: {name}")
    for digest in agent["model_files"].values():
        checked_sha(digest, "agent model file")
    expect(type(agent.get("seed")) is int and agent["seed"] in protocol.get("seeds", []),
           "seed not registered in frozen protocol")
    limits = protocol.get("budgets", {})
    agent_limits = agent.get("budgets", {})
    for name in ("max_turns", "max_new_tokens_per_turn", "max_total_new_tokens", "wall_seconds"):
        actual = positive_int(agent_limits.get(name), "agent budget " + name)
        expect(actual == positive_int(limits.get(name), "protocol budget " + name),
               f"budget mismatch: {name}")
    expected_retry_limit = limits.get("incomplete_tool_retry_tokens")
    observed_retry_limit = agent_limits.get("incomplete_tool_retry_tokens")
    if expected_retry_limit is not None:
        expected_retry_limit = positive_int(expected_retry_limit, "protocol budget incomplete_tool_retry_tokens")
        observed_retry_limit = positive_int(observed_retry_limit, "agent budget incomplete_tool_retry_tokens")
    expect(observed_retry_limit == expected_retry_limit,
           "budget mismatch: incomplete_tool_retry_tokens")
    steps = nonnegative_int(agent.get("tool_turns"), "tool_turns")
    tokens = nonnegative_int(agent.get("new_tokens"), "new_tokens")
    transcript = agent.get("transcript")
    expect(type(transcript) is list and len(transcript) == steps and
           steps <= limits["max_turns"] and tokens <= limits["max_total_new_tokens"],
           "missing trajectory or exceeded steps/tokens")
    if expected_retry_limit is not None:
        expanded_turns = []
        base_turn_limit = limits["max_new_tokens_per_turn"]
        expect(expected_retry_limit > base_turn_limit,
               "protocol retry budget must exceed the base per-turn budget")
        expect(expected_retry_limit <= limits["max_total_new_tokens"],
               "protocol retry budget exceeds the total token budget")
        for index, turn in enumerate(transcript):
            expect(type(turn) is dict, "malformed trajectory turn")
            turn_limit = positive_int(
                turn.get("generation_token_limit"), "turn generation_token_limit",
            )
            expect(turn_limit <= expected_retry_limit,
                   "turn generation limit exceeds protocol retry budget")
            if turn_limit > base_turn_limit:
                previous = transcript[index - 1] if index else {}
                expect(
                    previous.get("tool_call_output_status") in (
                        "truncated_at_token_limit", "unterminated_tool_call",
                    ) and type(previous.get("automatic_reminder")) is int
                    and previous["automatic_reminder"] > 0,
                    "expanded turn lacks an immediately preceding incomplete-call retry",
                )
                expanded_turns.append(turn.get("turn"))
        diagnostics = agent.get("tool_call_diagnostics")
        expect(type(diagnostics) is dict, "missing tool-call diagnostics")
        recorded_expanded = diagnostics.get(
            "expanded_incomplete_retry_turns",
        )
        expect(recorded_expanded == expanded_turns,
               "expanded retry turns do not match per-turn budgets")
    expect(agent.get("stop_reason") in _ALLOWED_STOP_REASONS, "invalid stop reason")
    elapsed = agent.get("elapsed_seconds")
    expect(type(elapsed) in (float, int) and 0 <= elapsed < float("inf"),
           "invalid elapsed duration")
    tool_errors = 0
    tool_calls = 0
    finished = False
    applied = False
    allowed_tools = set(protocol.get("interaction", {}).get("allowed_tools", ()))
    for i, turn in enumerate(transcript, 1):
        expect(type(turn) is dict and turn.get("turn") == i and
               isinstance(turn.get("model_output"), str) and
               type(turn.get("tool_calls")) is list, "malformed trajectory turn")
        for call in turn["tool_calls"]:
            expect(type(call) is dict, "malformed tool call")
            tool_calls += 1
            if allowed_tools and "name" in call:
                expect(call["name"] in allowed_tools, "undeclared agent tool")
            if call.get("name") == "apply_patch" and call.get("result", {}).get("applied") is True:
                applied = True
            expect("result" not in call or type(call["result"]) is dict,
                   "malformed tool result")
            if "error" in call or (isinstance(call.get("result"), dict) and
                                  call["result"].get("error")):
                tool_errors += 1
            if call.get("name") == "finish" and call.get("result", {}).get("finished") is True:
                finished = True
    expect(agent.get("stop_reason") != "agent_finished" or finished,
           "agent-finished status without a successful finish call")
    expect(agent.get("offline_environment_required") is True,
           "offline execution requirement not recorded")
    expect(type(agent.get("initial_messages")) is list and
           type(agent.get("runtime")) is dict, "missing prompt or runtime provenance")
    before, after = agent.get("source_hashes_before"), agent.get("source_hashes_after")
    expect(type(before) is dict and type(after) is dict and
           set(before) == set(task["source"]["files"]) == set(after),
           "incomplete source-file provenance")
    for values in (before, after):
        for digest in values.values():
            checked_sha(digest, "source file")
    diff = agent.get("candidate_diff")
    changed_paths = parse_diff(diff, set(task["source"]["files"]))
    diff_sha = sha256_bytes(diff.encode("utf-8"))
    expect(agent.get("candidate_diff_sha256") == diff_sha ==
           grade.get("candidate_diff_sha256"), "candidate/grader diff hash mismatch")
    expect(not changed_paths or applied, "source changes without a recorded successful patch tool")
    expect(set(changed_paths) == {p for p in before if before[p] != after[p]},
           "source hashes and modified file paths disagree")
    expect(grade.get("source_file_sha256") == after,
           "grader source-file hashes do not match agent final workspace")
    expect(type(grade.get("passed")) is bool and isinstance(grade.get("failures"), list)
           and isinstance(grade.get("checks"), dict),
           "invalid grader decision or missing diagnostic checks")
    expect(not grade.get("grader_error"), "grader reported an execution/provenance error")
    expect(not (grade["passed"] and grade["failures"]),
           "contradictory grader outcome")
    is_valid_attempt = bool(changed_paths) and agent["stop_reason"] == "agent_finished"
    success = bool(grade["passed"] and is_valid_attempt)
    if grade["passed"] and not is_valid_attempt:
        raise EvidenceError("grader passed but attempt is incomplete or no-op")
    return {
        "schema_version": 1, "protocol_id": protocol["protocol_id"],
        "task_id": task_id, "source_repository": expected_repo,
        "task_protocol_sha256": actual_hashes["protocol.lock.json"],
        "trusted_grader_sha256": actual_hashes["grader.py"],
        "source_revision": base, "model_repository": model["repository"],
        "model_snapshot": model["snapshot"], "seed": agent["seed"],
        "steps": steps, "tokens": tokens, "elapsed_seconds": elapsed,
        "tool_calls": tool_calls, "tool_errors": tool_errors,
        "runtime": agent["runtime"], "peak_rss_bytes": agent.get("max_rss_bytes"),
        "stop_reason": agent["stop_reason"], "diff_sha256": diff_sha,
        "changed_paths": changed_paths, "grader_passed": grade["passed"],
        "task_success": success,
        "failure_category": ("success" if success else
                             "no_source_change" if not changed_paths else
                             "incomplete_agent" if not is_valid_attempt else
                             "independent_grader_reject"),
        "evidence_class": "E1 historical-task measurement (not held-out, no model update)",
        "attestation_limit": "SHA-256 consistency only; no independently signed grader attestation",
        "resource_cost": "not observed unless independently supplied",
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                               sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--agent-record", type=Path, required=True)
    parser.add_argument("--grader-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path,
                        help="verify the actual prepared Git checkout without executing its code")
    args = parser.parse_args()
    protocol_path, task_root, agent_path, grader_path, output = (
        path.expanduser().resolve() for path in
        (args.protocol, args.task_root, args.agent_record, args.grader_record, args.output_dir)
    )
    workspace = args.workspace.expanduser().resolve() if args.workspace else None
    source_paths = {
        "protocol.json": protocol_path,
        "task.json": task_root / "task.json",
        "TASK.md": task_root / "TASK.md",
        "protocol.lock.json": task_root / "protocol.lock.json",
        "grader.py": task_root / "evaluator" / "grade.py",
        "agent.json": agent_path, "grade.json": grader_path,
    }
    if output.exists() or any(output == p or output.is_relative_to(p)
                              for p in [task_root, *source_paths.values(), *([workspace] if workspace else [])]):
        parser.error("output must be new and separate from all input/locked files")
    if any(p == output or p.is_relative_to(output) for p in source_paths.values()):
        parser.error("output would contain or overwrite an input")
    output.mkdir(parents=True)
    raw = output / "inputs"
    raw.mkdir()
    for name, path in source_paths.items():
        if path.is_file() and path.stat().st_size <= MAX_INPUT_BYTES:
            shutil.copyfile(path, raw / name)
    try:
        protocol = load_json(raw / "protocol.json")
        task = load_json(raw / "task.json")
        lock = load_json(raw / "protocol.lock.json")
        agent = load_json(raw / "agent.json")
        grade = load_json(raw / "grade.json")
        copied_paths = {name: raw / name for name in source_paths}
        result = inspect(protocol, task, lock, agent, grade, copied_paths)
        requires_workspace = protocol.get("workspace_verification", {}).get("required", False)
        if requires_workspace and workspace is None:
            raise EvidenceError("frozen protocol requires actual workspace verification")
        if workspace is not None:
            if any(workspace == p or workspace.is_relative_to(p) or p.is_relative_to(workspace)
                   for p in [task_root, *source_paths.values(), output]):
                raise EvidenceError("workspace must be isolated from trusted inputs and output")
            result["workspace_audit"] = verify_workspace(workspace, task, agent, grade)
        result["record_integrity"] = "valid"
        status = 0
    except (ValueError, KeyError, TypeError, OSError, AttributeError, UnicodeError,
            subprocess.SubprocessError) as exc:
        result = {"schema_version": 1, "record_integrity": "invalid",
                  "task_success": False, "failure_category": "invalid_evidence",
                  "error_type": type(exc).__name__, "error": str(exc)}
        status = 1
    write_json(output / "summary.json", result)
    manifest = {"schema_version": 1, "files": {
        str(p.relative_to(output)): sha256_file(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"
    }}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), **result}, sort_keys=True))
    return status


if __name__ == "__main__":
    sys.exit(main())
