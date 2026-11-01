#!/usr/bin/env python3
"""Independently grade the VARE Development attempt for vLLM #64674.

Candidate source is parsed on the parent but executed only in VARE's
unprivileged, networkless candidate container. The parent retains every
expected value and compares the framed return values itself.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from vare.isolated_candidate import FunctionCall, grade_calls
try:
    from .pinned_workspace_git import pinned_git, read_regular_source
    from .run_local_agent import run_public_tests
except ImportError:
    from pinned_workspace_git import pinned_git, read_regular_source
    from run_local_agent import run_public_tests

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "vllm-64674-responses-custom-tool-stream-decode"
SOURCE_FILES = (
    "vllm/entrypoints/openai/responses/streaming_events.py",
    "vllm/entrypoints/openai/responses/utils.py",
)
PUBLIC_TEST = "test_public.py"
VENDOR = ROOT / "results/mlsys-fresh-vllm-64674-agent-attempt-v1/trusted/vendor"
ADAPTER = ROOT / "results/mlsys-fresh-vllm-64674-agent-attempt-v1/trusted/eval_adapter.py"
IMAGE = "vare-mlsys-agent-cpu:0.1@sha256:747b240f21c9d3107b5b8820dafccbb7639ea56f2ab692528b2e0f5fd84be1f1"
NAMESPACE = {
    "__future__", "Any", "Iterable", "Literal", "json", "partial_json_parser",
    "Allow", "CUSTOM_TOOL_INPUT_KEY", "SimpleStreamingState",
    "StreamingResponsesResponse", "ResponseCustomToolCallInputDeltaEvent",
    "ResponseFunctionCallArgumentsDeltaEvent",
}
UTILITY_NAMES = {
    "encode_custom_tool_input", "_custom_tool_payload",
    "decode_custom_tool_input_prefix", "extend_custom_tool_input_prefix",
    "_CUSTOM_TOOL_INPUT_HEAD",
}
STREAM_NAMES = {
    "_emit_custom_tool_input_delta", "_custom_tool_input_prefix",
    "emit_simple_tool_call_delta",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes())


def agent_public_test_passed(agent_attempt: dict[str, Any]) -> bool:
    result = agent_attempt.get("public_test_result")
    return isinstance(result, dict) and result.get("passed") is True


def definitions(source: str, wanted: set[str]) -> str:
    tree = ast.parse(source)
    bindings: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node
    pending = list(wanted)
    selected: set[int] = set()
    while pending:
        name = pending.pop()
        node = bindings.get(name)
        if node is None or id(node) in selected:
            continue
        selected.add(id(node))
        pending.extend(
            child.id for child in ast.walk(node)
            if isinstance(child, ast.Name)
            and child.id in bindings and child.id not in NAMESPACE
        )
    lines = source.splitlines()
    return "\n\n".join(
        "\n".join(lines[node.lineno - 1:node.end_lineno])
        for node in tree.body if id(node) in selected
    )


def build_subject(workspace: Path, subject: Path) -> dict[str, str]:
    subject.mkdir(parents=True, exist_ok=False)
    (subject / "vendor").mkdir()
    shutil.copytree(VENDOR / "partial_json_parser", subject / "vendor/partial_json_parser")
    prelude = '''from __future__ import annotations\nimport json\nimport sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))\nimport partial_json_parser\nfrom partial_json_parser.core.options import Allow\nfrom typing import Any, Iterable, Literal\nCUSTOM_TOOL_INPUT_KEY = "input"\nSimpleStreamingState = object\nStreamingResponsesResponse = object\nResponseCustomToolCallInputDeltaEvent = lambda **kwargs: kwargs\nResponseFunctionCallArgumentsDeltaEvent = lambda **kwargs: kwargs\n'''
    utils_path, stream_path = (workspace / p for p in SOURCE_FILES)
    utils_source = read_regular_source(workspace, SOURCE_FILES[1]).decode("utf-8")
    stream_source = read_regular_source(workspace, SOURCE_FILES[0]).decode("utf-8")
    selected = prelude + "\n" + definitions(utils_source, UTILITY_NAMES)
    selected += "\n" + definitions(stream_source, STREAM_NAMES)
    (subject / "candidate_module.py").write_text(selected, encoding="utf-8")
    shutil.copy2(ADAPTER, subject / "eval_adapter.py")
    return {
        "candidate_module_sha256": file_sha(subject / "candidate_module.py"),
        "eval_adapter_sha256": file_sha(ADAPTER),
        "vendor_package_sha256": sha256(b"".join(
            p.read_bytes() for p in sorted((subject / "vendor/partial_json_parser").rglob("*.py"))
        )),
    }


def decode_reference(prefix: str, parser: Any, allow: Any) -> str:
    try:
        parsed = parser.loads(prefix, allow.STR | allow.OBJ)
    except (ValueError, IndexError, AssertionError):
        return ""
    payload = parsed.get("input") if isinstance(parsed, dict) else None
    if not isinstance(payload, str):
        return ""
    if payload and "\ud800" <= payload[-1] <= "\udbff":
        payload = payload[:-1]
    return payload


def hidden_cases(parser: Any, allow: Any) -> tuple[list[list[list[str]]], list[list[str]]]:
    rng = random.Random(64674007)
    alphabet = ["a", " ", '"', "\\", "/", "\n", "\t", "\x01", "\x1f", "\x85"]
    alphabet += ["é", "　", "😀", "𝄞"]
    batches: list[list[list[str]]] = []
    expected: list[list[str]] = []
    for ensure_ascii in (False, True):
        for _ in range(60):
            payload = "".join(rng.choices(alphabet, k=rng.randint(0, 50)))
            encoded = json.dumps({"input": payload}, ensure_ascii=ensure_ascii)
            chunks: list[str] = []
            offset = 0
            while offset < len(encoded):
                width = rng.randint(1, 9)
                chunks.append(encoded[offset:offset + width])
                offset += width
            prefix = ""
            observed: list[str] = []
            for chunk in chunks:
                prefix += chunk
                observed.append(decode_reference(prefix, parser, allow))
            batches.append(chunks)
            expected.append(observed)
    return batches, expected


def count_chunks(lines: int, width: int) -> list[str]:
    encoded = json.dumps({"input": 'echo "x > y"\n' * lines})
    return [encoded[index:index + width] for index in range(0, len(encoded), width)]


def grade_candidate(
    *, workspace: Path, output: Path, baseline_revision: str,
    expected_source_hashes: dict[str, str], protocol_sha256: str,
    assignment_sha256: str, expected_public_test_sha256: str,
    expected_task_json_sha256: str, agent_attempt_path: Path,
    expected_model_repository: str, expected_model_revision: str,
    expected_model_files: dict[str, str],
    timeout_s: float = 45,
) -> dict[str, Any]:
    workspace = workspace.resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not ADAPTER.is_file() or not VENDOR.is_dir():
        raise FileNotFoundError("frozen hidden-grader inputs are missing")
    head = pinned_git(workspace, "rev-parse", "HEAD").decode().strip()
    if head != baseline_revision:
        raise ValueError("candidate workspace HEAD differs from the frozen baseline")
    baseline: dict[str, str] = {}
    current: dict[str, str] = {}
    for relative in SOURCE_FILES:
        baseline_bytes = pinned_git(workspace, "show", f"{baseline_revision}:{relative}")
        current_bytes = read_regular_source(workspace, relative)
        baseline[relative] = sha256(baseline_bytes)
        current[relative] = sha256(current_bytes)
    if baseline != expected_source_hashes:
        raise ValueError("candidate workspace baseline blobs differ from the frozen manifest")
    public_test_hash = file_sha(workspace / PUBLIC_TEST)
    if public_test_hash != expected_public_test_sha256:
        raise ValueError("candidate public test differs from the frozen test")
    agent_attempt_path = agent_attempt_path.resolve(strict=True)
    agent_attempt = json.loads(agent_attempt_path.read_text(encoding="utf-8"))
    changed = subprocess.check_output(
        ["git", "-C", str(workspace), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).splitlines()
    allowed_changes = set(SOURCE_FILES)
    reported = {line[3:] for line in changed if len(line) >= 4}
    if reported - allowed_changes:
        raise ValueError(f"candidate changed files outside the source allowlist: {sorted(reported - allowed_changes)}")
    diff = pinned_git(workspace, "diff", "--no-ext-diff", "--binary", "--no-color", baseline_revision)
    subject = output.parent / f"{output.name}-candidate-subject"
    parser_vendor = VENDOR.resolve()
    import sys
    sys.path.insert(0, str(parser_vendor))
    import partial_json_parser
    from partial_json_parser.core.options import Allow
    batches, expected_prefixes = hidden_cases(partial_json_parser, Allow)
    staged_hashes = build_subject(workspace, subject)
    prefix_report = grade_calls(
        subject,
        [FunctionCall(
            case_id="hidden_randomized_stream_prefixes",
            module="eval_adapter", function="stream_payloads", args=[batches],
            expected=expected_prefixes,
        )],
        task_kind="correctness", image=IMAGE, timeout_s=timeout_s,
        memory_mb=512, discard_candidate_stdout=True,
    )
    count_report = grade_calls(
        subject,
        [FunctionCall(
            case_id="hidden_input_size_reparse_invariance",
            module="eval_adapter", function="reparse_count_pair",
            args=[count_chunks(17, 7), count_chunks(170, 7)], expected=True,
        )],
        task_kind="correctness", image=IMAGE, timeout_s=timeout_s,
        memory_mb=512, discard_candidate_stdout=True,
    )
    public_report = run_public_tests(workspace, IMAGE, expected_public_test_sha256)
    trusted_pass = prefix_report.passed and count_report.passed
    source_before = agent_attempt.get("source_hashes_before", {})
    source_after = agent_attempt.get("source_hashes_after", {})
    agent_identity_valid = (
        agent_attempt.get("task_id") == TASK_ID
        and agent_attempt.get("task_revisions", {}).get("base") == baseline_revision
        and agent_attempt.get("task_input_hashes", {}).get("task.json") == expected_task_json_sha256
        and agent_attempt.get("model_repository") == expected_model_repository
        and agent_attempt.get("model_snapshot") == expected_model_revision
        and agent_attempt.get("model_files") == expected_model_files
        and all(source_before.get(path) == digest for path, digest in expected_source_hashes.items())
        and all(source_after.get(path) == digest for path, digest in current.items())
    )
    terminal_state_valid = (
        agent_attempt.get("stop_reason") == "agent_finished"
        and agent_attempt.get("edit_applied") is True
        and agent_public_test_passed(agent_attempt)
        and agent_identity_valid
    )
    outcome = (
        terminal_state_valid and trusted_pass
        and public_report.get("passed") is True and current != baseline
    )
    result = {
        "schema_version": 1,
        "grader_id": "mlsys-vllm-64674-agent-independent-grader-v1",
        "task_id": TASK_ID,
        "protocol_sha256": protocol_sha256,
        "assignment_sha256": assignment_sha256,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate": {
            "workspace_head": head,
            "baseline_source_sha256": baseline,
            "candidate_source_sha256": current,
            "changed_paths": sorted(reported),
            "diff_sha256": sha256(diff),
            "diff_bytes": len(diff),
        },
        "agent_terminal_state": {
            "record_sha256": file_sha(agent_attempt_path),
            "task_id_matches": agent_attempt.get("task_id") == TASK_ID,
            "task_source_revision_matches": agent_attempt.get("task_revisions", {}).get("base") == baseline_revision,
            "task_manifest_hash_matches": agent_attempt.get("task_input_hashes", {}).get("task.json") == expected_task_json_sha256,
            "model_identity_matches": agent_attempt.get("model_repository") == expected_model_repository and agent_attempt.get("model_snapshot") == expected_model_revision and agent_attempt.get("model_files") == expected_model_files,
            "stop_reason": agent_attempt.get("stop_reason"),
            "edit_applied": agent_attempt.get("edit_applied") is True,
            "public_test_pass_in_terminal_record": agent_public_test_passed(agent_attempt),
            "source_hashes_match": agent_identity_valid,
            "terminal_state_valid": terminal_state_valid,
            "tool_turns": agent_attempt.get("tool_turns"),
            "new_tokens": agent_attempt.get("new_tokens"),
            "elapsed_seconds": agent_attempt.get("elapsed_seconds"),
        },
        "grader_inputs": {
            "adapter_sha256": file_sha(ADAPTER),
            "vendor_package_sha256": staged_hashes["vendor_package_sha256"],
            "image": IMAGE,
            "hidden_seed": 64674007,
            "randomized_payloads": 120,
            "public_test_sha256": public_test_hash,
            "source_path_hashes_verified": baseline == expected_source_hashes,
        },
        "public_source_change_required": True,
        "public_test_grade": public_report,
        "hidden_prefix_grade": prefix_report.to_dict(),
        "hidden_reparse_grade": count_report.to_dict(),
        "trusted_grade_pass": trusted_pass,
        "final_success": outcome,
        "feedback_to_model": False,
        "trust_boundary": "candidate implementation is AST-selected without execution in the parent and called in VARE's no-network, read-only-root, unprivileged candidate container; hidden expected values remain in the parent",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--source-hashes", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--assignment-sha256", required=True)
    parser.add_argument("--public-test-sha256", required=True)
    parser.add_argument("--task-json-sha256", required=True)
    parser.add_argument("--agent-attempt", type=Path, required=True)
    parser.add_argument("--model-repository", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-files", type=Path, required=True)
    args = parser.parse_args()
    hashes = json.loads(args.source_hashes.read_text(encoding="utf-8"))
    model_files = json.loads(args.model_files.read_text(encoding="utf-8"))
    report = grade_candidate(
        workspace=args.workspace, output=args.output,
        baseline_revision=args.baseline_revision,
        expected_source_hashes=hashes,
        protocol_sha256=args.protocol_sha256,
        assignment_sha256=args.assignment_sha256,
        expected_public_test_sha256=args.public_test_sha256,
        expected_task_json_sha256=args.task_json_sha256,
        agent_attempt_path=args.agent_attempt,
        expected_model_repository=args.model_repository,
        expected_model_revision=args.model_revision,
        expected_model_files=model_files,
    )
    print(json.dumps({"trusted_grade_pass": report["trusted_grade_pass"], "final_success": report["final_success"]}))
    return 0 if report["trusted_grade_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
