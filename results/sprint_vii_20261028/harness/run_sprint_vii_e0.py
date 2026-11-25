#!/usr/bin/env python3
"""Run the frozen, deterministic Sprint VII E0 edit-acceptance gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.agent_function_edit_v1 import FunctionEditLeaseStoreV2  # noqa: E402

RUNNER_PATH = Path(__file__).resolve()
GRADER_PATH = ROOT / "scripts" / "sprint_vii_e0_grader.py"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "GIT_EXTERNAL_DIFF": "",
    })
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    proc = subprocess.run(
        ["git", "-C", str(root), *args], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, env=env,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {proc.stderr.decode('utf-8', 'replace')[:500]}")
    return proc


def run_grader(root: Path, source_name: str, expected_sha: str) -> dict[str, Any]:
    command = [
        sys.executable, "-I", str(GRADER_PATH), "--workspace-root", str(root),
        "--source", source_name, "--expected-sha256", expected_sha,
    ]
    started = time.perf_counter()
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=20, check=False, cwd=ROOT)
    elapsed = round(time.perf_counter() - started, 6)
    result: dict[str, Any] = {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed_seconds": elapsed,
    }
    if proc.returncode != 0:
        return {**result, "receipt": None}
    try:
        receipt = json.loads(proc.stdout)
    except json.JSONDecodeError:
        receipt = None
    return {**result, "receipt": receipt}


def require_receipt(record: dict[str, Any], status: str, source_hash: str | None = None) -> bool:
    receipt = record.get("receipt")
    return bool(
        record.get("returncode") == 0
        and isinstance(receipt, dict)
        and receipt.get("status") == status
        and receipt.get("candidate_execution") == "never_imported_or_executed"
        and (source_hash is None or receipt.get("source_sha256") == source_hash)
    )


def write_new(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"append-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)


def make_workspace(root: Path, source: bytes, label: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    (root / "candidate.py").write_bytes(source)
    git(root, "init", "-q", "--initial-branch=main")
    git(root, "config", "user.name", "VARE Sprint VII E0")
    git(root, "config", "user.email", "e0@example.invalid")
    git(root, "config", "core.attributesFile", os.devnull)
    git(root, "add", "--", "candidate.py")
    git(root, "commit", "-q", "-m", f"{label} baseline")
    base_commit = git(root, "rev-parse", "HEAD").stdout.decode().strip()
    base_blob = git(root, "show", "HEAD:candidate.py").stdout
    if sha256_bytes(base_blob) != sha256_bytes(source):
        raise AssertionError("git baseline object does not match source bytes")
    return {"base_commit": base_commit, "base_source_sha256": sha256_bytes(base_blob)}


def grade_case(output_root: Path, name: str, source: bytes) -> dict[str, Any]:
    case_root = output_root / "grader-cases" / name
    case_root.mkdir(parents=True, exist_ok=False)
    candidate = case_root / "candidate.py"
    candidate.write_bytes(source)
    source_hash = sha256_bytes(source)
    return run_grader(case_root, "candidate.py", source_hash)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    output_root = args.output.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise SystemExit(f"refusing to overwrite existing E0 output: {output_root}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_bytes = protocol_path.read_bytes()
    current_hashes = {
        "runner": sha256_file(RUNNER_PATH),
        "grader": sha256_file(GRADER_PATH),
        "function_edit": sha256_file(ROOT / "src/vare/agent_function_edit_v1.py"),
        "transactional_edit": sha256_file(ROOT / "src/vare/agent_edit_v2.py"),
        "agent_runner": sha256_file(ROOT / "scripts/run_local_agent.py"),
        "schema_slots": sha256_file(ROOT / "src/vare/schema_slot_protocol.py"),
        "e0_test": sha256_file(ROOT / "tests/test_sprint_vii_e0.py"),
        "schema_slot_test": sha256_file(ROOT / "tests/test_schema_slot_protocol.py"),
        "function_edit_test": sha256_file(ROOT / "tests/test_agent_function_edit_v2.py"),
    }
    if protocol.get("code_sha256") != current_hashes:
        raise SystemExit("protocol code hashes do not match current E0 implementation")
    if protocol.get("fixture_id") != "vare-sprint-vii-e0-clamp-v1":
        raise SystemExit("unexpected E0 fixture id")

    output_root.mkdir(parents=True, exist_ok=False)
    candidate_root = output_root / "candidate-workspace"
    baseline = b"def clamp(value):\n    return value\n"
    known_fix = b"def clamp(value):\n    return max(0, value)\n"
    wrong_fix = b"def clamp(value):\n    return max(1, value)\n"
    source_events: list[dict[str, Any]] = []
    started = time.perf_counter()
    grader_hash_before = sha256_file(GRADER_PATH)
    identity = make_workspace(candidate_root, baseline, "e0")
    source_path = candidate_root / "candidate.py"
    store = FunctionEditLeaseStoreV2(
        candidate_root, {"candidate.py"}, token_factory=lambda: "e0-lease-0001",
    )
    listing = store.list_functions("candidate.py")
    source_events.append({"tool": "list_functions", "arguments": {"path": "candidate.py"}, "result": listing})
    read = store.read_function("candidate.py", "clamp")
    source_events.append({"tool": "read_function", "arguments": {
        "path": "candidate.py", "qualified_name": "clamp",
    }, "result": read})
    apply_result = store.replace_function_body_lines_transactional_v2(
        read.get("lease"), ["    return max(0, value)"],
    )
    source_events.append({"tool": "replace_function_body_transactional", "arguments": {
        "lease": read.get("lease"), "body_lines": ["    return max(0, value)"],
    }, "result": apply_result})
    edited_bytes = source_path.read_bytes()
    edited_hash = sha256_bytes(edited_bytes)
    diff_proc = git(
        candidate_root, "diff", "--exit-code", "--no-ext-diff", "--no-textconv",
        "--binary", "--", "candidate.py", check=False,
    )
    diff = diff_proc.stdout
    diff_paths = git(candidate_root, "diff", "--name-only", "--", check=False).stdout.decode().splitlines()
    status_lines = git(candidate_root, "status", "--porcelain=v1").stdout.decode().splitlines()
    tracked_path = git(candidate_root, "ls-files", "--error-unmatch", "--", "candidate.py", check=False)
    base_again = git(candidate_root, "show", "HEAD:candidate.py").stdout
    workspace_identity_ok = bool(
        git(candidate_root, "rev-parse", "HEAD").stdout.decode().strip() == identity["base_commit"]
        and tracked_path.returncode == 0
        and diff_paths == ["candidate.py"]
        and status_lines == [" M candidate.py"]
        and sha256_bytes(base_again) == identity["base_source_sha256"]
        and edited_hash != identity["base_source_sha256"]
        and diff
        and b"return max(0, value)" in diff
        and diff_proc.returncode == 1
    )

    baseline_grade = grade_case(output_root, "baseline", baseline)
    known_grade = run_grader(candidate_root, "candidate.py", edited_hash)
    wrong_grade = grade_case(output_root, "wrong-mutant", wrong_fix)
    spoof_source = (
        b"from pathlib import Path\n"
        b"Path(__file__).with_name('spoof-executed').write_text('spoofed')\n"
        b"def clamp(value):\n    return max(0, value)\n"
    )
    spoof_grade = grade_case(output_root, "grader-spoof", spoof_source)
    spoof_marker = output_root / "grader-cases" / "grader-spoof" / "spoof-executed"
    drift_grade = run_grader(candidate_root, "candidate.py", identity["base_source_sha256"])

    controls: dict[str, Any] = {}
    invalid_root = output_root / "invalid-patch-workspace"
    make_workspace(invalid_root, baseline, "invalid-patch")
    invalid_store = FunctionEditLeaseStoreV2(
        invalid_root, {"candidate.py"}, token_factory=lambda: "invalid-lease",
    )
    invalid_read = invalid_store.read_function("candidate.py", "clamp")
    invalid_before = (invalid_root / "candidate.py").read_bytes()
    invalid_patch = invalid_store.replace_function_body_lines_transactional_v2(
        invalid_read.get("lease"), ["    return value\n"],
    )
    controls["invalid_patch"] = {
        "result": invalid_patch,
        "source_unchanged": (invalid_root / "candidate.py").read_bytes() == invalid_before,
    }

    stale_root = output_root / "stale-source-workspace"
    make_workspace(stale_root, baseline, "stale-source")
    stale_store = FunctionEditLeaseStoreV2(
        stale_root, {"candidate.py"}, token_factory=lambda: "stale-lease",
    )
    stale_read = stale_store.read_function("candidate.py", "clamp")
    drifted = baseline + b"# concurrent source change\n"
    (stale_root / "candidate.py").write_bytes(drifted)
    stale_result = stale_store.replace_function_body_lines_transactional_v2(
        stale_read.get("lease"), ["    return max(0, value)"],
    )
    controls["stale_source"] = {
        "result": stale_result,
        "drifted_source_preserved": (stale_root / "candidate.py").read_bytes() == drifted,
    }

    symlink_root = output_root / "symlink-workspace"
    symlink_root.mkdir()
    symlink_target = symlink_root / "target.py"
    symlink_target.write_bytes(baseline)
    (symlink_root / "candidate.py").symlink_to(symlink_target.name)
    symlink_store = FunctionEditLeaseStoreV2(
        symlink_root, {"candidate.py"}, token_factory=lambda: "symlink-lease",
    )
    symlink_result = symlink_store.read_function("candidate.py", "clamp")
    controls["symlink"] = {
        "result": symlink_result,
        "target_unchanged": symlink_target.read_bytes() == baseline,
    }

    cases_ok = {
        "source_listed": listing.get("status") == "functions_listed"
            and [item.get("qualified_name") for item in listing.get("functions", [])] == ["clamp"],
        "source_read": read.get("status") == "read_complete"
            and read.get("body") == "    return value\n"
            and read.get("lease") == "e0-lease-0001",
        "hash_bound_atomic_edit": apply_result.get("applied") is True
            and apply_result.get("before_sha256") == identity["base_source_sha256"]
            and apply_result.get("after_sha256") == edited_hash,
        "source_changed": edited_bytes == known_fix,
        "canonical_diff": workspace_identity_ok,
        "baseline_fails": require_receipt(baseline_grade, "FAIL", sha256_bytes(baseline)),
        "known_fix_passes": require_receipt(known_grade, "PASS", edited_hash),
        "wrong_mutant_fails": require_receipt(wrong_grade, "FAIL", sha256_bytes(wrong_fix)),
        "grader_spoof_rejected": require_receipt(spoof_grade, "REJECT", sha256_bytes(spoof_source))
            and not spoof_marker.exists(),
        "grader_source_drift_rejected": require_receipt(drift_grade, "REJECT"),
        "invalid_patch_rejected_without_write": invalid_patch.get("error") == "invalid_replacement_lines"
            and controls["invalid_patch"]["source_unchanged"],
        "stale_source_rejected_without_overwrite": stale_result.get("error") == "stale_preimage"
            and controls["stale_source"]["drifted_source_preserved"],
        "symlink_rejected_without_target_change": symlink_result.get("status") == "valid_call_rejected_by_policy"
            and controls["symlink"]["target_unchanged"],
        "grader_immutable": grader_hash_before == sha256_file(GRADER_PATH),
    }
    elapsed_seconds = round(time.perf_counter() - started, 6)
    prior_artifact_bytes = sum(
        path.stat().st_size for path in output_root.rglob("*") if path.is_file()
    )
    artifact_budget = 5 * 1024 * 1024
    # Reserve 256 KiB for the final result and manifest before declaring this gate in-budget.
    cases_ok["wall_clock_budget"] = elapsed_seconds <= 60
    cases_ok["artifact_size_budget"] = prior_artifact_bytes + 256 * 1024 <= artifact_budget
    status = "PASS" if all(cases_ok.values()) else "FAIL"
    result = {
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": sha256_bytes(protocol_bytes),
        "fixture_id": protocol["fixture_id"],
        "status": status,
        "cases": cases_ok,
        "task_family_capability_claim": False,
        "model_inference": "NOT_RUN",
        "grader_execution_boundary": "candidate bytes parsed and interpreted under restricted AST; candidate never imported or executed",
        "source_identity": {
            **identity,
            "candidate_source_sha256": edited_hash,
            "canonical_diff_sha256": sha256_bytes(diff),
            "canonical_diff_utf8": diff.decode("utf-8", "strict"),
            "workspace_identity_verified": workspace_identity_ok,
        },
        "tool_events": source_events,
        "grader_outcomes": {
            "baseline": baseline_grade,
            "known_fix": known_grade,
            "wrong_mutant": wrong_grade,
            "grader_spoof": spoof_grade,
            "source_drift": drift_grade,
        },
        "negative_controls": controls,
        "code_sha256": current_hashes,
        "grader_sha256_before_after": {"before": grader_hash_before, "after": sha256_file(GRADER_PATH)},
        "resource": {
            "elapsed_seconds": elapsed_seconds,
            "artifact_bytes_before_receipt": prior_artifact_bytes,
            "final_artifact_bytes": None,
            "artifact_budget_bytes": artifact_budget,
            "receipt_and_manifest_reserve_bytes": 256 * 1024,
            "process_peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "process_peak_rss_units": "bytes on macOS; KiB on Linux",
        },
        "reproduction": [
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q tests/test_sprint_vii_e0.py",
            f"python scripts/run_sprint_vii_e0.py --protocol {protocol_path} --output {output_root}",
        ],
    }
    result_path = output_root / "e0-result.json"
    manifest_path = output_root / "sha256-manifest.json"
    final_artifact_bytes = prior_artifact_bytes
    for _ in range(5):
        result["resource"]["final_artifact_bytes"] = final_artifact_bytes
        result["status"] = status
        result_bytes = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        manifest = {
            "protocol_sha256": sha256_bytes(protocol_bytes),
            "result_sha256": sha256_bytes(result_bytes),
            "files": {
                "protocol": {"path": str(protocol_path), "sha256": sha256_bytes(protocol_bytes)},
                "runner": {"path": str(RUNNER_PATH), "sha256": current_hashes["runner"]},
                "grader": {"path": str(GRADER_PATH), "sha256": current_hashes["grader"]},
                "result": {"path": str(result_path), "sha256": sha256_bytes(result_bytes)},
            },
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        measured = prior_artifact_bytes + len(result_bytes) + len(manifest_bytes)
        if measured > artifact_budget:
            cases_ok["artifact_size_budget"] = False
            status = "FAIL"
            result["cases"] = cases_ok
        if measured == final_artifact_bytes:
            break
        final_artifact_bytes = measured
    else:
        raise RuntimeError("could not converge final E0 artifact size")
    if final_artifact_bytes > artifact_budget:
        raise RuntimeError("E0 artifacts exceeded the frozen size budget")
    write_new(result_path, result_bytes)
    write_new(manifest_path, manifest_bytes)
    print(json.dumps({"status": status, "output": str(output_root), "cases": cases_ok}, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
