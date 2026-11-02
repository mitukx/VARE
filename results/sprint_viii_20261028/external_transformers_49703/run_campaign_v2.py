from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[3]
TASK_BASE = Path(__file__).resolve().parent
RUNNER = ROOT / "scripts/run_local_agent.py"
GRADER = ROOT / "scripts/sprint_viii_transformers_49703_grader_v1.py"
RUNTIME_AUDITOR = ROOT / "scripts/audit_agent_runtime_snapshot_v1.py"
TASK_ROOT = TASK_BASE / "task-v2"
TEMPLATE = TASK_BASE / "candidate-template"
LOCALIZATION = TASK_BASE / "localization/localization-v2.json"
SOURCE_PATH = "src/transformers/models/mask2former/modeling_mask2former.py"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while retaining campaign evidence")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def append_exposure_event(output_root: Path, event: dict[str, Any]) -> None:
    """Durably record that this Development task was sent to the model."""
    path = output_root / "task-exposure.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        payload = (json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n").encode()
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while retaining task exposure")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def resource_snapshot(directory: Path, pid: int | None = None) -> dict[str, int | str]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    rss = 0
    if pid is not None:
        try:
            parent = psutil.Process(pid)
            rss = sum(child.memory_info().rss for child in [parent, *parent.children(recursive=True)]
                      if child.is_running())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            rss = 0
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "available_memory_bytes": memory.available,
        "memory_total_bytes": memory.total,
        "swap_used_bytes": swap.used,
        "swap_free_bytes": max(0, swap.total - swap.used),
        "swap_total_bytes": swap.total,
        "disk_free_bytes": shutil.disk_usage(directory).free,
        "process_tree_rss_bytes": rss,
    }


def stop_process_tree(process: subprocess.Popen[bytes]) -> None:
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        process.terminate()
        _gone, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_grader(workspace: Path, attempt_dir: Path) -> dict[str, Any]:
    command = [sys.executable, "-I", str(GRADER), "--workspace-root", str(workspace),
               "--source", SOURCE_PATH]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            timeout=30, check=False)
    write_new(attempt_dir / "grader.stdout.log", result.stdout.encode())
    write_new(attempt_dir / "grader.stderr.log", result.stderr.encode())
    try:
        receipt = json.loads(result.stdout) if result.returncode == 0 else None
    except json.JSONDecodeError:
        receipt = None
    return {
        "argv": command,
        "returncode": result.returncode,
        "stdout_sha256": sha256_bytes(result.stdout.encode()),
        "stderr_sha256": sha256_bytes(result.stderr.encode()),
        "grader_sha256": sha256_file(GRADER),
        "receipt": receipt,
    }


def load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    if not path.is_file():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = row.get("event")
        if isinstance(event, dict):
            events.append(event)
    return events


def run_attempt(*, arm: dict[str, Any], protocol: dict[str, Any], protocol_sha: str,
                attempt_dir: Path, model_dir: Path, wrapper_sha: str) -> dict[str, Any]:
    attempt_dir.mkdir(parents=True, exist_ok=False)
    workspace = attempt_dir / "workspace"
    shutil.copytree(TEMPLATE, workspace)
    source = workspace / SOURCE_PATH
    baseline_sha = protocol["task"]["baseline_source_sha256"]
    if source.is_symlink() or not source.is_file() or sha256_file(source) != baseline_sha:
        raise RuntimeError("candidate copy does not match the pinned upstream baseline")
    tracked = subprocess.run(["git", "-C", str(workspace), "ls-tree", "-r", "--name-only", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.splitlines()
    if tracked != [SOURCE_PATH]:
        raise RuntimeError(f"candidate git baseline has unexpected tracked files: {tracked}")

    budget = protocol["inference_budget"]
    limits = protocol["resource_limits"]
    before = resource_snapshot(attempt_dir)
    failures = [
        name for name, value, minimum in (
            ("available_memory_below_minimum", "available_memory_bytes", limits["minimum_available_memory_bytes"]),
            ("free_swap_below_minimum", "swap_free_bytes", limits["minimum_free_swap_bytes"]),
            ("disk_free_below_minimum", "disk_free_bytes", limits["minimum_disk_free_bytes"]),
        ) if before[value] < minimum
    ]
    preflight = {"protocol_sha256": protocol_sha, "arm_id": arm["arm_id"],
                 "resource_snapshot": before, "failures": failures,
                 "model_inference_started": False}
    write_new(attempt_dir / "preflight.json", (json.dumps(preflight, indent=2, sort_keys=True) + "\n").encode())
    if failures:
        result = {"status": "NOT_STARTED_RESOURCE_PREFLIGHT", "arm": arm,
                  "resource_before": before, "resource_failures": failures,
                  "protocol_sha256": protocol_sha}
        write_new(attempt_dir / "attempt-result.json",
                  (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        return result

    args = [
        sys.executable, str(RUNNER), "--workspace", str(workspace),
        "--model-path", str(model_dir), "--model-repository", protocol["model"]["repository"],
        "--backend", "mlx", "--mlx-device", "metal", "--task-root", str(TASK_ROOT),
        "--diff-base", "HEAD", "--output", str(attempt_dir / "agent.json"),
        "--event-journal", str(attempt_dir / "events.jsonl"),
        "--instruction", protocol["user_instruction"],
        "--seed", str(budget["seed"]), "--max-turns", str(budget["max_turns"]),
        "--max-no-tool-retries", str(budget["max_no_tool_retries"]),
        "--max-new-tokens-per-turn", str(budget["max_new_tokens_per_turn"]),
        "--max-total-new-tokens", str(budget["max_total_new_tokens"]),
        "--max-context-tokens", str(budget["max_context_tokens"]),
        "--wall-seconds", str(budget["agent_wall_seconds"]),
        "--temperature", str(budget["temperature"]), "--top-p", str(budget["top_p"]),
        "--top-k", str(budget["top_k"]), "--no-do-sample", "--edit-format", "structured_replace",
        "--tool-interface", arm["tool_interface"], "--generation-protocol", budget["generation_protocol"],
        "--navigation-guidance", budget["navigation_guidance"], "--require-source-read-before-edit",
        "--post-edit-navigation-gate", "--post-edit-completion-guidance",
        "--grader-entrypoint", str(GRADER),
    ]
    if arm.get("localization_record"):
        args.extend(["--localization-record", str(TASK_BASE / arm["localization_record"])])
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": "0"})
    safe_env = {key: env[key] for key in (
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "PYTHONHASHSEED")}
    write_new(attempt_dir / "command.json", (json.dumps({
        "argv": args, "cwd": str(ROOT), "environment": safe_env,
        "instruction": protocol["user_instruction"], "arm_id": arm["arm_id"],
        "wrapper_source_sha256": wrapper_sha,
    }, indent=2, sort_keys=True) + "\n").encode())

    sample_path = attempt_dir / "resource-samples.jsonl"
    started = time.monotonic()
    peak_rss = 0
    abort_reason = None
    with (attempt_dir / "stdout.log").open("xb") as stdout, \
            (attempt_dir / "stderr.log").open("xb") as stderr, sample_path.open("xb") as samples:
        append_exposure_event(attempt_dir.parent.parent, {
            "event": "development_task_sent_to_model_process",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "task_id": protocol["task"]["task_id"],
            "task_family": protocol["task"]["task_family"],
            "arm_id": arm["arm_id"],
            "protocol_sha256": protocol_sha,
            "prompt_sha256": protocol["task"]["prompt_sha256"],
            "heldout": False,
        })
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        while process.poll() is None:
            observed = resource_snapshot(attempt_dir, process.pid)
            peak_rss = max(peak_rss, int(observed["process_tree_rss_bytes"]))
            samples.write((json.dumps(observed, sort_keys=True) + "\n").encode())
            samples.flush()
            os.fsync(samples.fileno())
            if observed["available_memory_bytes"] < limits["minimum_available_memory_bytes"]:
                abort_reason = "available_memory_below_abort_threshold"
            elif observed["swap_free_bytes"] < limits["minimum_free_swap_bytes"]:
                abort_reason = "free_swap_below_abort_threshold"
            elif int(observed["swap_used_bytes"]) - int(before["swap_used_bytes"]) > limits["max_swap_growth_bytes"]:
                abort_reason = "swap_growth_above_frozen_ceiling"
            elif observed["disk_free_bytes"] < limits["minimum_disk_free_bytes"]:
                abort_reason = "disk_free_below_abort_threshold"
            elif int(observed["process_tree_rss_bytes"]) > limits["max_model_process_rss_bytes"]:
                abort_reason = "model_process_rss_above_abort_threshold"
            elif time.monotonic() - started > limits["attempt_wall_seconds"]:
                abort_reason = "attempt_wall_budget_exhausted"
            if abort_reason:
                stop_process_tree(process)
                break
            time.sleep(limits["sample_interval_seconds"])
        exit_code = process.wait()
    after = resource_snapshot(attempt_dir)

    agent_path = attempt_dir / "agent.json"
    try:
        agent = json.loads(agent_path.read_text(encoding="utf-8")) if agent_path.is_file() else None
    except json.JSONDecodeError:
        agent = None
    changed_paths = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--name-only", "--no-ext-diff", "--no-textconv", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    patch = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD"],
        check=True, capture_output=True,
    ).stdout
    write_new(attempt_dir / "candidate.patch", patch)
    candidate_sha = sha256_file(source) if source.is_file() and not source.is_symlink() else None
    grader = run_grader(workspace, attempt_dir)
    runtime_audit = subprocess.run(
        [sys.executable, "-I", str(RUNTIME_AUDITOR), str(attempt_dir / "runtime_snapshot")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    try:
        runtime_audit_record = json.loads(runtime_audit.stdout) if runtime_audit.returncode == 0 else None
    except json.JSONDecodeError:
        runtime_audit_record = None
    events = load_events(attempt_dir / "events.jsonl")
    tool_rows = [event for event in events if event.get("event_type") == "tool_result"]
    tool_names = [event.get("name") for event in tool_rows]
    search_queries = [event.get("arguments", {}).get("query") for event in tool_rows
                      if event.get("name") == "search_source"]
    useful_reads = [event for event in tool_rows
                    if event.get("name") == "read_lines" and event.get("result", {}).get("status") == "read_complete"]
    expected_controller_hashes = protocol["controller_source_files_sha256_by_interface"][arm["tool_interface"]]
    observed_controller_hashes = agent.get("controller_source_files_sha256") if isinstance(agent, dict) else None
    controller_identity_match = observed_controller_hashes == expected_controller_hashes
    scope_ok = changed_paths == [SOURCE_PATH]
    grade_status = grader.get("receipt", {}).get("status") if isinstance(grader.get("receipt"), dict) else None
    verified_repair = bool(
        abort_reason is None and exit_code == 0 and agent is not None
        and agent.get("model_explicitly_finished") is True
        and agent.get("edit_applied") is True and scope_ok
        and grade_status == "PASS" and runtime_audit.returncode == 0 and controller_identity_match
    )
    record = {
        "status": "RESOURCE_ABORTED" if abort_reason else ("COMPLETED" if exit_code == 0 else "RUNNER_FAILED"),
        "protocol_sha256": protocol_sha, "arm": arm, "exit_code": exit_code,
        "abort_reason": abort_reason, "wall_seconds": round(time.monotonic() - started, 3),
        "peak_process_tree_rss_bytes": peak_rss, "resource_before": before, "resource_after": after,
        "agent_record_sha256": sha256_file(agent_path) if agent_path.is_file() else None,
        "agent_stop_reason": agent.get("stop_reason") if isinstance(agent, dict) else None,
        "agent_outcome": agent.get("agent_outcome") if isinstance(agent, dict) else None,
        "model_explicitly_finished": agent.get("model_explicitly_finished") if isinstance(agent, dict) else None,
        "edit_attempted": agent.get("edit_attempted") if isinstance(agent, dict) else None,
        "edit_applied": agent.get("edit_applied") if isinstance(agent, dict) else None,
        "candidate_changed": candidate_sha != baseline_sha, "candidate_source_sha256": candidate_sha,
        "candidate_patch_sha256": sha256_bytes(patch), "candidate_patch_bytes": len(patch),
        "candidate_changed_paths": changed_paths, "candidate_scope_ok": scope_ok,
        "tool_call_count": len(tool_names), "tool_names": tool_names,
        "source_search_count": len(search_queries), "unique_source_search_queries": sorted(set(search_queries)),
        "valid_source_read_count": len(useful_reads),
        "generated_tokens": agent.get("new_tokens") if isinstance(agent, dict) else None,
        "public_test_runs": agent.get("public_test_runs") if isinstance(agent, dict) else None,
        "public_test_result": agent.get("public_test_result") if isinstance(agent, dict) else None,
        "controller_interventions": agent.get("controller_interventions") if isinstance(agent, dict) else None,
        "controller_source_hashes_match_protocol": controller_identity_match,
        "independent_grade": grader,
        "runtime_snapshot_audit": runtime_audit_record,
        "runtime_snapshot_audit_exit_code": runtime_audit.returncode,
        "verified_repair": verified_repair,
    }
    record_bytes = (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    write_new(attempt_dir / "attempt-result.json", record_bytes)
    raw_hashes = {
        path.relative_to(attempt_dir).as_posix(): sha256_file(path)
        for path in sorted(attempt_dir.rglob("*"))
        if path.is_file() and ".git" not in path.parts and path.name not in {"sha256-manifest.json"}
    }
    write_new(attempt_dir / "sha256-manifest.json", (json.dumps({
        "attempt_result_sha256": sha256_bytes(record_bytes), "protocol_sha256": protocol_sha,
        "raw_artifact_sha256": raw_hashes,
    }, indent=2, sort_keys=True) + "\n").encode())
    return record


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_campaign_v1.py PROTOCOL.json OUTPUT_DIRECTORY")
    protocol_path = Path(sys.argv[1]).resolve(strict=True)
    output_root = Path(sys.argv[2]).absolute()
    payload = protocol_path.read_bytes()
    protocol_sha = sha256_bytes(payload)
    sidecar = protocol_path.with_suffix(".sha256")
    if sidecar.read_text(encoding="ascii").split()[0] != protocol_sha:
        raise SystemExit("frozen campaign protocol checksum mismatch")
    protocol = json.loads(payload)
    if protocol.get("protocol_id") != "vare.sprint-viii.transformers-49703-development.v2":
        raise SystemExit("unsupported external campaign protocol")
    if protocol.get("frozen_before_model_inference") is not True:
        raise SystemExit("protocol is not frozen before inference")
    if output_root.exists() or output_root.is_symlink():
        raise SystemExit(f"refusing to overwrite prior campaign evidence: {output_root}")
    if sha256_file(Path(__file__)) != protocol["code_sha256"]["campaign_driver"]:
        raise SystemExit("campaign driver source changed after protocol freeze")
    for key, relative in (("runner", "scripts/run_local_agent.py"),
                          ("grader", "scripts/sprint_viii_transformers_49703_grader_v1.py"),
                          ("runtime_auditor", "scripts/audit_agent_runtime_snapshot_v1.py"),
                          ("runtime_snapshot_module", "src/vare/agent_runtime_snapshot_v1.py"),
                          ("task_contract_module", "src/vare/agent_task_contract.py")):
        if sha256_file(ROOT / relative) != protocol["code_sha256"][key]:
            raise SystemExit(f"runtime source changed after protocol freeze: {relative}")
    if sys.executable != protocol["runtime"]["python_executable"]:
        raise SystemExit("Python executable differs from the frozen protocol")
    model_dir = Path(protocol["model"]["snapshot_path"])
    model_manifest = {path.name: sha256_file(path) for path in sorted(model_dir.iterdir()) if path.is_file()}
    if model_manifest != protocol["model"]["files_sha256"]:
        raise SystemExit("cached model snapshot differs from the frozen model manifest")
    task_path = TASK_ROOT / "task.json"
    if sha256_file(task_path) != protocol["task"]["task_json_sha256"]:
        raise SystemExit("Development task changed after protocol freeze")
    if sha256_file(TEMPLATE / SOURCE_PATH) != protocol["task"]["baseline_source_sha256"]:
        raise SystemExit("candidate template changed after protocol freeze")
    if sha256_file(LOCALIZATION) != protocol["arms"][1]["localization_record_sha256"]:
        raise SystemExit("localized source record changed after protocol freeze")
    if set(protocol["controller_source_files_sha256_by_interface"]) != {
            arm["tool_interface"] for arm in protocol["arms"]}:
        raise SystemExit("protocol lacks exact controller source hashes for an arm")

    output_root.mkdir(parents=True, exist_ok=False)
    initial = resource_snapshot(output_root)
    limits = protocol["resource_limits"]
    failures = [name for name, value, minimum in (
        ("available_memory_below_minimum", "available_memory_bytes", limits["minimum_available_memory_bytes"]),
        ("free_swap_below_minimum", "swap_free_bytes", limits["minimum_free_swap_bytes"]),
        ("disk_free_below_minimum", "disk_free_bytes", limits["minimum_disk_free_bytes"]),
    ) if initial[value] < minimum]
    preflight = {"protocol_sha256": protocol_sha, "timestamp_utc": initial["timestamp_utc"],
                 "model_snapshot_verified": True, "task_contract_verified": True,
                 "candidate_baseline_verified": True, "localization_record_verified": True,
                 "resource_snapshot": initial, "failures": failures,
                 "model_inference_started": False}
    write_new(output_root / "preflight.json", (json.dumps(preflight, indent=2, sort_keys=True) + "\n").encode())
    if failures:
        summary = {"status": "NOT_STARTED_RESOURCE_PREFLIGHT", "protocol_sha256": protocol_sha,
                   "preflight": preflight, "task_family_denominator": 1, "arms": []}
        write_new(output_root / "campaign-summary.json", (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
        print(json.dumps({"status": summary["status"], "failures": failures}))
        return 2

    results = []
    for index, arm in enumerate(protocol["arms"]):
        result = run_attempt(
            arm=arm, protocol=protocol, protocol_sha=protocol_sha,
            attempt_dir=output_root / "attempts" / arm["arm_id"],
            model_dir=model_dir, wrapper_sha=protocol["code_sha256"]["campaign_driver"],
        )
        results.append(result)
        if result["status"] in {"RESOURCE_ABORTED", "NOT_STARTED_RESOURCE_PREFLIGHT"}:
            for remaining in protocol["arms"][index + 1:]:
                remaining_dir = output_root / "attempts" / remaining["arm_id"]
                remaining_dir.mkdir(parents=True, exist_ok=False)
                skipped = {"status": "NOT_STARTED_AFTER_RESOURCE_STOP", "arm": remaining,
                           "protocol_sha256": protocol_sha}
                write_new(remaining_dir / "attempt-result.json",
                          (json.dumps(skipped, indent=2, sort_keys=True) + "\n").encode())
                results.append(skipped)
            break
    summary = {
        "status": "COMPLETED" if len(results) == len(protocol["arms"])
        and all(not row["status"].startswith("NOT_STARTED") for row in results) else "PARTIAL",
        "protocol_sha256": protocol_sha,
        "task_family_denominator": 1,
        "claim_limit": protocol["claim_limit"],
        "arms": protocol["arms"],
        "results": results,
        "verified_repair_by_arm": {row["arm"]["arm_id"]: row.get("verified_repair")
                                    for row in results if isinstance(row.get("arm"), dict)},
        "independent_grade_by_arm": {row["arm"]["arm_id"]:
                                     row.get("independent_grade", {}).get("receipt", {}).get("status")
                                     if isinstance(row.get("independent_grade"), dict) else None
                                     for row in results if isinstance(row.get("arm"), dict)},
        "model_explicit_finish_by_arm": {row["arm"]["arm_id"]: row.get("model_explicitly_finished")
                                         for row in results if isinstance(row.get("arm"), dict)},
        "resource_after_campaign": resource_snapshot(output_root),
        "evidence_directory": str(output_root),
    }
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    write_new(output_root / "campaign-summary.json", summary_bytes)
    write_new(output_root / "sha256-manifest.json", (json.dumps({
        "protocol_sha256": protocol_sha,
        "campaign_summary_sha256": sha256_bytes(summary_bytes),
        "attempt_result_sha256": {
            row["arm"]["arm_id"]: sha256_file(output_root / "attempts" / row["arm"]["arm_id"] / "attempt-result.json")
            for row in results if isinstance(row.get("arm"), dict)
            and (output_root / "attempts" / row["arm"]["arm_id"] / "attempt-result.json").is_file()
        },
    }, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"status": summary["status"], "verified_repair_by_arm": summary["verified_repair_by_arm"],
                      "independent_grade_by_arm": summary["independent_grade_by_arm"],
                      "model_explicit_finish_by_arm": summary["model_explicit_finish_by_arm"],
                      "evidence_directory": str(output_root)}, sort_keys=True))
    return 0 if summary["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
