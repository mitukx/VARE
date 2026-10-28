#!/usr/bin/env python3
"""Run the frozen paired original/corrected-template Development campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def durable(path: Path, value: dict) -> None:
    encoded = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def git(workspace: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(workspace), *args], check=True,
                          capture_output=True, text=True, timeout=15).stdout.strip()


def validate_hashed_files(root: Path, files: dict[str, str], label: str) -> None:
    for relative, expected in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe {label} path: {relative}")
        path = root / relative_path
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"{label} drift: {relative}")


def validate_protocol(protocol_path: Path, expected_sha: str) -> dict:
    protocol_bytes = protocol_path.read_bytes()
    if hashlib.sha256(protocol_bytes).hexdigest() != expected_sha:
        raise ValueError("operator-supplied campaign protocol hash mismatch")
    protocol = json.loads(protocol_bytes)
    validate_hashed_files(ROOT, protocol["source_hashes"], "campaign implementation")
    validate_hashed_files(ROOT, protocol["asset_hashes"], "campaign asset")
    validate_hashed_files(ROOT, protocol.get("qualification_evidence_hashes", {}),
                          "task qualification evidence")
    return protocol


def event_chain(path: Path) -> tuple[list[dict], str]:
    previous = "0" * 64
    events = []
    for sequence, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        event = record["event"]
        canonical = json.dumps(event, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode()
        digest = hashlib.sha256(canonical).hexdigest()
        if (event.get("sequence") != sequence
                or record.get("previous_event_sha256") != previous
                or record.get("event_sha256") != digest):
            raise ValueError("invalid model event journal chain")
        events.append(event)
        previous = digest
    return events, previous


def monitored(command: list[str], directory: Path, *, wall_s: int,
              rss_limit: int, swap_growth_limit: int, env: dict[str, str]) -> dict:
    start = time.monotonic()
    swap_start = psutil.swap_memory().used
    samples = []
    abort = None
    with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                   cwd=ROOT, start_new_session=True)
        while process.poll() is None:
            elapsed = time.monotonic() - start
            try:
                parent = psutil.Process(process.pid)
                rss = sum(item.memory_info().rss for item in [parent, *parent.children(recursive=True)]
                          if item.is_running())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                rss = 0
            swap_used = psutil.swap_memory().used
            samples.append({"elapsed_s": round(elapsed, 3), "rss_bytes": rss,
                            "swap_used_bytes": swap_used,
                            "available_memory_bytes": psutil.virtual_memory().available})
            if elapsed > wall_s:
                abort = "wall_limit"
            elif rss > rss_limit:
                abort = "rss_limit"
            elif swap_used - swap_start > swap_growth_limit:
                abort = "host_swap_growth_limit"
            if abort:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(0.25)
        returncode = process.wait()
    result = {"command": command, "returncode": returncode,
              "elapsed_s": round(time.monotonic() - start, 3), "abort": abort,
              "samples": samples,
              "peak_sampled_rss_bytes": max((row["rss_bytes"] for row in samples), default=0),
              "swap_used_start_bytes": swap_start,
              "swap_used_end_bytes": psutil.swap_memory().used}
    durable(directory / "process.json", result)
    return result


def make_workspace(cell_dir: Path, task: dict, asset_root: Path) -> tuple[Path, Path, str]:
    workspace = cell_dir / "workspace"
    workspace.mkdir()
    candidate_source = workspace / task["source_path"]
    candidate_source.parent.mkdir(parents=True, exist_ok=True)
    candidate_source.write_bytes((asset_root / task["candidate_projection_asset"]).read_bytes())
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_NO_REPLACE_OBJECTS": "1"}
    for command in (["init", "-q"], ["add", task["source_path"]],
                    ["-c", "user.name=VARE experiment", "-c", "user.email=vare@localhost",
                     "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Pinned candidate projection"]):
        subprocess.run(["git", "-C", str(workspace), *command], check=True,
                       env=env, capture_output=True, timeout=15)
    base_revision = git(workspace, "rev-parse", "HEAD")
    task_root = cell_dir / "task"
    task_root.mkdir()
    task_record = {
        "task_id": task["task_id"],
        "prompt": task["task_prompt"],
        "source": {"base_revision": base_revision},
        "metadata": {"split": "development", "editable_files": [task["source_path"]],
                     "upstream_repository": task.get("repository", task.get("source_pack")),
                     "upstream_revision": task.get("base_revision")},
    }
    durable(task_root / "task.json", task_record)
    return workspace, task_root, base_revision


def preflight(protocol: dict) -> dict:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    result = {"available_memory_bytes": memory.available, "free_swap_bytes": swap.free,
              "minimum_available_memory_bytes": protocol["resources"]["minimum_available_memory_bytes"],
              "minimum_free_swap_bytes": protocol["resources"]["minimum_free_swap_bytes"],
              "passed": memory.available >= protocol["resources"]["minimum_available_memory_bytes"]
              and swap.free >= protocol["resources"]["minimum_free_swap_bytes"]}
    return result


def is_verified_repair(*, model_identity_verified: bool, normal_completion: bool,
                       agent: dict | None, source_changed: bool,
                       independent_grade_valid: bool, grade: dict | None) -> bool:
    """Count a repair only after every provenance and grading gate passes."""
    return bool(
        model_identity_verified and normal_completion and isinstance(agent, dict)
        and agent.get("stop_reason") == "agent_finished" and source_changed
        and independent_grade_valid and isinstance(grade, dict)
        and grade.get("success") is True
        and (agent.get("public_test_result") or {}).get("passed") is True
    )


def run_cell(output: Path, cell: dict, protocol: dict, asset_root: Path) -> dict:
    cell_dir = output / cell["id"]
    cell_dir.mkdir()
    gate = preflight(protocol)
    durable(cell_dir / "preflight.json", gate)
    if not gate["passed"]:
        result = {"cell_id": cell["id"], "status": "PREFLIGHT_REJECTED",
                  "model_invocations": 0, "preflight": gate}
        durable(cell_dir / "result.json", result)
        return result

    task = next(item for item in protocol["tasks"] if item["id"] == cell["task_id"])
    workspace, task_root, base_revision = make_workspace(cell_dir, task, asset_root)
    candidate = workspace / task["source_path"]
    instruction = task["instruction"]
    command = [protocol["model"]["python"], str(ROOT / protocol["model"]["entrypoint"]),
               "--workspace", str(workspace), "--model-path", protocol["model"]["path"],
               "--model-repository", protocol["model"]["repository"],
               "--backend", "mlx", "--mlx-device", "cpu",
               "--task-root", str(task_root), "--output", str(cell_dir / "agent-record.json"),
               "--event-journal", str(cell_dir / "events.jsonl"),
               "--instruction", instruction, "--seed", str(cell["seed"]),
               "--max-turns", str(protocol["budgets"]["max_turns"]),
               "--max-no-tool-retries", str(protocol["budgets"]["max_no_tool_retries"]),
               "--max-new-tokens-per-turn", str(protocol["budgets"]["max_new_tokens_per_turn"]),
               "--max-total-new-tokens", str(protocol["budgets"]["max_total_new_tokens"]),
               "--max-context-tokens", str(protocol["budgets"]["max_context_tokens"]),
               "--wall-seconds", str(protocol["budgets"]["internal_wall_seconds"]),
               "--no-do-sample", "--temperature", "0", "--top-p", "0", "--top-k", "0",
               "--edit-format", "structured_replace", "--tool-interface", "structured_replace_v1",
               "--require-split", "development",
               "--public-contract-grader", str(ROOT / protocol["grader"]),
               "--public-contract-task", task["id"],
               "--max-public-test-runs", str(protocol["budgets"]["max_public_test_runs"])]
    if cell["template_arm"] == "corrected":
        command += ["--tokenizer-config-override", str(ROOT / protocol["model"]["corrected_tokenizer_config"])]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_NO_REPLACE_OBJECTS": "1"}
    process = monitored(command, cell_dir, wall_s=protocol["resources"]["external_wall_seconds"],
                        rss_limit=protocol["resources"]["rss_limit_bytes"],
                        swap_growth_limit=protocol["resources"]["swap_growth_limit_bytes"], env=env)
    model_record_path = cell_dir / "agent-record.json"
    events_path = cell_dir / "events.jsonl"
    integrity_errors = []
    try:
        agent = json.loads(model_record_path.read_text()) if model_record_path.is_file() else None
    except (OSError, json.JSONDecodeError) as error:
        agent = None
        integrity_errors.append(f"agent record is unreadable: {type(error).__name__}")
    try:
        events, final_event_hash = event_chain(events_path) if events_path.is_file() else ([], None)
        if not events_path.is_file():
            integrity_errors.append("required event journal is missing")
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
        events, final_event_hash = [], None
        integrity_errors.append(f"event journal audit failed: {type(error).__name__}: {error}")
    invocations = sum(event.get("event_type") == "invocation_started" for event in events)
    if isinstance(agent, dict) and invocations != 1:
        integrity_errors.append(f"expected one model invocation, observed {invocations}")
    generations = [event for event in events if event.get("event_type") == "model_output_completed"]
    for event in generations:
        prompt = event.get("rendered_prompt")
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if isinstance(prompt, str) else None
        if prompt_hash is None or prompt_hash != event.get("rendered_prompt_sha256"):
            integrity_errors.append(f"rendered prompt identity mismatch at generation {event.get('turn')}")
    if isinstance(agent, dict):
        transcript = agent.get("transcript", [])
        transcript_outputs = [turn.get("model_output") for turn in transcript if "model_output" in turn]
        event_outputs = [event.get("model_output") for event in generations]
        transcript_hashes = [turn.get("rendered_prompt_sha256") for turn in transcript
                             if "model_output" in turn]
        event_hashes = [event.get("rendered_prompt_sha256") for event in generations]
        if transcript_outputs != event_outputs or transcript_hashes != event_hashes:
            integrity_errors.append("model output or rendered-prompt journal differs from transcript")

    grade_dir = cell_dir / "grading"
    grade_dir.mkdir()
    grade_process = monitored(
        [protocol["driver_python"], "-I", str(ROOT / protocol["grader"]), "--task", task["id"],
         "--source", str(candidate)],
        grade_dir, wall_s=protocol["resources"]["grader_wall_seconds"],
        rss_limit=protocol["resources"]["grader_rss_limit_bytes"],
        swap_growth_limit=protocol["resources"]["grader_swap_growth_limit_bytes"], env=env,
    )
    try:
        grade = json.loads((grade_dir / "stdout.log").read_text()) if grade_process["returncode"] == 0 else None
    except json.JSONDecodeError:
        grade = None

    arm = cell["template_arm"]
    expected_template = protocol["model"]["original_template_sha256"] if arm == "original" else protocol["model"]["corrected_template_sha256"]
    source_changed = sha(candidate) != task["candidate_projection_sha256"]
    independent_grade_valid = (isinstance(grade, dict) and grade.get("task") == task["id"]
                               and grade.get("source_sha256") == sha(candidate)
                               and grade.get("admitted") is True)
    normal_completion = (process["abort"] is None and process["returncode"] == 0
                         and isinstance(agent, dict) and not integrity_errors)
    tokenizer_identity = agent.get("tokenizer_configuration", {}) if isinstance(agent, dict) else {}
    expected_override_hash = (protocol["model"]["corrected_tokenizer_config_sha256"]
                              if arm == "corrected" else None)
    identity_ok = (isinstance(agent, dict)
                   and agent.get("model_files") == protocol["model"]["model_files_sha256"]
                   and tokenizer_identity.get("original_config_sha256")
                   == protocol["model"]["original_tokenizer_config_sha256"]
                   and tokenizer_identity.get("override_config_sha256") == expected_override_hash
                   and tokenizer_identity.get("effective_chat_template_sha256") == expected_template)
    if isinstance(agent, dict) and not identity_ok:
        integrity_errors.append("model/tokenizer identity mismatch")
        normal_completion = False
    completed_repair = is_verified_repair(
        model_identity_verified=identity_ok, normal_completion=normal_completion,
        agent=agent, source_changed=source_changed,
        independent_grade_valid=independent_grade_valid, grade=grade,
    )
    transcript = agent.get("transcript", []) if isinstance(agent, dict) else []
    tool_calls = [call for turn in transcript for call in turn.get("tool_calls", [])]
    statuses = [turn.get("tool_output_status") for turn in transcript if turn.get("tool_output_status")]
    counts = {
        "valid_json_tool_calls": sum(status == "complete_tool_call" for status in statuses),
        "invalid_json_or_schema": sum(status == "invalid_json_or_schema" for status in statuses),
        "truncated_tool_calls": sum(status == "truncated_tool_call" for status in statuses),
        "list_files_calls": sum(call.get("name") == "list_files" for call in tool_calls),
        "source_search_calls": sum(call.get("name") == "search_text" for call in tool_calls),
        "source_read_calls": sum(call.get("name") == "read_file_range" for call in tool_calls),
        "edit_tool_calls": sum(call.get("name") == "replace_text" for call in tool_calls),
        "public_test_calls": sum(call.get("name") == "run_public_tests" for call in tool_calls),
        "edit_attempted": bool(agent and agent.get("edit_attempted")),
        "edit_applied": bool(agent and agent.get("edit_applied")),
        "public_test_passed": bool(agent and (agent.get("public_test_result") or {}).get("passed")),
        "source_changed": source_changed,
        "independent_grade_success": bool(independent_grade_valid and grade.get("success")),
        "completed_verified_repair": completed_repair,
    }
    status = "PROVENANCE_FAILURE" if integrity_errors else (
        "GRADED" if (normal_completion and grade_process["abort"] is None
                           and grade_process["returncode"] == 0 and isinstance(grade, dict)) else (
        "RESOURCE_ABORT" if process["abort"] is not None else (
            "UN_GRADED" if normal_completion else "INFERENCE_FAILED")))
    result = {
        "cell_id": cell["id"], "task_id": task["id"], "template_arm": arm,
        "seed": cell["seed"], "status": status,
        "preflight": gate, "process": process,
        "model_invocations": invocations, "generation_count": len(generations),
        "event_chain_final_sha256": final_event_hash,
        "base_revision": base_revision,
        "candidate_source_sha256": sha(candidate),
        "agent_record_sha256": sha(model_record_path) if model_record_path.is_file() else None,
        "event_journal_sha256": sha(events_path) if events_path.is_file() else None,
        "grade_process": grade_process, "grade": grade,
        "model_tokenizer_identity_verified": identity_ok,
        "integrity_errors": integrity_errors,
        "metrics": counts,
        "agent_stop_reason": agent.get("stop_reason") if isinstance(agent, dict) else None,
        "new_tokens": agent.get("new_tokens") if isinstance(agent, dict) else None,
        "elapsed_seconds": agent.get("elapsed_seconds") if isinstance(agent, dict) else None,
        "max_rss_bytes": agent.get("max_rss_bytes") if isinstance(agent, dict) else None,
        "public_test_result": agent.get("public_test_result") if isinstance(agent, dict) else None,
    }
    durable(cell_dir / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = validate_protocol(args.protocol.resolve(), args.protocol_sha256)
    for filename, expected in protocol["model"]["model_files_sha256"].items():
        if sha(Path(protocol["model"]["path"]) / filename) != expected:
            raise ValueError(f"model file drift: {filename}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    durable(output / "protocol.json", protocol)
    rows = []
    for cell in protocol["cells"]:
        print(f"START {cell['id']}", flush=True)
        result = run_cell(output, cell, protocol, ROOT / protocol["asset_root"])
        rows.append(result)
        print(json.dumps({"cell_id": result["cell_id"], "status": result["status"],
                          "metrics": result.get("metrics", {}),
                          "agent_stop_reason": result.get("agent_stop_reason")}, sort_keys=True), flush=True)
        if result["status"] in {"PREFLIGHT_REJECTED", "RESOURCE_ABORT", "INFERENCE_FAILED",
                                 "UN_GRADED", "PROVENANCE_FAILURE"}:
            break
    complete_cells = len(rows) == len(protocol["cells"])
    summary = {
        "protocol_sha256": args.protocol_sha256,
        "cells": [{"cell_id": row["cell_id"], "status": row["status"],
                   "task_id": row.get("task_id"), "template_arm": row.get("template_arm"),
                   "metrics": row.get("metrics", {})} for row in rows],
        "campaign_complete": complete_cells and all(row["status"] == "GRADED" for row in rows),
        "verified_repairs": sum(row.get("metrics", {}).get("completed_verified_repair", False) for row in rows),
        "training": "NO_GO_SINGLE_GREEDY_ATTEMPT_PER_TASK_AND_ARM",
        "heldout_evaluation": "NOT_RUN",
    }
    durable(output / "summary.json", summary)
    return 0 if summary["campaign_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
