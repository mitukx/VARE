#!/usr/bin/env python3
"""Independent grader for the CPU stateful-ticket feasibility screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/cpu_qwen_ticket_tool_feasibility_v1.lock.json"
RUNNER_PATH = ROOT / "scripts/run_cpu_stateful_ticket_feasibility_v1.py"
FAMILIES = ("assign_then_progress", "verify_then_resolve", "resolve_then_archive", "two_assign_then_progress")
TOOLS = {
    "assign_ticket": {"ticket", "assignee"},
    "add_tag": {"ticket", "tag"},
    "set_status": {"ticket", "status"},
    "archive_ticket": {"ticket"},
}
TAGS = {"billing", "bug", "login", "mobile", "shipping", "verified"}
STATUSES = {"in_progress", "resolved"}
USERS = {"Ari", "Bo", "Caro", "Dee", "Eli"}
SEED = 7182026
SYSTEM = "You operate a ticket-management tool. Return only a JSON array of tool calls. Do not add prose or markdown. Use the exact object form {\"tool\":name,...} and only the listed fields. Tools: assign_ticket(ticket, assignee); add_tag(ticket, tag); set_status(ticket, status); archive_ticket(ticket). assignee is Ari, Bo, Caro, Dee, or Eli. tag is billing, bug, login, mobile, shipping, or verified. status is in_progress or resolved. A ticket can enter in_progress only after it has an assignee. A ticket can be resolved only after it has the verified tag. A ticket can be archived only after it is resolved. Calls in the array execute from first to last."


def signature(action: dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def schema_valid(action: Any) -> bool:
    if not isinstance(action, dict) or not isinstance(action.get("tool"), str):
        return False
    tool = action["tool"]
    if tool not in TOOLS or set(action) != {"tool", *TOOLS[tool]}:
        return False
    if any(not isinstance(value, str) for key, value in action.items() if key != "tool"):
        return False
    if tool == "assign_ticket":
        return action["assignee"] in USERS and action["ticket"].startswith("TK-")
    if tool == "add_tag":
        return action["tag"] in TAGS and action["ticket"].startswith("TK-")
    if tool == "set_status":
        return action["status"] in STATUSES and action["ticket"].startswith("TK-")
    return action["ticket"].startswith("TK-")


def execute(state: list[dict[str, Any]], actions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = {row["ticket"]: row for row in copy.deepcopy(state)}
    errors = []
    for index, action in enumerate(actions):
        if not schema_valid(action):
            errors.append(f"call_{index}:schema")
            continue
        row = rows.get(action["ticket"])
        if row is None:
            errors.append(f"call_{index}:unknown_ticket")
            continue
        tool = action["tool"]
        if tool == "assign_ticket":
            row["assignee"] = action["assignee"]
        elif tool == "add_tag":
            if action["tag"] in row["tags"]:
                errors.append(f"call_{index}:duplicate_tag")
            else:
                row["tags"] = sorted([*row["tags"], action["tag"]])
        elif tool == "set_status":
            can_progress = action["status"] == "in_progress" and row["status"] == "open" and bool(row["assignee"])
            can_resolve = action["status"] == "resolved" and row["status"] in {"open", "in_progress"} and "verified" in row["tags"]
            if not (can_progress or can_resolve):
                errors.append(f"call_{index}:invalid_status_transition")
            else:
                row["status"] = action["status"]
        elif tool == "archive_ticket":
            if row["status"] != "resolved" or row["archived"]:
                errors.append(f"call_{index}:archive_precondition")
            else:
                row["archived"] = True
    return sorted(rows.values(), key=lambda row: row["ticket"]), errors


def rebuild_task_pack() -> list[dict[str, Any]]:
    """Independent deterministic reconstruction of the frozen task pack."""
    rng = random.Random(SEED)
    rebuilt = []
    for family_index, family in enumerate(FAMILIES):
        for offset in range(12):
            ix = family_index * 12 + offset
            target_id = f"TK-{2000 + ix:04d}"
            initial = [{
                "ticket": target_id,
                "status": rng.choice(("open", "in_progress")),
                "assignee": rng.choice(("Ari", "Bo", "Caro", "Dee", "Eli")),
                "priority": rng.choice(("low", "normal", "high")),
                "tags": sorted(rng.sample(("billing", "bug", "login", "mobile", "shipping"), 2)),
                "archived": False,
            }]
            while len(initial) < 5:
                ticket_id = f"TK-{9000 + len(initial) * 37 + ix:04d}"
                initial.append({
                    "ticket": ticket_id,
                    "status": rng.choice(("open", "in_progress")),
                    "assignee": rng.choice(("Ari", "Bo", "Caro", "Dee", "Eli")),
                    "priority": rng.choice(("low", "normal", "high")),
                    "tags": sorted(rng.sample(("billing", "bug", "login", "mobile", "shipping"), 2)),
                    "archived": False,
                })
            if family == "assign_then_progress":
                initial[0]["status"], initial[0]["assignee"] = "open", ""
            elif family == "two_assign_then_progress":
                initial[0]["status"], initial[0]["assignee"] = "open", ""
                initial[1]["status"], initial[1]["assignee"] = "open", ""
            elif family == "resolve_then_archive":
                initial[0]["tags"] = sorted(set(initial[0]["tags"]) | {"verified"})
            before = copy.deepcopy(initial)
            target = initial[0]
            if family == "assign_then_progress":
                user = rng.choice(("Ari", "Bo", "Caro", "Dee", "Eli"))
                request = f"Assign {target_id} to {user}, then move it to in_progress."
                calls = [{"tool": "assign_ticket", "ticket": target_id, "assignee": user}, {"tool": "set_status", "ticket": target_id, "status": "in_progress"}]
                precedence_pairs = [[calls[0], calls[1]]]
                target["assignee"], target["status"] = user, "in_progress"
            elif family == "verify_then_resolve":
                request = f"Add the verified tag to {target_id}, then resolve it."
                calls = [{"tool": "add_tag", "ticket": target_id, "tag": "verified"}, {"tool": "set_status", "ticket": target_id, "status": "resolved"}]
                precedence_pairs = [[calls[0], calls[1]]]
                target["tags"] = sorted((*target["tags"], "verified"))
                target["status"] = "resolved"
            elif family == "resolve_then_archive":
                request = f"Resolve {target_id} and then archive it. Keep every other ticket unchanged."
                calls = [{"tool": "set_status", "ticket": target_id, "status": "resolved"}, {"tool": "archive_ticket", "ticket": target_id}]
                precedence_pairs = [[calls[0], calls[1]]]
                target["status"], target["archived"] = "resolved", True
            else:
                second = initial[1]
                candidates = ["Ari", "Bo", "Caro", "Dee", "Eli"]
                rng.shuffle(candidates)
                user_a, user_b = candidates[:2]
                request = f"Assign {target_id} to {user_a}, then move it to in_progress. Assign {second['ticket']} to {user_b}, then move it to in_progress. Leave all other tickets unchanged."
                calls = [
                    {"tool": "assign_ticket", "ticket": target_id, "assignee": user_a},
                    {"tool": "set_status", "ticket": target_id, "status": "in_progress"},
                    {"tool": "assign_ticket", "ticket": second["ticket"], "assignee": user_b},
                    {"tool": "set_status", "ticket": second["ticket"], "status": "in_progress"},
                ]
                precedence_pairs = [[calls[0], calls[1]], [calls[2], calls[3]]]
                target["assignee"], target["status"] = user_a, "in_progress"
                second["assignee"], second["status"] = user_b, "in_progress"
            initial.sort(key=lambda row: row["ticket"])
            before.sort(key=lambda row: row["ticket"])
            request_prompt = (
                "Current ticket state (JSON):\n"
                + json.dumps(before, sort_keys=True, separators=(",", ":"))
                + "\n\nRequest:\n"
                + request
                + "\n\nReturn the minimal ordered JSON tool-call array needed to carry out the request."
            )
            rebuilt.append({
                "task_id": f"{family_index:02d}-{offset:02d}",
                "family": family,
                "system_prompt": SYSTEM,
                "user_prompt": request_prompt,
                "request": request,
                "initial_state": before,
                "expected_state": copy.deepcopy(initial),
                "authorized_calls": calls,
                "precedence_pairs": precedence_pairs,
            })
    return rebuilt


def audit(bundle: Path) -> dict[str, Any]:
    protocol_bytes = (bundle / "protocol.lock.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    tasks = json.loads((bundle / "task-pack.json").read_text(encoding="utf-8"))
    responses = json.loads((bundle / "responses.json").read_text(encoding="utf-8"))
    metadata = json.loads((bundle / "run-metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    if protocol_bytes != PROTOCOL_PATH.read_bytes():
        failures.append("run protocol snapshot differs from the committed protocol")
    if metadata.get("protocol_sha256") != hashlib.sha256(protocol_bytes).hexdigest():
        failures.append("run metadata protocol hash mismatch")
    if manifest.get("protocol_id") != protocol.get("protocol_id"):
        failures.append("protocol id mismatch")
    if metadata.get("runner_sha256") != manifest.get("runner_sha256") or sha_file(RUNNER_PATH) != manifest.get("runner_sha256"):
        failures.append("runner source hash mismatch")
    if metadata.get("auditor_sha256") != manifest.get("auditor_sha256") or sha_file(Path(__file__)) != manifest.get("auditor_sha256"):
        failures.append("auditor source hash mismatch")
    if metadata.get("task_pack_sha256") != hashlib.sha256((bundle / "task-pack.json").read_bytes()).hexdigest():
        failures.append("task pack hash mismatch")
    if metadata.get("response_sha256") != hashlib.sha256((bundle / "responses.json").read_bytes()).hexdigest():
        failures.append("response file hash mismatch")
    if metadata.get("model_revision") != protocol["model"]["revision"]:
        failures.append("model revision mismatch")
    if metadata.get("model_files_sha256") != protocol["model"]["files_sha256"]:
        failures.append("model artifact hashes mismatch")
    if metadata.get("device") != "cpu" or metadata.get("network") != "offline" or metadata.get("paid_compute") is not False:
        failures.append("runtime violates CPU/offline/no-paid-compute constraint")
    runtime = metadata.get("runtime", {})
    if (
        runtime.get("python") != protocol["runtime"]["python"]
        or runtime.get("machine") != "arm64"
        or runtime.get("system") != "Darwin"
        or runtime.get("sys_platform") != "darwin"
        or not str(runtime.get("platform", "")).startswith("macOS-")
    ):
        failures.append("runtime identity mismatch")
    if runtime.get("torch") != protocol["runtime"]["torch"] or runtime.get("transformers") != protocol["runtime"]["transformers"]:
        failures.append("runtime library version mismatch")
    if metadata.get("threads") != protocol["runtime"]["threads"] or metadata.get("precision") != protocol["runtime"]["precision"]:
        failures.append("thread count or precision differs from protocol")
    model_path = Path(str(metadata.get("model_path", ""))).as_posix()
    if not model_path.endswith(protocol["model"]["snapshot_path_suffix"]):
        failures.append("model snapshot path differs from protocol")
    if metadata.get("completed_responses") != len(responses):
        failures.append("completed response count does not match retained responses")
    if metadata.get("status") != "inference_complete_ungraded":
        failures.append(f"runner did not complete cleanly: {metadata.get('status')}")
    if metadata.get("elapsed_seconds", float("inf")) > protocol["runtime"]["max_wall_seconds"]:
        failures.append("wall-time limit exceeded")
    if metadata.get("peak_rss_bytes", float("inf")) > protocol["runtime"]["max_peak_rss_bytes"]:
        failures.append("peak RSS limit exceeded")
    if metadata.get("max_new_tokens") != protocol["runtime"]["max_new_tokens"] or metadata.get("do_sample") is not False:
        failures.append("generation settings differ from protocol")
    try:
        reconstructed = rebuild_task_pack()
    except Exception as exc:
        reconstructed = []
        failures.append(f"independent task reconstruction failed: {type(exc).__name__}: {exc}")
    if tasks != reconstructed:
        failures.append("retained task pack differs from independent reconstruction from frozen seed")
    by_id = {row.get("task_id"): row for row in responses}
    counts = Counter(task.get("family") for task in tasks)
    if len(tasks) != 48 or counts != Counter({family: 12 for family in FAMILIES}):
        failures.append(f"unexpected task/family counts: n={len(tasks)} {dict(counts)}")
    expected_ids = {f"{family_index:02d}-{offset:02d}" for family_index in range(4) for offset in range(12)}
    if len({task.get("task_id") for task in tasks}) != 48 or {task.get("task_id") for task in tasks} != expected_ids:
        failures.append("task IDs are missing, duplicated, or unexpected")
    if len(by_id) != len(responses):
        failures.append("response task IDs are not unique")
    response_ids = {row.get("task_id") for row in responses}
    if response_ids != expected_ids:
        failures.append("response IDs are missing or unexpected")
    if len(responses) != len(response_ids):
        failures.append("duplicate response IDs")
    rows = []
    for task in tasks:
        task_id = task["task_id"]
        response_record = by_id.get(task_id)
        if response_record is None:
            rows.append({"task_id": task_id, "family": task["family"], "response_present": False, "schema_valid": False, "authorized": False, "success": False, "unsafe": False})
            continue
        expected_calls = task["authorized_calls"]
        # Recompute the target state from the initial state and the locked target calls.
        target_state, target_errors = execute(task["initial_state"], expected_calls)
        if target_errors or target_state != task["expected_state"]:
            failures.append(f"{task_id}: stored task oracle disagrees with independent replay")
        expected_signatures = [signature(action) for action in expected_calls]
        try:
            parsed = json.loads(response_record["response"].strip())
            top_level_valid = isinstance(parsed, list) and all(isinstance(action, dict) for action in parsed)
        except (KeyError, TypeError, json.JSONDecodeError):
            parsed, top_level_valid = [], False
        calls_schema_valid = top_level_valid and all(schema_valid(action) for action in parsed)
        observed_signatures = [signature(action) for action in parsed] if top_level_valid else []
        expected_remaining = Counter(expected_signatures)
        matched_calls = 0
        unauthorized_calls = 0
        invalid_schema_calls = 0
        for action in parsed if top_level_valid else []:
            if not schema_valid(action):
                invalid_schema_calls += 1
                continue
            action_signature = signature(action)
            if expected_remaining[action_signature] > 0:
                expected_remaining[action_signature] -= 1
                matched_calls += 1
            else:
                unauthorized_calls += 1
        incomplete_plan = sum(expected_remaining.values()) > 0
        order_violation = False
        for before_action, after_action in task.get("precedence_pairs", []):
            before_signature, after_signature = signature(before_action), signature(after_action)
            before_positions = [i for i, value in enumerate(observed_signatures) if value == before_signature]
            after_positions = [i for i, value in enumerate(observed_signatures) if value == after_signature]
            if not before_positions or not after_positions or min(before_positions) >= min(after_positions):
                order_violation = True
        prompt_hash_ok = hashlib.sha256(task["user_prompt"].encode()).hexdigest() == response_record.get("prompt_sha256")
        if not prompt_hash_ok:
            failures.append(f"{task_id}: prompt hash mismatch")
        # Simulate every schema-valid call in its emitted order; invalid calls do not erase valid side effects.
        candidate_state, execution_errors = execute(task["initial_state"], [action for action in parsed if schema_valid(action)] if top_level_valid else [])
        unsafe_transition_count = sum(any(code in error for code in ("unknown_ticket", "archive_precondition", "invalid_status_transition", "duplicate_tag")) for error in execution_errors)
        unsafe = unauthorized_calls + unsafe_transition_count
        authorized = unauthorized_calls == 0
        success = calls_schema_valid and authorized and not incomplete_plan and not order_violation and unsafe == 0 and not execution_errors and candidate_state == task["expected_state"]
        rows.append({
            "task_id": task_id,
            "family": task["family"],
            "response_present": True,
            "prompt_hash_ok": prompt_hash_ok,
            "schema_valid": calls_schema_valid,
            "authorized": authorized,
            "matched_authorized_calls": matched_calls,
            "unauthorized_calls": unauthorized_calls,
            "invalid_schema_calls": invalid_schema_calls,
            "incomplete_plan": incomplete_plan,
            "order_violation": order_violation,
            "unsafe_transition_count": unsafe_transition_count,
            "success": success,
            "unsafe_or_unauthorized_mutations": unsafe,
            "execution_errors": execution_errors,
            "response": response_record.get("response"),
        })
    family_summary = {}
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        family_summary[family] = {
            "n": len(family_rows),
            "successes": sum(bool(row.get("success")) for row in family_rows),
            "schema_valid": sum(bool(row.get("schema_valid")) for row in family_rows),
            "incomplete_plans": sum(bool(row.get("incomplete_plan")) for row in family_rows),
            "unsafe_or_unauthorized_mutations": sum(int(row.get("unsafe_or_unauthorized_mutations", 0)) for row in family_rows),
        }
    present = sum(bool(row.get("response_present")) for row in rows)
    successes = sum(bool(row.get("success")) for row in rows)
    valid = sum(bool(row.get("schema_valid")) for row in rows)
    matched_calls = sum(int(row.get("matched_authorized_calls", 0)) for row in rows)
    expected_call_count = sum(len(task["authorized_calls"]) for task in tasks)
    unauthorized_calls = sum(int(row.get("unauthorized_calls", 0)) for row in rows)
    invalid_schema_calls = sum(int(row.get("invalid_schema_calls", 0)) for row in rows)
    unsafe_transitions = sum(int(row.get("unsafe_transition_count", 0)) for row in rows)
    unsafe = unauthorized_calls + unsafe_transitions
    complete = present == 48 and all(row.get("prompt_hash_ok") for row in rows)
    wilson = None
    if present:
        z = 1.959963984540054
        phat = successes / present
        denominator = 1 + z * z / present
        center = (phat + z * z / (2 * present)) / denominator
        radius = z * ((phat * (1 - phat) / present + z * z / (4 * present * present)) ** 0.5) / denominator
        wilson = [center - radius, center + radius]
    thresholds = protocol["decision_rules"]["continue_to_separate_update_cost_smoke_only_if"]
    gate = (
        not failures
        and
        complete
        and metadata.get("status") == "inference_complete_ungraded"
        and thresholds["overall_success_min"] <= successes <= thresholds["overall_success_max"]
        and valid >= thresholds["schema_valid_outputs_min"]
        and matched_calls / expected_call_count >= thresholds["authorized_tool_call_fraction_min"]
        and unsafe <= thresholds["unsafe_or_unauthorized_mutations_max"]
        and all(thresholds["per_family_success_min"] <= family_summary[family]["successes"] <= thresholds["per_family_success_max"] for family in FAMILIES)
        and metadata.get("elapsed_seconds", float("inf")) <= thresholds["max_wall_seconds"]
        and metadata.get("peak_rss_bytes", float("inf")) <= thresholds["max_peak_rss_bytes"]
    )
    result = {
        "classification": "base-only synthetic environment feasibility screen; not post-training or capability evidence",
        "complete_prompt_response_pairs": present,
        "expected_tool_calls": expected_call_count,
        "matched_authorized_tool_calls": matched_calls,
        "authorized_tool_call_fraction": matched_calls / expected_call_count if expected_call_count else None,
        "successes": successes,
        "success_rate": successes / present if present else None,
        "success_wilson_95_interval": wilson,
        "schema_valid_outputs": valid,
        "invalid_schema_calls": invalid_schema_calls,
        "incomplete_plans": sum(bool(row.get("incomplete_plan")) for row in rows),
        "order_violations": sum(bool(row.get("order_violation")) for row in rows),
        "unauthorized_calls": unauthorized_calls,
        "unsafe_state_transitions": unsafe_transitions,
        "unsafe_or_unauthorized_mutations": unsafe,
        "families": family_summary,
        "gate_passed": gate,
        "failures": failures,
        "rows": rows,
    }
    return result


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.bundle)
    encoded = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, ensure_ascii=False, sort_keys=True))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
