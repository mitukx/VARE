#!/usr/bin/env python3
"""Reconstruct the sprint's model, grading, and retention facts from raw files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chain(path: Path) -> list[dict]:
    previous = "0" * 64
    events = []
    for sequence, line in enumerate(path.read_bytes().splitlines(), 1):
        item = json.loads(line)
        event = item["event"]
        canonical = json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        current = digest(canonical)
        if (event["sequence"] != sequence or item["previous_event_sha256"] != previous
                or item["event_sha256"] != current):
            raise ValueError(f"event chain mismatch: {path}:{sequence}")
        previous = current
        events.append(event)
    if not events:
        raise ValueError("empty event chain")
    return events


def strict_tool_call(raw: str) -> bool:
    stripped = raw.strip()
    opening, closing = "<tool_call>", "</tool_call>"
    if not stripped.startswith(opening) or not stripped.endswith(closing):
        return False
    if stripped.count(opening) != 1 or stripped.count(closing) != 1:
        return False
    try:
        payload = json.loads(stripped[len(opening):-len(closing)])
    except json.JSONDecodeError:
        return False
    return (isinstance(payload, dict) and set(payload) == {"name", "arguments"}
            and isinstance(payload["name"], str) and isinstance(payload["arguments"], dict))


def audit(outputs: Path) -> dict:
    assets = ROOT / "benchmarks/vare_development_repair_v1"
    definitions = {
        "model-campaign": "campaign-protocol.json",
        "cpu-model-campaign": "campaign-cpu-v2-protocol.json",
        "qwen3-cpu-model-campaign": "campaign-qwen3-cpu-protocol.json",
    }
    cells = []
    for directory, protocol_name in definitions.items():
        protocol_path = assets / protocol_name
        protocol = json.loads(protocol_path.read_text())
        summary = json.loads((outputs / directory / "summary.json").read_text())
        if summary["protocol_sha256"] != digest(protocol_path.read_bytes()):
            raise ValueError("campaign protocol mismatch")
        cases = {case["id"]: case for case in protocol["cases"]}
        for result in summary["rows"]:
            case = cases[result["case_id"]]
            cell = outputs / directory / case["id"]
            if json.loads((cell / "result.json").read_text()) != result:
                raise ValueError("summary differs from raw cell result")
            events = chain(cell / "events.jsonl")
            invocations = [event for event in events if event["event_type"] == "invocation_started"]
            if len(invocations) != 1:
                raise ValueError("duplicate or missing model invocation")
            invocation = invocations[0]
            commit = invocation["policy_source_commit"]
            for name, expected in protocol["source_hashes"].items():
                raw = subprocess.check_output(["git", "-C", str(ROOT), "--no-replace-objects", "show", f"{commit}:{name}"])
                if digest(raw) != expected:
                    raise ValueError(f"committed source identity mismatch: {name}")
            if invocation["model_files_sha256"] != case["model_files_sha256"]:
                raise ValueError("model identity mismatch")
            source = cell / "workspace" / case["source_path"]
            before = digest((assets / case["baseline_asset"]).read_bytes())
            after = digest(source.read_bytes())
            outputs_done = [event for event in events if event["event_type"] == "model_output_completed"]
            tool_events = [event for event in events if event["event_type"] == "tool_result"]
            generated_tokens = sum(event["generated_tokens"] for event in outputs_done)
            process = json.loads((cell / "inference/process.json").read_text())
            row = {"campaign": directory, "case_id": case["id"], "source_commit": commit,
                   "status": result["status"], "completed_generations": len(outputs_done),
                   "valid_tool_json": sum(strict_tool_call(event["model_output"]) for event in outputs_done),
                   "tool_results": len(tool_events), "retained_generated_tokens": generated_tokens,
                   "source_changed": before != after, "candidate_sha256": after,
                   "inference_wall_s": process["elapsed_s"], "peak_sampled_rss_bytes": process["peak_sampled_rss_bytes"],
                   "host_swap_growth_bytes": process["swap_used_end_bytes"] - process["swap_used_start_bytes"],
                   "abort": process["abort"], "model_invocations": len(invocations),
                   "independent_grader_success": None}
            if result["status"] == "GRADED":
                record = json.loads((cell / "agent-record.json").read_text())
                if record["new_tokens"] != generated_tokens:
                    raise ValueError("agent token total mismatch")
                grade = json.loads((cell / "grading/stdout.log").read_text())
                if grade != result["grade"] or grade["candidate_sha256"] != after:
                    raise ValueError("grade/candidate identity mismatch")
                if grade["success"] != (bool(grade.get("checks")) and all(grade["checks"].values())):
                    raise ValueError("grade success inconsistent with checks")
                pending = json.loads((cell / "pending-grade.json").read_text())
                for name, expected in pending["files_sha256"].items():
                    if digest((cell / name).read_bytes()) != expected:
                        raise ValueError("pending recovery artifact mismatch")
                row["independent_grader_success"] = grade["success"]
                row["stop_reason"] = record["stop_reason"]
                row["recovered_after_sigkill"] = result["recovered_after_sigkill"]
                if row["recovered_after_sigkill"] and not (cell / "fault-injected.json").exists():
                    raise ValueError("recovery lacks fault marker")
            cells.append(row)
    return {"status": "AUDIT_PASS", "cells": cells,
            "completed_graded_attempts": sum(row["status"] == "GRADED" for row in cells),
            "successful_repairs": sum(row["independent_grader_success"] is True for row in cells),
            "resource_aborted_attempts": sum(row["abort"] is not None for row in cells),
            "scope": "Raw-source, event-chain, model-identity, artifact, and outcome reconstruction. It is not external independent review or an authenticated hardware execution log."}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.outputs.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cells"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
