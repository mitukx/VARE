#!/usr/bin/env python3
"""Create a hash-bound retrospective action analysis of frozen Phase A v6."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/phase-a-v6/protocol-v1/run-2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"),
    ).hexdigest()


def verify_raw_bundle() -> None:
    subprocess.run(
        ["shasum", "-a", "256", "-c", "SHA256SUMS"],
        cwd=RUN, check=True, capture_output=True, text=True,
    )


def analyze_rollout(manifest_row: dict) -> dict:
    task = manifest_row["task_id"]
    rollout_id = manifest_row["rollout_id"]
    folder = RUN / task / rollout_id
    agent_path = folder / "agent.json"
    agent = json.loads(agent_path.read_text(encoding="utf-8"))
    turns = agent["transcript"]
    tool_counts: Counter[str] = Counter()
    source_reads = []
    searches = []
    edits = []
    tests = []
    finishes = []
    exact_duplicate_source_reads = 0
    first_edit_call_index = None
    all_calls = []
    output_chars = 0

    for turn in turns:
        for call in turn.get("tool_calls", []):
            all_calls.append((turn["turn"], call))
    for call_index, (turn_number, call) in enumerate(all_calls, start=1):
        name = call.get("name", "unknown")
        result = call.get("result")
        tool_counts[name] += 1
        output_chars += len(json.dumps(result, ensure_ascii=False, sort_keys=True))
        if name == "read_file":
            payload = result if isinstance(result, dict) else {}
            content_fingerprint = fingerprint([
                payload.get("path"), payload.get("start_line"),
                payload.get("end_line"), payload.get("content", ""),
            ])
            source_reads.append({
                "turn": turn_number,
                "path": payload.get("path"),
                "start_line": payload.get("start_line"),
                "end_line": payload.get("end_line"),
                "content_chars": len(payload.get("content", "")),
                "result_fingerprint": fingerprint([name, call.get("arguments"), result]),
                "content_fingerprint": content_fingerprint,
            })
        elif name == "search_text":
            search_payload = result if isinstance(result, list) else []
            searches.append({
                "turn": turn_number,
                "query": call.get("arguments", {}).get("query"),
                "path": call.get("arguments", {}).get("path"),
                "hit_count": len(search_payload),
                "result_fingerprint": fingerprint([name, call.get("arguments"), result]),
            })
        elif name in {"replace_text", "insert_text", "write_file", "apply_patch", "recovered_edit"}:
            if first_edit_call_index is None:
                first_edit_call_index = call_index
            edit_result = result if isinstance(result, dict) else {}
            edits.append({
                "turn": turn_number,
                "tool": name,
                "applied": edit_result.get("applied") is True,
                "error": edit_result.get("error"),
            })
            automatic = edit_result.get("automatic_public_test")
            if isinstance(automatic, dict):
                tests.append({"kind": "automatic", **_test_outcome(automatic)})
        elif name == "run_public_tests":
            payload = result if isinstance(result, dict) else {}
            tests.append({"kind": "manual", **_test_outcome(payload)})
        elif name == "finish":
            finishes.append({"turn": turn_number, "finished": bool(result and result.get("finished"))})

    read_content_counts = Counter(item["content_fingerprint"] for item in source_reads)
    exact_duplicate_source_reads = sum(count - 1 for count in read_content_counts.values())
    unique_read_fingerprints = len(read_content_counts)
    search_counts = Counter(item["result_fingerprint"] for item in searches)
    grade_path = folder / "independent-grade.json"
    grade = json.loads(grade_path.read_text(encoding="utf-8")) if grade_path.is_file() else {}
    return {
        "task_id": task,
        "rollout_id": rollout_id,
        "seed": manifest_row.get("seed"),
        "agent_record_sha256": sha256(agent_path),
        "turns": len(turns),
        "stop_reason": agent.get("stop_reason"),
        "no_tool_retries": agent.get("no_tool_retries", 0),
        "tool_calls_by_name": dict(sorted(tool_counts.items())),
        "search_calls": len(searches),
        "searches_with_hits": sum(item["hit_count"] > 0 for item in searches),
        "exact_duplicate_search_results": sum(count - 1 for count in search_counts.values()),
        "read_calls": len(source_reads),
        "read_content_chars": sum(item["content_chars"] for item in source_reads),
        "unique_read_outputs": unique_read_fingerprints,
        "exact_duplicate_read_outputs": exact_duplicate_source_reads,
        "tool_calls_before_first_edit": (
            first_edit_call_index - 1 if first_edit_call_index is not None else None
        ),
        "first_edit_turn": min((edit["turn"] for edit in edits), default=None),
        "read_calls_before_first_edit": sum(
            item["turn"] < min((edit["turn"] for edit in edits), default=10**9)
            for item in source_reads
        ),
        "edit_calls": len(edits),
        "applied_edit_calls": sum(item["applied"] for item in edits),
        "rejected_edit_calls": sum(not item["applied"] for item in edits),
        "edit_results": edits,
        "public_test_events": tests,
        "finish_calls": len(finishes),
        "successful_finish_calls": sum(item["finished"] for item in finishes),
        "prompt_tokens_summed_over_turns": sum(turn.get("prompt_tokens", 0) for turn in turns),
        "generated_tokens": sum(turn.get("generated_tokens", 0) for turn in turns),
        "tool_result_json_characters": output_chars,
        "max_context_tokens": agent.get("budgets", {}).get("max_context_tokens"),
        "independent_grader": {
            "passed": grade.get("passed"),
            "failure": grade.get("failure"),
            "status": grade.get("status"),
        },
    }


def _test_outcome(result: dict) -> dict:
    returncode = result.get("returncode")
    stderr = result.get("stderr", "")
    legacy_classification = None
    if returncode == 125:
        legacy_classification = (
            "docker_bind_mount_launch_failure"
            if "bind source path does not exist" in stderr
            else "docker_cli_exit_125_unclassified"
        )
    return {
        "passed": result.get("passed"),
        "returncode": returncode,
        "legacy_infrastructure_classification": legacy_classification,
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
    }


def build() -> dict:
    verify_raw_bundle()
    manifest_path = RUN / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [analyze_rollout(row) for row in manifest["rollouts"]]
    tools: Counter[str] = Counter()
    for row in rows:
        tools.update(row["tool_calls_by_name"])
    tests = [event for row in rows for event in row["public_test_events"]]
    tests_by_code = Counter(str(event["returncode"]) for event in tests)
    stops = Counter(row["stop_reason"] for row in rows)
    raw_files = [
        line.split(maxsplit=1)[1].lstrip("*")
        for line in (RUN / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "analysis_id": "vare.phase-a-v6.action-analysis.v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_run_manifest_sha256": sha256(manifest_path),
        "source_sha256sums_sha256": sha256(RUN / "SHA256SUMS"),
        "source_raw_file_count": len(raw_files),
        "rollout_count": len(rows),
        "raw_bundle_verified": True,
        "definitions": {
            "exact_duplicate_read_outputs": "Repeated read_file calls with identical arguments and identical result bytes.",
            "prompt_tokens_summed_over_turns": "Sum of recorded prompt_tokens per generation; not deduplicated unique context tokens.",
            "public_test_events": "Manual run_public_tests calls plus automatic public-test records attached to edit outcomes; not duplicate top-level latest-state fields.",
            "returncode_125_classification": "Historical infrastructure failure only when stderr contains the recorded Docker daemon bind-source diagnostic.",
        },
        "aggregate": {
            "tool_calls": dict(sorted(tools.items())),
            "tool_call_total": sum(tools.values()),
            "turn_total": sum(row["turns"] for row in rows),
            "no_tool_retries": sum(row["no_tool_retries"] for row in rows),
            "search_calls": sum(row["search_calls"] for row in rows),
            "searches_with_hits": sum(row["searches_with_hits"] for row in rows),
            "exact_duplicate_search_results": sum(row["exact_duplicate_search_results"] for row in rows),
            "source_reads": sum(row["read_calls"] for row in rows),
            "unique_read_outputs": sum(row["unique_read_outputs"] for row in rows),
            "exact_duplicate_read_outputs": sum(row["exact_duplicate_read_outputs"] for row in rows),
            "edit_calls": sum(row["edit_calls"] for row in rows),
            "applied_edit_calls": sum(row["applied_edit_calls"] for row in rows),
            "rejected_edit_calls": sum(row["rejected_edit_calls"] for row in rows),
            "rollouts_with_any_edit": sum(row["edit_calls"] > 0 for row in rows),
            "rollouts_without_edit": sum(row["edit_calls"] == 0 for row in rows),
            "tool_calls_before_first_edit_by_rollout": {
                f"{row['task_id']}/{row['rollout_id']}": row["tool_calls_before_first_edit"]
                for row in rows
            },
            "public_test_events": len(tests),
            "public_test_returncodes": dict(sorted(tests_by_code.items())),
            "public_test_docker_bind_failures": sum(
                event["legacy_infrastructure_classification"] == "docker_bind_mount_launch_failure"
                for event in tests
            ),
            "independent_verified_successes": sum(
                row["independent_grader"]["passed"] is True for row in rows
            ),
            "stop_reasons": dict(sorted(stops.items())),
            "prompt_tokens_summed_over_turns": sum(row["prompt_tokens_summed_over_turns"] for row in rows),
            "generated_tokens": sum(row["generated_tokens"] for row in rows),
            "tool_result_json_characters": sum(row["tool_result_json_characters"] for row in rows),
        },
        "rollouts": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"refusing to overwrite analysis output: {args.output}")
    result = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
