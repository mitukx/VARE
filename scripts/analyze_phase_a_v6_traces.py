"""Recompute bounded action-completion metrics from frozen Phase A v6 traces."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


EDIT_TOOLS = {"replace_text", "insert_text", "apply_patch", "write_file"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_rollout(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    grade_path = path.parent / "independent-grade.json"
    grade = json.loads(grade_path.read_text(encoding="utf-8"))
    calls: list[dict[str, Any]] = []
    for transcript_turn in record.get("transcript", []):
        for call in transcript_turn.get("tool_calls", []):
            calls.append({"turn": transcript_turn.get("turn"), **call})

    reads = [call for call in calls if call.get("name") == "read_file"]
    read_regions = [
        (
            call.get("arguments", {}).get("path"),
            call.get("arguments", {}).get("start_line", 1),
            call.get("arguments", {}).get("end_line"),
        )
        for call in reads
    ]
    searches = [call for call in calls if call.get("name") == "search_text"]
    edits = [call for call in calls if call.get("name") in EDIT_TOOLS]
    first_edit = min((call.get("turn", 0) for call in edits), default=None)
    tests = [call for call in calls if call.get("name") == "run_public_tests"]
    finish_calls = [call for call in calls if call.get("name") == "finish"]
    automatic_tests = [
        call["result"]["automatic_public_test"]
        for call in calls
        if isinstance(call.get("result"), dict)
        and isinstance(call["result"].get("automatic_public_test"), dict)
    ]

    def result_applied(call: dict[str, Any]) -> bool:
        result = call.get("result")
        return isinstance(result, dict) and result.get("applied") is True

    def useful_search(call: dict[str, Any]) -> bool:
        result = call.get("result")
        return isinstance(result, list) and bool(result)

    return {
        "task_id": record.get("task_id"),
        "seed": record.get("seed"),
        "trace_path": path.as_posix(),
        "trace_sha256": sha256(path),
        "turns": record.get("tool_turns"),
        "tokens": record.get("new_tokens"),
        "wall_seconds": record.get("elapsed_seconds"),
        "stop_reason": record.get("stop_reason"),
        "tool_calls_by_name": dict(Counter(str(call.get("name")) for call in calls)),
        "search_calls": len(searches),
        "search_calls_with_results": sum(useful_search(call) for call in searches),
        "read_calls": len(reads),
        "unique_read_regions": len(set(read_regions)),
        "repeated_read_calls": len(read_regions) - len(set(read_regions)),
        "turns_before_first_edit": first_edit - 1 if first_edit is not None else None,
        "edit_attempts": len(edits),
        "applied_edits": sum(result_applied(call) for call in edits),
        "rejected_edits": sum(not result_applied(call) for call in edits),
        "public_test_tool_calls": len(tests),
        "automatic_public_test_records": len(automatic_tests),
        "public_test_attempts": len(tests) + len(automatic_tests),
        "public_test_launch_failures_125": sum(
            "bind source path does not exist" in str(result.get("feedback", ""))
            for result in automatic_tests
        ) + sum(
            isinstance(call.get("result"), dict)
            and call["result"].get("returncode") == 125
            for call in tests
        ),
        "no_tool_retries": record.get("no_tool_retries"),
        "format_parse_failures": record.get("format_parse_failure_count"),
        "finish_calls": len(finish_calls),
        "finish_accepted": any(
            isinstance(call.get("result"), dict)
            and call["result"].get("finished") is True
            for call in finish_calls
        ),
        "independent_reward": int(grade.get("passed") is True),
        "independent_failure": grade.get("failure"),
        "grade_sha256": sha256(grade_path),
        "candidate_edit_applied": record.get("edit_applied"),
        "public_test_passed": record.get("public_test_passed"),
    }


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase A v6 trace reanalysis",
        "",
        f"Protocol SHA-256: `{report['protocol_sha256']}`.",
        "The source run and its original report remain unchanged.",
        "",
        "| Task | Seed | Turns | Tokens | Search | Reads (unique/repeat) | Turns before edit | Edits (applied/rejected) | Public-test attempts | Stop reason | Grader failure | Reward |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in report["rollouts"]:
        reads = f"{row['read_calls']} ({row['unique_read_regions']}/{row['repeated_read_calls']})"
        edits = f"{row['edit_attempts']} ({row['applied_edits']}/{row['rejected_edits']})"
        lines.append(
            f"| `{row['task_id']}` | {row['seed']} | {row['turns']} | {row['tokens']} | "
            f"{row['search_calls']} ({row['search_calls_with_results']} nonempty) | {reads} | "
            f"{row['turns_before_first_edit'] if row['turns_before_first_edit'] is not None else 'none'} | "
            f"{edits} | {row['public_test_attempts']} | `"
            f"{row['stop_reason']}` | `{row['independent_failure']}` | "
            f"{row['independent_reward']} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate interpretation",
            "",
            f"- Rollouts: {report['rollout_count']}; valid model tool calls: {report['tool_call_count']}; "
            f"tool turns: {report['tool_turn_count']}; generated tokens: {report['generated_tokens']}; "
            f"wall time: {report['wall_seconds']:.2f}s.",
            f"- Search calls: {report['search_calls']} ({report['nonempty_search_calls']} returned matches); "
            f"source reads: {report['read_calls']} ({report['unique_read_regions']} unique regions, "
            f"{report['repeated_read_calls']} repeated regions).",
            f"- Edit actions: {report['edit_attempts']} ({report['applied_edits']} applied, "
            f"{report['rejected_edits']} rejected across {report['rollouts_with_edit_attempts']} "
            f"rollouts); public-test attempts: {report['public_test_attempts']} "
            f"({report['public_test_launch_failures_125']} exit-125 bind launch failures); "
            f"no-tool retries: {report['no_tool_retries']}.",
            f"- Nonempty source searches: {report['nonempty_search_calls']}/"
            f"{report['search_calls']}; empty searches: "
            f"{report['search_calls'] - report['nonempty_search_calls']}. The exact-region "
            "read-repeat measure does not judge whether a distinct read was useful.",
            f"- Parse failures: {report['format_parse_failures']}; accepted finishes: "
            f"{report['accepted_finishes']}; independent rewards: {report['reward_sum']}/"
            f"{report['rollout_count']}.",
            "- Every recorded public-test return code 125 was a Docker bind-source launch failure "
            "(`bind source path does not exist`), not a timeout or assertion. Such outcomes are "
            "infrastructure failures and must not be interpreted as model reward.",
            "- The controller already gained stable Colima-visible workspace staging and implicit "
            "latest-read handle binding in PR #106. This analysis does not justify another "
            "outcome-tuned controller change before a new cohort and a clean resource preflight.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    output_dir = args.output_dir
    if output_dir.exists():
        parser.error("refusing to overwrite analysis evidence")
    protocol_path = run_dir / "protocol-v1.json"
    manifest_path = run_dir / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rollouts = sorted(run_dir.glob("transformers-*/**/agent.json"))
    rows = [analyze_rollout(path) for path in rollouts]
    report: dict[str, Any] = {
        "protocol_id": json.loads(protocol_path.read_text(encoding="utf-8"))["protocol_id"],
        "protocol_sha256": sha256(protocol_path),
        "run_manifest_sha256": sha256(manifest_path),
        "source_head": manifest.get("source_repo_head"),
        "rollout_count": len(rows),
        "tool_call_count": sum(sum(row["tool_calls_by_name"].values()) for row in rows),
        "tool_turn_count": sum(int(row["turns"] or 0) for row in rows),
        "generated_tokens": sum(int(row["tokens"] or 0) for row in rows),
        "wall_seconds": sum(float(row["wall_seconds"] or 0) for row in rows),
        "search_calls": sum(row["search_calls"] for row in rows),
        "nonempty_search_calls": sum(row["search_calls_with_results"] for row in rows),
        "read_calls": sum(row["read_calls"] for row in rows),
        "unique_read_regions": sum(row["unique_read_regions"] for row in rows),
        "repeated_read_calls": sum(row["repeated_read_calls"] for row in rows),
        "edit_attempts": sum(row["edit_attempts"] for row in rows),
        "applied_edits": sum(row["applied_edits"] for row in rows),
        "rejected_edits": sum(row["rejected_edits"] for row in rows),
        "rollouts_with_edit_attempts": sum(row["edit_attempts"] > 0 for row in rows),
        "public_test_tool_calls": sum(row["public_test_tool_calls"] for row in rows),
        "automatic_public_test_records": sum(row["automatic_public_test_records"] for row in rows),
        "public_test_attempts": sum(row["public_test_attempts"] for row in rows),
        "public_test_launch_failures_125": sum(
            row["public_test_launch_failures_125"] for row in rows
        ),
        "no_tool_retries": sum(int(row["no_tool_retries"] or 0) for row in rows),
        "format_parse_failures": sum(int(row["format_parse_failures"] or 0) for row in rows),
        "accepted_finishes": sum(row["finish_accepted"] for row in rows),
        "reward_sum": sum(int(row["independent_reward"] or 0) for row in rows),
        "rollouts": rows,
    }
    output_dir.mkdir(parents=True)
    json_path = output_dir / "analysis-v1.json"
    md_path = output_dir / "analysis-v1.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(report_markdown(report), encoding="utf-8")
    sums = {path.name: sha256(path) for path in (json_path, md_path)}
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())),
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "rollout_count", "tool_call_count", "tool_turn_count", "generated_tokens",
        "wall_seconds", "edit_attempts", "applied_edits", "public_test_attempts",
        "no_tool_retries", "accepted_finishes", "reward_sum",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
