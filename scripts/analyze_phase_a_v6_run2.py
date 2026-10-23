#!/usr/bin/env python3
"""Derive an append-only failure analysis from immutable Phase A v6 run-2 records."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/phase-a-v6/protocol-v1/run-2"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_run_manifest() -> None:
    for line in (RUN / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = RUN / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"historical run evidence failed hash verification: {relative}")


def rollout_records() -> list[dict[str, Any]]:
    records = []
    for path in sorted(RUN.glob("*/greedy-0/agent.json")) + sorted(
        RUN.glob("*/sampled-*/agent.json")
    ):
        agent = json.loads(path.read_text(encoding="utf-8"))
        task_id = path.parent.parent.name
        tools: list[tuple[str, dict[str, Any], int]] = []
        reads: list[tuple[str, int | None, int | None]] = []
        searches: list[str] = []
        test_results: list[dict[str, Any]] = []
        edit_outcomes: Counter[str] = Counter()
        raw_tool_output_chars = 0
        for turn in agent["transcript"]:
            turn_no = int(turn["turn"])
            for call in turn.get("tool_calls", []):
                name = str(call.get("name", "unknown"))
                args = call.get("arguments") or {}
                result = call.get("result")
                tools.append((name, args, turn_no))
                raw_tool_output_chars += len(
                    json.dumps(result, ensure_ascii=False, sort_keys=True)
                )
                if name == "read_file":
                    reads.append((str(args.get("path", "")), args.get("start_line"), args.get("end_line")))
                elif name == "search_text":
                    searches.append(str(args.get("query", "")))
                elif name == "run_public_tests" and isinstance(result, dict):
                    test_results.append(result)
                if name in {"replace_text", "insert_text", "write_file", "apply_patch", "recovered_edit"}:
                    if isinstance(result, dict) and result.get("applied") is True:
                        edit_outcomes["applied"] += 1
                    elif isinstance(result, dict):
                        edit_outcomes[str(result.get("error") or result.get("format_parse_failure") or "rejected")] += 1
                    else:
                        edit_outcomes["malformed_result"] += 1
        # Automatic raw results are retained at the turn level. The edit result
        # also contains a model-facing feedback summary, which is not a second
        # container invocation and must not be counted as one.
        test_results.extend(
            item
            for turn in agent["transcript"]
            for item in turn.get("automatic_public_tests", [])
            if isinstance(item, dict)
        )
        # Replace/insert tool results and the automatic_public_tests list refer
        # to the same invocation. Deduplicate by exact serialized record.
        deduplicated: dict[str, dict[str, Any]] = {}
        for item in test_results:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            deduplicated[key] = item
        test_results = list(deduplicated.values())

        first_edit_position = next(
            (index for index, (name, _, _) in enumerate(tools, 1)
             if name in {"replace_text", "insert_text", "write_file", "apply_patch", "recovered_edit"}),
            None,
        )
        distinct_reads = set(reads)
        independent_path = path.parent / "independent-grade.json"
        grade = json.loads(independent_path.read_text(encoding="utf-8"))
        records.append(
            {
                "task_id": task_id,
                "rollout_id": path.parent.name,
                "stop_reason": agent.get("stop_reason"),
                "tool_turns": agent.get("tool_turns"),
                "no_tool_retries": agent.get("no_tool_retries"),
                "new_tokens": agent.get("new_tokens"),
                "prompt_tokens": sum(int(turn.get("prompt_tokens", 0)) for turn in agent["transcript"]),
                "edit_attempted": agent.get("edit_attempted"),
                "edit_applied": agent.get("edit_applied"),
                "format_parse_failure_count": agent.get("format_parse_failure_count"),
                "edit_attempt_tool_position": first_edit_position,
                "tool_calls_before_first_edit": (
                    None if first_edit_position is None else first_edit_position - 1
                ),
                "tool_call_counts": dict(sorted(Counter(name for name, _, _ in tools).items())),
                "edit_outcomes": dict(sorted(edit_outcomes.items())),
                "source_search_calls": len(searches),
                "source_search_queries": searches,
                "source_read_calls": len(reads),
                "distinct_source_regions_read": len(distinct_reads),
                "repeated_exact_source_reads": len(reads) - len(distinct_reads),
                "public_test_invocations": [
                    {
                        "classification": item.get("classification"),
                        "returncode": item.get("returncode"),
                        "stderr": item.get("stderr", ""),
                        "error": item.get("error"),
                    }
                    for item in test_results
                ],
                "raw_tool_output_chars": raw_tool_output_chars,
                "finish_calls": sum(name == "finish" for name, _, _ in tools),
                "successful_finish_calls": sum(
                    name == "finish" and isinstance(result, dict) and result.get("finished") is True
                    for turn in agent["transcript"]
                    for result in [call.get("result") for call in turn.get("tool_calls", [])
                                   if call.get("name") == "finish"]
                ),
                "independent_grade_passed": grade.get("passed") is True,
                "independent_grade_failure": grade.get("failure"),
                "agent_json_sha256": sha256_file(path),
                "independent_grade_sha256": sha256_file(independent_path),
            }
        )
    return records


def sum_counters(values: Any) -> Counter[str]:
    total: Counter[str] = Counter()
    for value in values:
        total.update(value)
    return total


def analyze() -> dict[str, Any]:
    verify_run_manifest()
    records = rollout_records()
    tool_counts: Counter[str] = Counter()
    stop_counts: Counter[str] = Counter()
    public_results = []
    for record in records:
        tool_counts.update(record["tool_call_counts"])
        stop_counts.update([record["stop_reason"]])
        public_results.extend(record["public_test_invocations"])
    prompt_tokens = sum(record["prompt_tokens"] for record in records)
    generated_tokens = sum(record["new_tokens"] for record in records)
    return {
        "analysis_schema_version": 1,
        "source_run": "results/phase-a-v6/protocol-v1/run-2",
        "source_run_manifest_sha256": sha256_file(RUN / "run-manifest.json"),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_sha256_manifest_verified": True,
        "rollout_count": len(records),
        "rollouts": records,
        "totals": {
            "tool_calls_by_name": dict(sorted(tool_counts.items())),
            "stop_reasons": dict(sorted(stop_counts.items())),
            "source_search_calls": sum(record["source_search_calls"] for record in records),
            "source_read_calls": sum(record["source_read_calls"] for record in records),
            "distinct_source_regions_read": sum(record["distinct_source_regions_read"] for record in records),
            "repeated_exact_source_reads": sum(record["repeated_exact_source_reads"] for record in records),
            "rollouts_with_edit_attempt": sum(record["edit_attempted"] is True for record in records),
            "rollouts_with_applied_edit": sum(record["edit_applied"] is True for record in records),
            "edit_outcomes": dict(sorted(sum_counters(record["edit_outcomes"] for record in records).items())),
            "tool_turns": sum(record["tool_turns"] for record in records),
            "no_tool_retries": sum(record["no_tool_retries"] for record in records),
            "format_parse_failures": sum(record["format_parse_failure_count"] for record in records),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "prompt_plus_generated_tokens": prompt_tokens + generated_tokens,
            "raw_tool_output_chars": sum(record["raw_tool_output_chars"] for record in records),
            "finish_calls": sum(record["finish_calls"] for record in records),
            "successful_finish_calls": sum(record["successful_finish_calls"] for record in records),
            "public_test_invocations": len(public_results),
            "public_test_exit_125": sum(result.get("returncode") == 125 for result in public_results),
            "public_test_assertions_executed": sum(result.get("returncode") not in (None, 125) for result in public_results),
            "independent_verified_successes": sum(record["independent_grade_passed"] for record in records),
        },
        "interpretation": {
            "exit_125": "Docker CLI/daemon failed to create the bind mount; test code did not execute.",
            "read_efficiency": "Repeated exact path/line-range reads are counted separately; distinct reads are not asserted to be semantically useful.",
            "tool_output_measurement": "Raw serialized tool-result character count. Exact model-visible serialized payloads were not retained in run-2.",
            "reward_validity": "Run-2's zero rewards are historical records. Public-test launch failures mean those invocations provide no assertion-based feedback and cannot be interpreted as test failures.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("refusing to overwrite analysis evidence")
    result = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["totals"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
