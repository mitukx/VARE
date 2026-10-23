#!/usr/bin/env python3
"""Independently audit the retained one-shot Qwen development run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/qwen3_4b_mlsys_development_feasibility_v1.lock.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _variance(values: list[bool]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((float(value) - mean) ** 2 for value in values) / len(values)


def _git_bytes(revision: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{revision}:{relative}"],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def audit(run: Path) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result = json.loads((run / "feasibility.json").read_text(encoding="utf-8"))
    preflight = json.loads((run / "preflight.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run / "raw-records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest_rows = {}
    for line in (run / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        manifest_rows[name] = digest
    expected_files = {"feasibility.json", "raw-records.jsonl", "preflight.json"}
    if set(manifest_rows) != expected_files:
        raise ValueError("run checksum manifest has missing or extra entries")
    for name, digest in manifest_rows.items():
        if sha256(run / name) != digest:
            raise ValueError(f"run checksum mismatch: {name}")
    if result["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("result protocol identity differs")
    if preflight["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("preflight protocol identity differs")
    if preflight["model_snapshot_files_sha256"] != protocol["model"]["snapshot_files_sha256"]:
        raise ValueError("model snapshot files differ from frozen protocol")
    run_commit = result["implementation_commit"]
    for relative, expected in protocol["implementation_sha256"].items():
        actual = hashlib.sha256(_git_bytes(run_commit, relative)).hexdigest()
        if actual != expected:
            raise ValueError(f"implementation source differs at run commit: {relative}")
    run_protocol = _git_bytes(run_commit, "protocols/qwen3_4b_mlsys_development_feasibility_v1.lock.json")
    if hashlib.sha256(run_protocol).hexdigest() != sha256(PROTOCOL):
        raise ValueError("run commit does not contain the current frozen protocol")
    if [record["task_id"] for record in records if record["kind"] == "greedy"] != protocol["task_ids"]:
        raise ValueError("greedy task order differs from the frozen order")
    if len(records) != len(protocol["task_ids"]) * 5:
        raise ValueError("run does not contain exactly the frozen 15 rollouts")

    summaries = {}
    for task_id in protocol["task_ids"]:
        task_rows = [row for row in records if row["task_id"] == task_id]
        greedy = [row for row in task_rows if row["kind"] == "greedy"]
        sampled = [row for row in task_rows if row["kind"] == "sampled"]
        if len(greedy) != 1 or len(sampled) != protocol["sampling"]["sampled_trials_per_task"]:
            raise ValueError(f"rollout count differs for {task_id}")
        if [row["seed"] for row in sampled] != protocol["sampling"]["sampled_seeds"][task_id]:
            raise ValueError(f"sampling seeds differ for {task_id}")
        for row in task_rows:
            output_hash = hashlib.sha256(row["candidate_output"].encode()).hexdigest()
            if row["candidate_output_sha256"] != output_hash:
                raise ValueError("candidate output hash mismatch")
            if row["prompt_sha256"] != protocol["prompts"][task_id]["sha256"]:
                raise ValueError(f"prompt differs from protocol for {task_id}")
        rewards = [row["passed"] for row in sampled]
        summaries[task_id] = {
            "greedy_success": greedy[0]["passed"],
            "sample_successes": rewards,
            "pass_at_k": any(rewards),
            "reward_variance": _variance(rewards),
            "unique_sample_output_hashes": len(
                {row["candidate_output_sha256"] for row in sampled}
            ),
            "failures": {
                failure: sum(row.get("failure") == failure for row in task_rows)
                for failure in ("format_failure", "wrong_patch")
            },
        }
    summary_fields = ("greedy_success", "sample_successes", "pass_at_k", "reward_variance")
    stored_summaries = {
        task_id: {field: values[field] for field in summary_fields}
        for task_id, values in result["task_summary"].items()
    }
    reconstructed_summaries = {
        task_id: {field: values[field] for field in summary_fields}
        for task_id, values in summaries.items()
    }
    if reconstructed_summaries != stored_summaries:
        raise ValueError("stored task summaries do not reconstruct from raw rollouts")
    if result["records"] != records:
        raise ValueError("JSON result rows differ from append-only raw records")
    if result["policy_updates_run"] != 0:
        raise ValueError("feasibility protocol unexpectedly includes a policy update")
    if not all(row["failure"] == "format_failure" for row in records):
        raise ValueError("observed failure classes differ from the recorded feasibility result")
    for row in records:
        if row["process_peak_rss_bytes"] > protocol["limits"]["max_process_rss_bytes"]:
            raise ValueError("process RSS exceeded frozen limit")
    if result["memory_end"]["available_bytes"] < protocol["limits"]["min_available_memory_bytes"]:
        raise ValueError("available memory fell below frozen floor")
    if result["memory_end"]["swap_free_bytes"] < protocol["limits"]["min_swap_free_bytes"]:
        raise ValueError("free swap fell below frozen floor")
    return {
        "audit_passed": True,
        "protocol_sha256": sha256(PROTOCOL),
        "implementation_commit": run_commit,
        "rollouts": len(records),
        "model_revision": protocol["model"]["immutable_revision"],
        "task_summary": summaries,
        "checksums_verified": sorted(expected_files),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run.resolve(strict=True)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
