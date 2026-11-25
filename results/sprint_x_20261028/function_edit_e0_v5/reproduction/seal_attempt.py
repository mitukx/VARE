#!/usr/bin/env python3
"""Independently grade and seal the completed v5 trajectory."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1]
ATTEMPT = EVIDENCE / "attempt-001"
BASE = Path("/tmp/vare-sprint-x-function-e0-005/candidate-workspace")
PYTHON = Path("/tmp/vare-sprint-vii-mlx/bin/python")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def main() -> int:
    if not (ATTEMPT / "agent-result.json").is_file():
        raise SystemExit("agent result missing; cannot seal")
    source = BASE / "src/quant.py"
    grade_start = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run([
        str(PYTHON), "-I", str(EVIDENCE / "grader/independent_grader_v1.py"),
        "--candidate", str(source),
    ], capture_output=True, text=True, check=False)
    grade_end = datetime.now(timezone.utc).isoformat()
    if completed.returncode != 0:
        raise SystemExit(f"independent grader crashed: {completed.stderr}")
    grade = json.loads(completed.stdout)
    (ATTEMPT / "independent-grade.json").write_text(
        json.dumps(grade, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (ATTEMPT / "candidate-src-quant.py").write_bytes(source.read_bytes())
    patch = subprocess.check_output(["git", "-C", str(BASE), "diff", "--binary", "--", "src/quant.py"])
    (ATTEMPT / "candidate.patch").write_bytes(patch)

    records = [json.loads(line) for line in (ATTEMPT / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    previous = "0" * 64
    chain_ok = True
    for record in records:
        event = record["event"]
        canonical = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        expected_hash = sha_bytes(canonical.encode("utf-8"))
        if record.get("previous_event_sha256") != previous or record.get("event_sha256") != expected_hash:
            chain_ok = False
            break
        previous = expected_hash
    result = json.loads((ATTEMPT / "agent-result.json").read_text(encoding="utf-8"))
    transcript = result.get("transcript", [])
    calls = [call for turn in transcript for call in turn.get("tool_calls", []) if call.get("name")]
    metrics = {
        "turns": len(transcript),
        "tool_call_count": len(calls),
        "tool_names": [call["name"] for call in calls],
        "valid_schema_turns": sum(turn.get("tool_output_status") == "complete_tool_call" for turn in transcript),
        "generated_tokens": result.get("new_tokens"),
        "sum_prompt_tokens": sum(turn.get("prompt_tokens", 0) for turn in transcript),
        "sum_generation_wall_seconds": round(sum(turn.get("generation_seconds", 0.0) for turn in transcript), 6),
        "edit_applied": result.get("edit_applied"),
        "model_explicitly_finished": result.get("model_explicitly_finished"),
        "agent_stop_reason": result.get("stop_reason"),
        "candidate_diff_sha256": sha(ATTEMPT / "candidate.patch"),
        "candidate_source_sha256": sha(ATTEMPT / "candidate-src-quant.py"),
    }
    summary = {
        "protocol_id": json.loads((EVIDENCE / "protocol.json").read_text())["protocol_id"],
        "protocol_sha256": (EVIDENCE / "protocol.sha256").read_text().split()[0],
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "grader_started_after_model_run": True,
        "grader_execution_window": {"start_utc": grade_start, "end_utc": grade_end},
        "independent_grade": grade,
        "grader_receipt_sha256": sha(ATTEMPT / "independent-grade.json"),
        "candidate_execution": "grader interpreted a restricted AST; candidate was never imported or executed",
        "event_chain_valid": chain_ok,
        "event_count": len(records),
        "event_journal_sha256": sha(ATTEMPT / "events.jsonl"),
        "metrics": metrics,
        "resource_summary": json.loads((ATTEMPT / "resource-summary.json").read_text()),
        "public_tests": "not exposed or run",
        "classification": "synthetic end-to-end feasibility; not an external Development family",
    }
    (ATTEMPT / "verification-summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if chain_ok and grade.get("status") in {"PASS", "FAIL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
