"""Audit append-only public-test invocation accounting for Phase A v6 run-2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/phase-a-v6/protocol-v1/run-2"
CORRECTION = RUN / "public-test-classification-correction-v2/classification.json"


def test_correction_accounts_for_every_serialized_public_test_invocation() -> None:
    correction = json.loads(CORRECTION.read_text())
    events = correction["candidate_container_results"]
    observed: list[tuple[str, str, str, int | None, float | None]] = []
    for agent_path in sorted(RUN.glob("*/**/agent.json")):
        agent = json.loads(agent_path.read_text())
        rel = agent_path.relative_to(RUN).as_posix()
        digest = hashlib.sha256(agent_path.read_bytes()).hexdigest()
        for turn_index, turn in enumerate(agent.get("transcript", [])):
            if isinstance(turn, dict):
                for result_index, result in enumerate(turn.get("automatic_public_tests", [])):
                    observed.append((rel, digest, "automatic_after_edit", turn_index, result_index))
        if isinstance(agent.get("public_test_result"), dict):
            observed.append((rel, digest, "explicit_tool_call", None, None))

    recorded = [
        (item["agent_json"], item["agent_json_sha256"], item["event_kind"],
         item["transcript_turn"], item["result_index"])
        for item in events
    ]
    assert sorted(recorded, key=repr) == sorted(observed, key=repr)
    assert len(events) == correction["serialized_invocation_count"] == 7
    assert correction["automatic_after_edit_count"] == 1
    assert correction["explicit_tool_call_count"] == 6
    assert correction["classification_counts"] == {
        "container_launch_failure": 7,
        "candidate_assertion_failure": 0,
        "timeout": 0,
        "test_functions_executed": 0,
    }
    assert all(event["returncode"] == 125 for event in events)
    assert all(event["classification"] == "container_launch_failure" for event in events)
    assert all(event["test_function_executed"] is False for event in events)


def test_correction_manifest_and_frozen_run_manifest_verify() -> None:
    sums = (CORRECTION.parent / "SHA256SUMS").read_text().split()
    assert sums[0] == hashlib.sha256(CORRECTION.read_bytes()).hexdigest()
    run_sums = (RUN / "SHA256SUMS").read_text().splitlines()
    for line in run_sums:
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256((RUN / rel).read_bytes()).hexdigest() == digest
