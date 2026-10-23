"""Audit append-only public-test invocation accounting for Phase A v6 run-2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/phase-a-v6/protocol-v1/run-2"
CORRECTION = RUN / "public-test-classification-correction-v3/classification.json"


def test_correction_accounts_for_every_serialized_public_test_invocation() -> None:
    correction = json.loads(CORRECTION.read_text())
    events = correction["candidate_container_results"]
    observed: dict[tuple[str, str, int | None, int | None], dict] = {}
    for agent_path in sorted(RUN.glob("*/**/agent.json")):
        agent = json.loads(agent_path.read_text())
        rel = agent_path.relative_to(RUN).as_posix()
        digest = hashlib.sha256(agent_path.read_bytes()).hexdigest()
        for turn_index, turn in enumerate(agent.get("transcript", [])):
            if isinstance(turn, dict):
                for result_index, result in enumerate(turn.get("automatic_public_tests", [])):
                    observed[(rel, "automatic_after_edit", turn_index, result_index)] = {
                        "digest": digest,
                        "result": result,
                    }
        if isinstance(agent.get("public_test_result"), dict):
            observed[(rel, "explicit_tool_call", None, None)] = {
                "digest": digest,
                "result": agent["public_test_result"],
            }

    recorded = {
        (item["agent_json"], item["event_kind"], item["transcript_turn"], item["result_index"]): item
        for item in events
    }
    assert recorded.keys() == observed.keys()
    for key, item in recorded.items():
        raw = observed[key]
        result = raw["result"]
        assert item["agent_json_sha256"] == raw["digest"]
        assert item["returncode"] == result["returncode"]
        assert item["elapsed_seconds"] == result["elapsed_seconds"]
        assert item["result_sha256"] == hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert "bind source path does not exist" in result["stderr"].lower()
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
    correction = json.loads(CORRECTION.read_text())
    assert correction["source_run_manifest_sha256"] == hashlib.sha256(
        (RUN / "run-manifest.json").read_bytes()
    ).hexdigest()
    assert correction["source_run_checksums_sha256"] == hashlib.sha256(
        (RUN / "SHA256SUMS").read_bytes()
    ).hexdigest()
    assert correction["source_report_sha256"] == hashlib.sha256(
        (RUN / "REPORT.md").read_bytes()
    ).hexdigest()
