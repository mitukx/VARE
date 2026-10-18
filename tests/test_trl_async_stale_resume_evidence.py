from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.reproduce_trl_async_stale_resume_v1 import (
    audit_bundle,
    execution_platform,
    expected_runtime_environment,
    parse_run_log,
    parse_pytest_outcome,
    protocol_timeout_seconds,
    resolved_requirements,
    run_pytest_arm,
    source_revision,
    sha256_file,
    validate_new_run_protocol,
    write_json,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
LOCKED_RUN = ROOT / "results/trl-async-stale-resume-v1/run-2-locked-repetitions"
EPOCH_AWARE_RUN = ROOT / "results/trl-async-stale-resume-v2/run-1-epoch-aware-replay"
V2_REVIEW_FIX_RUN = ROOT / "results/trl-async-stale-resume-v2/run-2-review-fixes"
V1_FALSE_NEGATIVE = ROOT / "results/trl-async-stale-resume-v1/run-3-final-runner-replay"
V3_PROTOCOL = ROOT / "protocols/trl_async_stale_resume_v3.lock.json"
V4_PROTOCOL = ROOT / "protocols/trl_async_stale_resume_v4.lock.json"
V4_PLATFORM_RUN = ROOT / "results/trl-async-stale-resume-v4/run-1-platform-locked"


def test_reproduction_parser_extracts_stale_cursor_and_replay_count():
    text = (ROOT / "results/trl-async-stale-resume-v1/run-1-original-threshold/baseline-2.log").read_text(
        encoding="utf-8"
    )
    observed = parse_run_log(text)
    assert observed["saved_prompt_index"] == 2
    assert observed["resumed_row"] == 2
    assert observed["duplicated_rows"] == 144


@pytest.mark.parametrize(
    "contradictory_measurement, field",
    [("saved prompt_index: 2", "saved_prompt_index"), ("rows trained TWICE: 999", "duplicated_rows")],
)
def test_audit_rejects_resealed_bundle_with_duplicate_measurement(
    tmp_path: Path, contradictory_measurement: str, field: str
):
    copied = tmp_path / "run"
    shutil.copytree(V4_PLATFORM_RUN, copied)
    summary = json.loads((copied / "summary.json").read_text(encoding="utf-8"))
    candidate_run = next(run for run in summary["runs"] if run["arm"] == "candidate")
    log = copied / candidate_run["log"]
    log.write_text(
        log.read_text(encoding="utf-8") + f"\n{contradictory_measurement}\n",
        encoding="utf-8",
    )
    write_manifest(copied)

    with pytest.raises(ValueError, match=f"repeats {field}"):
        audit_bundle(copied)


def test_pytest_outcome_parser_preserves_failed_and_passed_counts():
    assert parse_pytest_outcome("1 failed, 1 passed, 2 warnings in 8.32s") == {
        "failed": 1,
        "passed": 1,
        "warnings": 2,
    }


def test_pytest_outcome_parser_rejects_multiple_summaries():
    with pytest.raises(ValueError, match="multiple pytest result summaries"):
        parse_pytest_outcome("1 failed in 0.01s\n1 passed in 0.01s\n")


def test_pytest_outcome_parser_allows_missing_summary_only_for_failed_arm():
    assert parse_pytest_outcome("ERROR collecting test_example.py\n", allow_missing=True) == {}
    with pytest.raises(ValueError, match="lacks a pytest result summary"):
        parse_pytest_outcome("ERROR collecting test_example.py\n")


def test_execution_platform_uses_the_selected_python(monkeypatch: pytest.MonkeyPatch):
    expected = {
        "sys_platform": "darwin",
        "system": "Darwin",
        "release": "27.0.0",
        "machine": "x86_64",
        "mac_ver": "27.0.1",
        "hardware_model": "MacBookPro18,3",
        "processor": "Apple M1 Pro",
        "memory_bytes": 34359738368,
    }
    calls = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(expected))

    monkeypatch.setattr("scripts.reproduce_trl_async_stale_resume_v1.subprocess.run", fake_run)

    assert execution_platform("/selected/python") == expected
    assert calls[0][:2] == ["/selected/python", "-c"]


def test_v3_lock_pins_transitive_dependencies_and_timeout():
    lock = json.loads(V3_PROTOCOL.read_text(encoding="utf-8"))
    requirements = resolved_requirements(lock["environment"])
    assert requirements is not None
    assert len(requirements) > 40
    assert requirements["packaging"]
    assert requirements["pluggy"]
    assert requirements["numpy"]
    assert lock["reproducer"]["per_run_timeout_seconds"] == 600
    assert protocol_timeout_seconds(lock) == 600
    v2_lock = json.loads((ROOT / "protocols/trl_async_stale_resume_v2.lock.json").read_text())
    with pytest.raises(ValueError, match="no valid locked per-run timeout"):
        protocol_timeout_seconds(v2_lock)


def test_new_reproductions_require_a_platform_bound_protocol():
    v3_lock = json.loads(V3_PROTOCOL.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="execution-platform fields"):
        validate_new_run_protocol(v3_lock)
    validate_new_run_protocol(json.loads(V4_PROTOCOL.read_text(encoding="utf-8")))


def test_pytest_arm_timeout_preserves_partial_output_as_failed_result(tmp_path: Path):
    output, exit_code, timed_out = run_pytest_arm(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"],
        tmp_path,
        dict(os.environ),
        timeout_seconds=2,
    )
    assert timed_out is True
    assert exit_code == 124
    assert "started" in output
    assert "VARE_RUNNER_TIMEOUT_SECONDS=2" in output


def _convert_bundle_to_v3(copied: Path) -> dict:
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lock = json.loads(V3_PROTOCOL.read_text(encoding="utf-8"))
    summary["protocol_id"] = lock["protocol_id"]
    summary["protocol_sha256"] = sha256_file(V3_PROTOCOL)
    summary["environment"] = expected_runtime_environment(lock["environment"])
    runner_path = ROOT / "scripts/reproduce_trl_async_stale_resume_v1.py"
    runner_bytes = runner_path.read_bytes()
    (copied / "runner.py").write_bytes(runner_bytes)
    summary["runner_sha256"] = hashlib.sha256(runner_bytes).hexdigest()
    for run in summary["runs"]:
        run["timed_out"] = False
        run["timeout_seconds"] = lock["reproducer"]["per_run_timeout_seconds"]
    write_json(summary_path, summary)
    write_manifest(copied)
    return summary


def test_audit_rejects_resealed_v3_bundle_with_changed_transitive_dependency(tmp_path: Path):
    copied = tmp_path / "run"
    _convert_bundle_to_v3(copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["environment"]["versions"]["packaging"] = "0.0.0-tampered"
    write_json(summary_path, summary)
    write_manifest(copied)

    with pytest.raises(ValueError, match="recorded runtime environment"):
        audit_bundle(copied)


def test_audit_rejects_resealed_v4_bundle_with_changed_execution_platform(tmp_path: Path):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lock = json.loads(V4_PROTOCOL.read_text(encoding="utf-8"))
    summary["protocol_id"] = lock["protocol_id"]
    summary["protocol_sha256"] = sha256_file(V4_PROTOCOL)
    summary["environment"] = expected_runtime_environment(lock["environment"])
    runner_bytes = (ROOT / "scripts/reproduce_trl_async_stale_resume_v1.py").read_bytes()
    (copied / "runner.py").write_bytes(runner_bytes)
    summary["runner_sha256"] = hashlib.sha256(runner_bytes).hexdigest()
    for run in summary["runs"]:
        run["timed_out"] = False
        run["timeout_seconds"] = lock["reproducer"]["per_run_timeout_seconds"]
    summary["environment"]["execution_platform"]["machine"] = "x86_64"
    write_json(summary_path, summary)
    write_manifest(copied)

    with pytest.raises(ValueError, match="recorded runtime environment"):
        audit_bundle(copied)


def test_audit_retains_v3_timeout_as_nonpassing_arm(tmp_path: Path):
    copied = tmp_path / "run"
    summary = _convert_bundle_to_v3(copied)
    summary_path = copied / "summary.json"
    failed_run = next(run for run in summary["runs"] if run["arm"] == "candidate")
    failed_run["exit_code"] = 124
    failed_run["timed_out"] = True
    failed_run["accepted"] = False
    log_path = copied / failed_run["log"]
    log_path.write_text(
        log_path.read_text(encoding="utf-8") + "\nVARE_RUNNER_TIMEOUT_SECONDS=600\n",
        encoding="utf-8",
    )
    summary["all_checks_passed"] = False
    write_json(summary_path, summary)
    write_manifest(copied)

    assert audit_bundle(copied)["all_checks_passed"] is False


def test_audit_seals_early_failed_arm_with_partial_measurements(tmp_path: Path):
    copied = tmp_path / "run"
    summary = _convert_bundle_to_v3(copied)
    summary_path = copied / "summary.json"
    failed_run = next(run for run in summary["runs"] if run["arm"] == "candidate")
    failed_run["exit_code"] = 2
    failed_run["timed_out"] = False
    failed_run["accepted"] = False
    failed_run["observed"] = {"first_run_trained_rows": 144}
    (copied / failed_run["log"]).write_text(
        "ERROR collecting test_repro.py\n"
        "run 1 trained rows: 144\n"
        "1 error in 0.01s\n"
        "VARE_RUNNER_EXIT_CODE=2\n",
        encoding="utf-8",
    )
    summary["all_checks_passed"] = False
    write_json(summary_path, summary)
    write_manifest(copied)

    audited = audit_bundle(copied)

    assert audited["all_checks_passed"] is False
    assert failed_run["observed"] == {"first_run_trained_rows": 144}


def test_audit_cli_returns_nonzero_for_valid_nonpassing_bundle(tmp_path: Path):
    copied = tmp_path / "run"
    summary = _convert_bundle_to_v3(copied)
    summary_path = copied / "summary.json"
    failed_run = next(run for run in summary["runs"] if run["arm"] == "candidate")
    failed_run["exit_code"] = 124
    failed_run["timed_out"] = True
    failed_run["accepted"] = False
    log_path = copied / failed_run["log"]
    log_path.write_text(
        log_path.read_text(encoding="utf-8") + "\nVARE_RUNNER_TIMEOUT_SECONDS=600\n",
        encoding="utf-8",
    )
    summary["all_checks_passed"] = False
    write_json(summary_path, summary)
    write_manifest(copied)

    result = subprocess.run(
        [sys.executable, "scripts/audit_trl_async_stale_resume_v1.py", str(copied)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert json.loads(result.stdout)["all_checks_passed"] is False


def test_v2_review_fix_bundle_recomputes_from_hashed_logs_and_retained_runner():
    summary = audit_bundle(V2_REVIEW_FIX_RUN)
    assert summary["run_count"] == 6
    assert summary["all_checks_passed"] is True


def test_v4_bundle_binds_and_audits_the_execution_platform():
    summary = audit_bundle(V4_PLATFORM_RUN)
    lock = json.loads(V4_PROTOCOL.read_text(encoding="utf-8"))
    assert summary["all_checks_passed"] is True
    assert summary["environment"]["execution_platform"] == lock["environment"]["execution_platform"]
    assert summary["environment"]["device_used"] == "CPU"


def test_epoch_aware_bundle_distinguishes_next_epoch_from_replayed_position():
    summary = audit_bundle(EPOCH_AWARE_RUN)
    assert summary["protocol_id"] == "trl-async-stale-resume-v2"
    baseline = [run["observed"]["duplicated_rows"] for run in summary["runs"] if run["arm"] == "baseline"]
    candidate = [run["observed"]["duplicated_rows"] for run in summary["runs"] if run["arm"] == "candidate"]
    assert baseline == [144, 145, 148]
    assert candidate == [1, 1, 1]
    assert summary["all_checks_passed"] is True


def test_superseded_v1_nonpass_is_retained_but_rejected_without_runner_source():
    summary = json.loads((V1_FALSE_NEGATIVE / "summary.json").read_text(encoding="utf-8"))
    assert summary["protocol_id"] == "trl-async-stale-resume-v1"
    assert summary["all_checks_passed"] is False
    with pytest.raises(ValueError, match="runner source is not retained"):
        audit_bundle(V1_FALSE_NEGATIVE)
    candidate_failures = [
        run for run in summary["runs"] if run["arm"] == "candidate" and not run["accepted"]
    ]
    assert [run["repetition"] for run in candidate_failures] == [1]
    failed_log = (V1_FALSE_NEGATIVE / candidate_failures[0]["log"]).read_text(encoding="utf-8")
    assert parse_pytest_outcome(failed_log)["failed"] == 1


def test_legacy_v1_bundle_is_not_accepted_without_its_runner_source():
    with pytest.raises(ValueError, match="runner source is not retained"):
        audit_bundle(LOCKED_RUN)


def test_audit_rejects_resealed_runner_hash_without_retained_source(tmp_path: Path):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runner_sha256"] = "0" * 64
    write_json(summary_path, summary)
    write_manifest(copied)

    with pytest.raises(ValueError, match="runner source is not retained"):
        audit_bundle(copied)


def test_audit_accepts_exact_runner_source_embedded_in_bundle(tmp_path: Path):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    runner_sha = json.loads((copied / "summary.json").read_text(encoding="utf-8"))["runner_sha256"]
    snapshot = ROOT / "results/trl-async-stale-resume-runner-snapshots" / f"runner-{runner_sha}.py"
    shutil.copyfile(snapshot, copied / "runner.py")
    write_manifest(copied)

    assert audit_bundle(copied)["all_checks_passed"] is True


def test_audit_rejects_resealed_success_bundle_with_failure_in_raw_pytest_log(tmp_path: Path):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    log = copied / "candidate-1.log"
    log.write_text(
        log.read_text(encoding="utf-8")
        + "\n1 failed in 0.01s\n1 passed in 0.01s\n",
        encoding="utf-8",
    )
    write_manifest(copied)

    with pytest.raises(ValueError, match="multiple pytest result summaries"):
        audit_bundle(copied)


@pytest.mark.parametrize("decision_field", ["accepted", "all_checks_passed"])
def test_audit_rejects_integer_values_for_boolean_decisions(tmp_path: Path, decision_field: str):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if decision_field == "accepted":
        summary["runs"][0][decision_field] = 1
    else:
        summary[decision_field] = 1
    write_json(summary_path, summary)
    write_manifest(copied)

    with pytest.raises(ValueError, match="must be a boolean"):
        audit_bundle(copied)


def test_locked_run_bundle_rejects_changed_raw_log(tmp_path: Path):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    log = copied / "baseline-1.log"
    log.write_text(log.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_bundle(copied)


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_reproduction_rejects_dirty_source_checkout(tmp_path: Path, dirty_kind: str):
    checkout = tmp_path / "source"
    checkout.mkdir()
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Evidence Test"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "evidence@example.invalid"], check=True
    )
    tracked = checkout / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "tracked.py"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "--quiet", "-m", "base"], check=True)

    if dirty_kind == "tracked":
        tracked.write_text("value = 2\n", encoding="utf-8")
    else:
        (checkout / "rogue.py").write_text("value = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source checkout must be clean"):
        source_revision(checkout)


@pytest.mark.parametrize("inventory_error", ["missing_arm", "duplicate"])
def test_audit_rejects_resealed_bundle_with_invalid_run_inventory(
    tmp_path: Path, inventory_error: str
):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = [run for run in summary["runs"] if run["arm"] == "candidate"]
    if inventory_error == "missing_arm":
        summary["runs"] = candidates + [
            {**candidates[0], "repetition": repetition, "log": f"candidate-{repetition}.log"}
            for repetition in range(4, 7)
        ]
    else:
        summary["runs"] = [candidates[0].copy() for _ in range(6)]
    # Keep the self-reported six-run count and reseal every file to exercise
    # arm/repetition validation rather than manifest integrity.
    summary["run_count"] = 6
    summary["all_checks_passed"] = True
    write_json(summary_path, summary)
    write_manifest(copied)

    with pytest.raises(ValueError, match="duplicate run identity|run inventory mismatch"):
        audit_bundle(copied)


@pytest.mark.parametrize("provenance_field", ["environment", "gist_source_sha256"])
def test_audit_rejects_resealed_bundle_with_unlocked_execution_provenance(
    tmp_path: Path, provenance_field: str
):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if provenance_field == "environment":
        summary["environment"]["versions"]["torch"] = "0.0.0-tampered"
    else:
        summary["gist_source_sha256"] = "0" * 64
    write_json(summary_path, summary)
    write_manifest(copied)

    with pytest.raises(ValueError, match="recorded runtime environment|recorded Gist source hash"):
        audit_bundle(copied)
