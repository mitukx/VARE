from __future__ import annotations

import json
import shutil
from pathlib import Path
import subprocess

import pytest

from scripts.reproduce_trl_async_stale_resume_v1 import (
    audit_bundle,
    parse_run_log,
    parse_pytest_outcome,
    source_revision,
    write_json,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
LOCKED_RUN = ROOT / "results/trl-async-stale-resume-v1/run-2-locked-repetitions"
EPOCH_AWARE_RUN = ROOT / "results/trl-async-stale-resume-v2/run-1-epoch-aware-replay"
V2_REVIEW_FIX_RUN = ROOT / "results/trl-async-stale-resume-v2/run-2-review-fixes"
V1_FALSE_NEGATIVE = ROOT / "results/trl-async-stale-resume-v1/run-3-final-runner-replay"


def test_reproduction_parser_extracts_stale_cursor_and_replay_count():
    text = (ROOT / "results/trl-async-stale-resume-v1/run-1-original-threshold/baseline-2.log").read_text(
        encoding="utf-8"
    )
    observed = parse_run_log(text)
    assert observed["saved_prompt_index"] == 2
    assert observed["resumed_row"] == 2
    assert observed["duplicated_rows"] == 144


def test_pytest_outcome_parser_preserves_failed_and_passed_counts():
    assert parse_pytest_outcome("1 failed, 1 passed, 2 warnings in 8.32s") == {
        "failed": 1,
        "passed": 1,
        "warnings": 2,
    }


def test_v2_review_fix_bundle_recomputes_from_hashed_logs_and_retained_runner():
    summary = audit_bundle(V2_REVIEW_FIX_RUN)
    assert summary["run_count"] == 6
    assert summary["all_checks_passed"] is True


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
    log.write_text(log.read_text(encoding="utf-8") + "\n1 failed in 0.01s\n", encoding="utf-8")
    write_manifest(copied)

    with pytest.raises(ValueError, match="recorded exit code contradicts raw pytest result"):
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
