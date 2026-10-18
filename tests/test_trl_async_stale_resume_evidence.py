from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.reproduce_trl_async_stale_resume_v1 import audit_bundle, parse_run_log


ROOT = Path(__file__).resolve().parents[1]
LOCKED_RUN = ROOT / "results/trl-async-stale-resume-v1/run-2-locked-repetitions"
EPOCH_AWARE_RUN = ROOT / "results/trl-async-stale-resume-v2/run-1-epoch-aware-replay"
V1_FALSE_NEGATIVE = ROOT / "results/trl-async-stale-resume-v1/run-3-final-runner-replay"


def test_reproduction_parser_extracts_stale_cursor_and_replay_count():
    text = (ROOT / "results/trl-async-stale-resume-v1/run-1-original-threshold/baseline-2.log").read_text(
        encoding="utf-8"
    )
    observed = parse_run_log(text)
    assert observed["saved_prompt_index"] == 2
    assert observed["resumed_row"] == 2
    assert observed["duplicated_rows"] == 144


def test_locked_run_bundle_recomputes_from_hashed_logs():
    summary = audit_bundle(LOCKED_RUN)
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


def test_superseded_row_identity_run_retains_its_nonpass():
    summary = audit_bundle(V1_FALSE_NEGATIVE)
    assert summary["protocol_id"] == "trl-async-stale-resume-v1"
    assert summary["all_checks_passed"] is False
    candidate_failures = [
        run for run in summary["runs"] if run["arm"] == "candidate" and not run["accepted"]
    ]
    assert [run["repetition"] for run in candidate_failures] == [1]


def test_locked_run_bundle_rejects_changed_raw_log(tmp_path: Path):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    log = copied / "baseline-1.log"
    log.write_text(log.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_bundle(copied)
