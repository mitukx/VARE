from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.audit_verifier_integrity_evidence_v2 import audit_chain, write_chain


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/verifier_integrity_pilot_v4/run1"
AUDITOR = ROOT / "scripts/audit_verifier_integrity_evidence_v2.py"


def _copy_run(destination: Path) -> Path:
    copied = destination / "run1"
    shutil.copytree(RUN, copied)
    return copied


def test_committed_run_has_a_complete_transitive_evidence_chain():
    result = audit_chain(RUN, AUDITOR)

    assert result["status"] == "pass"
    assert result["artifact_count"] == 41
    assert result["scored_outcome_count"] == 18
    assert result["published_outer_entries_matching"] == 35
    assert result["published_outer_entries"] == 36
    assert result["reconstructed_outer_entries_matching"] == 36
    assert result["historical_mismatch_preserved"] is True
    assert result["recovered_original_chain_valid"] is True


def test_outcome_bytes_are_bound_to_the_top_level_chain(tmp_path: Path):
    run = _copy_run(tmp_path)
    write_chain(run, AUDITOR)
    outcome = run / "outcomes/2dcf69441110ee64-baseline.json"
    outcome.write_bytes(outcome.read_bytes() + b" ")

    with pytest.raises(ValueError, match="artifact hash/size mismatch"):
        audit_chain(run, AUDITOR)


def test_missing_scored_outcome_fails_closed(tmp_path: Path):
    run = _copy_run(tmp_path)
    write_chain(run, AUDITOR)
    (run / "outcomes/2dcf69441110ee64-baseline.json").unlink()

    with pytest.raises(ValueError, match="artifact inventory is not exact"):
        audit_chain(run, AUDITOR)


def test_historical_record_must_remain_byte_identical(tmp_path: Path):
    run = _copy_run(tmp_path)
    write_chain(run, AUDITOR)
    (run / "run-evidence-manifest.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash/size mismatch"):
        audit_chain(run, AUDITOR)
