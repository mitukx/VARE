from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_verifier_drift_pilot_stop import (
    validate_output_path,
    write_report,
)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def test_existing_success_evidence_cannot_be_selected_as_audit_output(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "SHA256.json").write_bytes(b'{"frozen":true}\n')
    (run / "summary.json").write_bytes(b'{"status":"complete"}\n')
    before = _snapshot(run)

    with pytest.raises(ValueError, match="already exists"):
        validate_output_path(run / "SHA256.json", run)

    assert _snapshot(run) == before


def test_existing_failure_record_cannot_be_overwritten(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    failure = b'{"type":"OriginalFailure","message":"retain"}\n'
    (run / "failure.json").write_bytes(failure)
    before = _snapshot(run)

    with pytest.raises(ValueError, match="already exists"):
        validate_output_path(run / "failure.json", run)

    assert _snapshot(run) == before
    assert (run / "failure.json").read_bytes() == failure


def test_existing_external_protocol_or_runner_cannot_be_selected_as_output(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    supplied_sources = [tmp_path / "protocol.json", tmp_path / "runner.py"]
    for path in supplied_sources:
        path.write_bytes(f"frozen:{path.name}".encode())
    before = {path: path.read_bytes() for path in supplied_sources}

    for path in supplied_sources:
        with pytest.raises(ValueError, match="already exists"):
            validate_output_path(path, run)

    assert {path: path.read_bytes() for path in supplied_sources} == before


def test_new_report_cannot_be_created_inside_retained_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    before = _snapshot(run)

    with pytest.raises(ValueError, match="outside the retained run"):
        validate_output_path(run / "audit.json", run)

    assert _snapshot(run) == before
    assert not (run / "audit.json").exists()


def test_output_symlink_cannot_redirect_write_to_supplied_source(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes(b'{"frozen":"protocol"}\n')
    output = tmp_path / "audit.json"
    output.symlink_to(protocol)
    before = protocol.read_bytes()

    with pytest.raises(ValueError, match="already exists"):
        validate_output_path(output, run)

    assert output.is_symlink()
    assert protocol.read_bytes() == before


def test_exclusive_report_write_preserves_destination_created_after_preflight(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    output = validate_output_path(tmp_path / "audit.json", run)
    raced = b'{"preexisting":"do not replace"}\n'
    output.write_bytes(raced)

    with pytest.raises(ValueError, match="appeared after preflight"):
        write_report(output, {"status": "PASS"})

    assert output.read_bytes() == raced
