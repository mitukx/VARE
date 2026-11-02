from __future__ import annotations

from pathlib import Path

from scripts import run_local_agent


def test_diff_validation_failure_is_returned_for_trajectory_recording(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_validation(*_args, **_kwargs):
        raise ValueError("candidate workspace contains staged or non-allowlisted changes")

    monkeypatch.setattr(run_local_agent, "candidate_source_diff", fail_validation)

    diff, error = run_local_agent.canonical_candidate_diff_or_error(
        tmp_path, ["module.py"], {"module.py": b"before\n"}, "a" * 40,
    )

    assert diff == ""
    assert error == "ValueError: candidate workspace contains staged or non-allowlisted changes"
