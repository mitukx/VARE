"""Read-only preflight and synthetic negative-control unit tests (no Docker needed)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import preflight_mlxlm_1912_v4 as preflight


def _fake_pins(monkeypatch, source: Path) -> None:
    monkeypatch.setattr(preflight, "verify_protocol_v4", lambda: {
        "protocol_id": "mlxlm-1912-isolated-task-v4",
        "runtime": {"image": "pinned/image@sha256:abc"},
    })
    monkeypatch.setattr(preflight, "BASELINE_SOURCE_SHA256", hashlib.sha256(b"baseline").hexdigest())
    monkeypatch.setattr(preflight, "FIXED_SOURCE_SHA256", hashlib.sha256(b"fixed").hexdigest())

    def git(argv, *, env=None):
        assert argv[-2:] == ["rev-parse", "--show-toplevel"]
        return str(source.resolve()).encode()

    monkeypatch.setattr(preflight, "_run", git)
    monkeypatch.setattr(
        preflight, "_pinned_source",
        lambda _repo, revision: b"baseline"
        if revision == preflight.BASELINE_COMMIT else b"fixed",
    )


def test_preflight_checks_both_pinned_revisions_without_creating_output(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _fake_pins(monkeypatch, source)
    output = tmp_path / "run-1"
    info = preflight.verify_inputs(source, output)
    assert info["protocol_id"] == "mlxlm-1912-isolated-task-v4"
    assert info["image"] == "pinned/image@sha256:abc"
    assert not output.exists()


def test_preflight_refuses_existing_evidence(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _fake_pins(monkeypatch, source)
    output = tmp_path / "run-1"
    output.mkdir()
    with pytest.raises(FileExistsError):
        preflight.verify_inputs(source, output)


def test_preflight_rejects_fixed_source_drift(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _fake_pins(monkeypatch, source)
    monkeypatch.setattr(preflight, "_pinned_source", lambda _repo, _revision: b"baseline")
    with pytest.raises(ValueError, match="pinned source hash mismatch"):
        preflight.verify_inputs(source, tmp_path / "run-1")


@pytest.mark.parametrize("bad_state", ["passed", "wrong_case"])
def test_negative_control_never_certifies_false_rejection(
    tmp_path: Path, monkeypatch, bad_state: str
) -> None:
    def fake_git(_argv, *, env=None):
        return b""

    def fake_export(_checkout, workspace):
        target = workspace / preflight.TARGET_PATH
        target.parent.mkdir(parents=True)
        target.write_text(
            "import mlx.core as mx\n\n"
            "def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:\n"
            "    return logprobs\n\n"
            "@partial(mx.compile)\ndef next_function():\n    pass\n",
            encoding="utf-8",
        )

    def fake_grade(workspace, *, image):
        source = (workspace / preflight.TARGET_PATH).read_text(encoding="utf-8")
        assert "mx.where(mx.arange(logprobs.shape[-1]) == 0" in source
        if bad_state == "passed":
            return {"passed": True, "cases": [{"case_id": "shifted_maximum_nonzero_values", "passed": True}]}
        return {"passed": False, "cases": [{"case_id": "other_case", "passed": False}]}

    monkeypatch.setattr(preflight, "_run", fake_git)
    monkeypatch.setattr(preflight, "export_candidate", fake_export)
    monkeypatch.setattr(preflight, "grade_workspace", fake_grade)
    output = tmp_path / "evidence"
    output.mkdir()
    assert not preflight.run_adversarial_probe(tmp_path, output, "pinned/image")
    assert (output / "adversarial-negative-control.json").is_file()
    assert (output / "adversarial-negative-control.sha256").is_file()


def test_negative_control_requires_shifted_maximum_to_fail(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(preflight, "_run", lambda _argv, *, env=None: b"")

    def fake_export(_checkout, workspace):
        target = workspace / preflight.TARGET_PATH
        target.parent.mkdir(parents=True)
        target.write_text(
            "def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:\n"
            "    return logprobs\n\n"
            "@partial(mx.compile)\ndef next_function():\n    pass\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(preflight, "export_candidate", fake_export)
    monkeypatch.setattr(
        preflight, "grade_workspace",
        lambda _workspace, *, image: {
            "passed": False,
            "cases": [
                *[
                    {"case_id": f"existing-control-{index}", "passed": True,
                     "failure": None}
                    for index in range(12)
                ],
                {"case_id": "shifted_maximum_nonzero_values",
                 "passed": False, "failure": "wrong_output"},
                {"case_id": "leading_maximum_nonzero_value",
                 "passed": False, "failure": "wrong_output"},
            ],
        },
    )
    output = tmp_path / "evidence"
    output.mkdir()
    assert preflight.run_adversarial_probe(tmp_path, output, "pinned/image")
