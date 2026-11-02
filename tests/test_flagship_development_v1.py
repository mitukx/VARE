"""The prospective grader preserves shared historical comparison behavior."""
import hashlib
from pathlib import Path

import pytest

from scripts import qualify_flagship_development_v1 as experiment
from vare import isolated_candidate as grading

IMAGE = "python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258"


@pytest.mark.parametrize("value", [None, [None, 2], {"routes": [None, "ok"]}])
@pytest.mark.parametrize("bool_as_number", [False, True])
def test_private_null_comparison(monkeypatch, tmp_path: Path, value, bool_as_number):
    original = grading.values_close
    monkeypatch.setattr(grading, "_one_call", lambda *a, **kw: (value, 0.0, None))
    report = experiment.null_safe_grade_calls(
        tmp_path, [grading.FunctionCall("null", "subject", "run", [], value)],
        task_kind="correctness", image=IMAGE, bool_as_number=bool_as_number,
    )
    assert report.passed
    assert grading.values_close is original
    assert not grading.values_close(None, None, rtol=0, atol=0)


@pytest.mark.parametrize("other", [False, 0, "null", [], {}])
def test_null_substitution_rejected(monkeypatch, tmp_path, other):
    monkeypatch.setattr(grading, "_one_call", lambda *a, **kw: (other, 0.0, None))
    report = experiment.null_safe_grade_calls(
        tmp_path, [grading.FunctionCall("wrong", "subject", "run", [], None)],
        task_kind="correctness", image=IMAGE,
    )
    assert not report.passed


def test_external_protocol_anchor_rejects_coherent_working_tree_changes(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text('{}')
    anchor = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text('{"implementation_sha256": {}}')
    with pytest.raises(ValueError, match="operator-supplied"):
        experiment.load_protocol(path, anchor)


def test_thinking_control_includes_ordinary_null_and_disable_cases():
    cases = {call.case_id: call for call in experiment.thinking_calls()}
    assert cases["ordinary-null"].expected["start"] is None
    assert cases["thinking-disabled"].expected["rendered"] == ""
    assert cases["thinking-enabled"].expected["rendered"] == "<think>"
