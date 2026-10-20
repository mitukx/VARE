"""Expected exceptions are part of the isolated grader's parent decision."""
from __future__ import annotations

import vare.isolated_candidate as isolated
from vare.isolated_candidate import CandidateException, FunctionCall


def test_expected_exception_case_is_not_a_process_failure(monkeypatch, tmp_path):
    call = FunctionCall(
        case_id="empty-window", module="grad_accum", function="average_microbatch_grads",
        args=[[]], expected=None, kwargs={"accumulation_steps": 2},
        expected_exception="ValueError",
    )
    monkeypatch.setattr(
        isolated, "_one_call",
        lambda *_args, **_kwargs: (CandidateException("ValueError"), 0.01, None),
    )
    report = isolated.grade_calls(
        tmp_path, [call], task_kind="correctness", image="python@sha256:" + "a" * 64,
    )
    assert report.passed is True
    assert report.cases[0].failure is None


def test_exception_wire_payload_is_typed_and_nonce_bound():
    value, failure = isolated._parse_result(
        b'VARE_RESULT test-nonce {"exception":"ValueError"}', "test-nonce",
    )
    assert failure is None
    assert value == CandidateException("ValueError")
    _value, failure = isolated._parse_result(
        b'VARE_RESULT other-nonce {"exception":"ValueError"}', "test-nonce",
    )
    assert failure == "malformed_result"
