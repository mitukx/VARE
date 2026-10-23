from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

from scripts import grade_transformers_phase_a_integrity_v10 as grader
from scripts import grade_transformers_phase_a_v1 as phase_a
from vare.isolated_candidate import FunctionCall
from vare.isolated_candidate import values_close


ROOT = Path(__file__).resolve().parents[1]
IMAGE = (
    "vare/transformers-phase-a-arm64:1@sha256:"
    "7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"
)


def test_v10_full_registry_comparison_accepts_matching_null_routes_and_rejects_mutations():
    protocol, lock, _ = grader._load_protocol()
    expected = grader._complete_registry_expected_output(protocol, lock, phase_a)

    assert sum(value is None for value in expected["runtime_registry"].values()) == 10
    assert grader._values_close_with_null(expected, expected, rtol=0, atol=0)

    mutated = deepcopy(expected)
    null_route = next(
        name for name, value in mutated["runtime_registry"].items() if value is None
    )
    mutated["runtime_registry"][null_route] = "ForgedTokenizer"
    assert not grader._values_close_with_null(mutated, expected, rtol=0, atol=0)


def test_v10_uses_a_private_parent_comparator_namespace():
    def compare(actual, expected, *, rtol, atol):
        return values_close(actual, expected, rtol=rtol, atol=atol)

    null_safe = grader._with_null_safe_comparator(compare)
    expected = {"routes": [None, {"optional": None}]}

    assert compare(None, None, rtol=0, atol=0) is False
    assert null_safe(expected, expected, rtol=0, atol=0) is True
    assert compare(None, None, rtol=0, atol=0) is False
    assert null_safe(None, "forged", rtol=0, atol=0) is False


def test_v10_grade_path_uses_the_null_safe_clone_without_mutating_shared_globals(
    monkeypatch, tmp_path,
):
    observed: dict[str, object] = {}

    def fake_grade_calls(_subject, calls, **_kwargs):
        observed["null_routes"] = sum(
            value is None for value in calls[0].expected["runtime_registry"].values()
        )
        observed["nulls_compare_equal"] = values_close(None, None, rtol=0, atol=0)
        return {"passed": False}

    def fake_phase_a_grade(task_id, _workspace, _image):
        return phase_a.grade_calls(
            tmp_path,
            [FunctionCall(task_id, "adapter", "routes", [], {})],
        )

    monkeypatch.setattr(phase_a, "grade_calls", fake_grade_calls)
    monkeypatch.setattr(phase_a, "grade", fake_phase_a_grade)
    assert grader.grade(grader.TASK_ID, tmp_path, IMAGE) == {"passed": False}
    assert observed == {"null_routes": 10, "nulls_compare_equal": True}
    assert phase_a.grade_calls is fake_grade_calls


def test_v10_cloned_real_parent_grade_calls_preserves_comparator_keywords(
    monkeypatch, tmp_path,
):
    from vare.isolated_candidate import CallReport

    actual = {"routes": [None], "numeric": [True]}
    expected = {"routes": [None], "numeric": [1.0]}
    cloned_grade_calls = grader._with_null_safe_comparator(phase_a.grade_calls)
    monkeypatch.setitem(
        cloned_grade_calls.__globals__, "_one_call",
        lambda *_args, **_kwargs: (actual, 0.01, None),
    )
    call = FunctionCall("interface", "adapter", "routes", [], expected, {})

    report = cloned_grade_calls(
        tmp_path, [call], task_kind="correctness", image=IMAGE, bool_as_number=True
    )
    assert isinstance(report, CallReport)
    assert report.passed is True

    strict_report = cloned_grade_calls(
        tmp_path, [call], task_kind="correctness", image=IMAGE, bool_as_number=False
    )
    assert strict_report.passed is False
    assert strict_report.cases[0].failure == "wrong_output"


def test_v10_protocol_and_implementation_identities_are_locked():
    protocol, lock, protocol_sha = grader._load_protocol()

    assert protocol["protocol_id"] == "vare-coding-agent-phase-a-integrity-v10"
    assert protocol_sha == lock["protocol_sha256"]
    assert protocol["exposure_classification"] == "retrospective-development-consumed"
    expected_paths = {
        "scripts/grade_transformers_phase_a_v1.py",
        "scripts/grade_transformers_phase_a_integrity_v10.py",
        "src/vare/isolated_candidate.py",
        "src/vare/container_isolation.py",
        "benchmarks/mlsys_repo_eval_v4/"
        "transformers_49116_tokenizer_routing_v2_grader_v10/grader_adapter.py",
    }
    assert set(lock["implementation_sha256"]) == expected_paths
    for relative_path, expected_sha in lock["implementation_sha256"].items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected_sha


def test_v10_refuses_other_or_fresh_task_ids(tmp_path: Path):
    try:
        grader.grade("transformers-49412-fusion-registration-idempotence-v2", tmp_path, IMAGE)
    except ValueError as exc:
        assert "only its pinned Development task" in str(exc)
    else:
        raise AssertionError("v10 accepted a task outside its retrospective scope")
