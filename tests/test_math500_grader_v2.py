import time

import pytest
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

import scripts.math500_grader_v2 as grader
from scripts.math500_grader_v2 import exact_match, has_valid_final_box, parse_prediction


def _trl_1_1_accuracy_reward_semantics(completion: str, solution: str) -> float | None:
    """Mirror the pinned TRL accuracy_reward parse/extract/verify calls."""
    gold = parse(solution)
    if not gold:
        return None
    predicted = parse(
        completion,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(units=True),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    return float(verify(gold, predicted, timeout_seconds=5))


def test_final_box_controls_multiple_box_and_malformed_final_cases():
    assert exact_match(r"work \boxed{2}; answer \boxed{2+0}", "2")
    assert not exact_match(r"work \boxed{42}; answer \boxed{41}", "42")
    assert not has_valid_final_box(r"work \boxed{42}; answer \boxed{41")
    assert not has_valid_final_box(r"work \boxed{42}; answer \boxed{}")


def test_symbolic_equivalence_and_non_equivalence_controls():
    assert exact_match(r"\boxed{\frac{2}{4}}", r"\frac{1}{2}")
    assert exact_match(r"\boxed{\sqrt{12}/2}", r"\sqrt{3}")
    assert not exact_match(r"\boxed{\sqrt{12}/2}", r"\sqrt{2}")
    assert not exact_match(r"\boxed{2.56}", r"2.557")


def test_long_prefix_is_bounded_and_still_grades_short_final_answer():
    started = time.perf_counter()
    assert exact_match("reasoning " * 10_000 + r"\boxed{1/2}", r"\frac{1}{2}")
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0
    assert not has_valid_final_box("x" * (grader.MAX_COMPLETION_CHARS + 1) + r"\boxed{1}")
    assert not has_valid_final_box(r"\boxed{" + "1" * (grader.MAX_FINAL_BOX_CHARS + 1) + "}")


def test_symbolic_parser_exception_fails_closed(monkeypatch):
    monkeypatch.setattr(grader, "latex2sympy", lambda _text: (_ for _ in ()).throw(RuntimeError("bad parser input")))
    assert not exact_match(r"\boxed{1}", "1")


def test_symbolic_parser_timeout_fails_closed(monkeypatch):
    monkeypatch.setattr(grader, "PARSER_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(grader, "latex2sympy", lambda _text: time.sleep(0.2))
    started = time.perf_counter()
    parsed = parse_prediction(r"\boxed{1}")
    assert parsed is not None and parsed.value is None
    assert not exact_match(r"\boxed{1}", "1")
    assert time.perf_counter() - started < 0.5


def test_symbolic_comparison_timeout_fails_closed(monkeypatch):
    monkeypatch.setattr(grader, "PARSER_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(grader, "_same_value", lambda _a, _b: time.sleep(0.2))
    started = time.perf_counter()
    assert not exact_match(r"\boxed{1}", "1")
    assert time.perf_counter() - started < 0.5


def test_expected_reward_disagreement_is_detected_and_not_hidden():
    completion = r"\boxed{90^\circ}"
    solution = r"\boxed{\frac{\pi}{2}}"
    reward = _trl_1_1_accuracy_reward_semantics(completion, solution)
    independent = exact_match(completion, solution)
    assert reward == 0.0
    assert independent is True


def test_reward_and_independent_grader_agree_on_simple_correct_and_wrong_cases():
    for completion, expected in ((r"\boxed{1/2}", True), (r"\boxed{3/4}", False)):
        assert bool(_trl_1_1_accuracy_reward_semantics(completion, r"\frac{1}{2}")) is expected
        assert exact_match(completion, r"\frac{1}{2}") is expected
