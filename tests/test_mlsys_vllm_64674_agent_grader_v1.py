from __future__ import annotations

from typing import Any

import pytest

from scripts.run_mlsys_vllm_64674_agent_grader_v1 import agent_public_test_passed


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        ({}, False),
        ({"public_test_result": None}, False),
        ({"public_test_result": {"passed": False}}, False),
        ({"public_test_result": {"passed": True}}, True),
        ({"public_test_result": ["passed"]}, False),
    ],
)
def test_agent_public_test_passed_requires_explicit_true_mapping(
    attempt: dict[str, Any], expected: bool
) -> None:
    assert agent_public_test_passed(attempt) is expected
