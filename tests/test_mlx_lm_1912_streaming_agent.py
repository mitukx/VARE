from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_local_agent_streaming_v1 import instrument_generate_turn


def test_generation_is_returned_and_durably_logged(tmp_path: Path) -> None:
    log = tmp_path / "turns.jsonl"
    log.touch()

    def generate(*_args: object, **_kwargs: object) -> tuple[str, int, int]:
        return ('<tool_call>{"name":"finish"}</tool_call>', 9, 81)

    wrapped = instrument_generate_turn(generate, log)
    result = wrapped("mlx", object(), object(), [], [], limit=16, args=object())

    assert result == ('<tool_call>{"name":"finish"}</tool_call>', 9, 81)
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["event"] == "generation_completed"
    assert record["sequence"] == 1
    assert record["model_output"] == result[0]
    assert record["generated_tokens"] == 9
    assert record["prompt_tokens"] == 81


def test_generation_exception_is_retained_before_propagation(tmp_path: Path) -> None:
    log = tmp_path / "turns.jsonl"
    log.touch()

    def fail(*_args: object, **_kwargs: object) -> tuple[str, int, int]:
        raise ValueError("context budget exhausted")

    wrapped = instrument_generate_turn(fail, log)
    with pytest.raises(ValueError, match="context budget exhausted"):
        wrapped("mlx", object(), object(), [], [], limit=16, args=object())

    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["event"] == "generation_exception"
    assert record["exception_type"] == "ValueError"
    assert record["exception_text"] == "context budget exhausted"
