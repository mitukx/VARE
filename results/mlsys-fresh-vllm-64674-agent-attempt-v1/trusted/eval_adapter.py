"""Candidate-side adapter. It contains no test oracle or expected values."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import candidate_module as candidate


def stream_payloads(batches: list[list[str]]) -> list[list[str]]:
    observed: list[list[str]] = []
    for chunks in batches:
        state = SimpleNamespace(
            accumulated_text="",
            tool_call_payload="",
            tool_call_payload_end=0,
            tool_call_is_custom=True,
            output_index=0,
            current_item_id="hidden-item",
        )
        values: list[str] = []
        for chunk in chunks:
            candidate.emit_simple_tool_call_delta(state, chunk)
            values.append(state.tool_call_payload)
        observed.append(values)
    return observed


def reparse_count_pair(short_chunks: list[str], long_chunks: list[str]) -> bool:
    parser = candidate.partial_json_parser
    original = parser.loads

    def count(chunks: list[str]) -> int:
        calls = 0

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        parser.loads = wrapped
        try:
            state = SimpleNamespace(
                accumulated_text="",
                tool_call_payload="",
                tool_call_payload_end=0,
                tool_call_is_custom=True,
                output_index=0,
                current_item_id="count-item",
            )
            for chunk in chunks:
                candidate.emit_simple_tool_call_delta(state, chunk)
        finally:
            parser.loads = original
        return calls

    return count(short_chunks) == count(long_chunks)
