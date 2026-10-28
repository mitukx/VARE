from __future__ import annotations

import pytest

from vare.agent_json_eos_v1 import has_complete_json_object


@pytest.mark.parametrize("raw", [
    '{"name":"finish","arguments":{"summary":"done"}}',
    '```json\n{"name":"finish","arguments":{"summary":"done"}}',
    '<|im_start|>\n{"name":"finish","arguments":{"summary":"done"}}',
    '{"name":"finish","arguments":{"summary":"done"}}<|im_start|>',
    '{"name":"finish","arguments":{"summary":"done"}}<|im_end|>',
])
def test_detects_complete_json_object_at_parser_safe_boundary(raw: str) -> None:
    assert has_complete_json_object(raw)


@pytest.mark.parametrize("raw", [
    "",
    '{"name":"finish","arguments":',
    '["not", "an object"]',
    '{"name":"finish","arguments":{}} additional prose',
    '{"name":"finish","arguments":{}}<|endoftext|>additional prose',
])
def test_does_not_terminate_on_incomplete_or_unsafe_output(raw: str) -> None:
    assert not has_complete_json_object(raw)
