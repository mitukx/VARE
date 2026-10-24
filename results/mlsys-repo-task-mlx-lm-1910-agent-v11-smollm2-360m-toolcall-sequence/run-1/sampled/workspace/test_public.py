#!/usr/bin/env python3
"""Unchanged public regression from mlx-lm PR #1910."""
from qwen3_coder import parse_tool_call

schema = {"anyOf": [{"type": "object"}, {"type": "array"}, {"type": "string"}]}
tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"value": schema}}}}]
cases = [
    ('{"kind": "new"}', {"kind": "new"}),
    ('["a", "b"]', ["a", "b"]),
    ("plain text", "plain text"),
    ('["unfinished"', '["unfinished"'),
    ("123", "123"),
    ('"quoted"', '"quoted"'),
]
for raw, expected in cases:
    call = f"<function=f><parameter=value>{raw}</parameter></function>"
    actual = parse_tool_call(call, tools)["arguments"]
    assert actual == {"value": expected}, (raw, expected, actual)
print(f"PASS: {len(cases)} public regression cases")
