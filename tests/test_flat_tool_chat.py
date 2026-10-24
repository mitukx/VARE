import pytest

from vare.flat_tool_chat import flatten_tool_messages


TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read one file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}]


def test_flat_template_receives_tool_schema_and_round_trips_tool_history():
    messages = [
        {"role": "system", "content": "Fix the issue."},
        {"role": "user", "content": "Inspect parser.py."},
        {"role": "assistant", "tool_calls": [{
            "type": "function",
            "function": {"name": "read_file", "arguments": {"path": "parser.py"}},
        }]},
        {"role": "tool", "name": "read_file", "content": "line 1: source"},
    ]

    result = flatten_tool_messages(messages, TOOLS)

    assert [item["role"] for item in result] == ["system", "user", "assistant", "user"]
    assert '"name":"read_file"' in result[0]["content"]
    assert '<tool_call>{"arguments":{"path":"parser.py"},"name":"read_file"}</tool_call>' in result[2]["content"]
    assert "Tool result from read_file:\nline 1: source" == result[3]["content"]


def test_flat_template_rejects_unknown_roles_and_malformed_tool_schema():
    with pytest.raises(ValueError, match="unsupported chat role"):
        flatten_tool_messages([{"role": "developer", "content": "x"}], TOOLS)
    with pytest.raises(ValueError, match="named function"):
        flatten_tool_messages([{"role": "system", "content": "x"}], [{"type": "function"}])
