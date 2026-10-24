"""Lossless adaptation of the existing tool controller to Mistral chat roles."""
from scripts.run_flagship_teacher_v1 import plain_mistral_messages


def test_tool_results_and_system_survive_role_adaptation():
    result = plain_mistral_messages([
        {"role": "system", "content": "SYSTEM AND TOOL DEFINITIONS"},
        {"role": "user", "content": "TASK"},
        {"role": "assistant", "content": "TOOL CALL"},
        {"role": "user", "content": "RESULT ONE"},
        {"role": "user", "content": "RESULT TWO"},
    ])
    assert [message["role"] for message in result] == ["user", "assistant", "user"]
    assert result[0]["content"] == "SYSTEM AND TOOL DEFINITIONS\n\nTASK"
    assert result[1]["content"] == "TOOL CALL"
    assert result[2]["content"] == "RESULT ONE\n\nRESULT TWO"
