"""Render tool conversations for chat templates that do not support tools."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def flatten_tool_messages(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Convert structured tool history into system/user/assistant chat messages.

    Some small instruction checkpoints ship a plain role/content template and
    silently discard the tool schema. The explicit interface and flattened
    history keep the existing strict tool-call wire format usable with those
    tokenizers.
    """
    descriptions: list[str] = []
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
            raise ValueError("tool schema must contain a named function")
        descriptions.append(json.dumps({
            "name": function["name"],
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    interface = (
        "\nAVAILABLE TOOLS\n"
        "To call a tool, emit exactly one line in this form: "
        '<tool_call>{"name":"tool_name","arguments":{}}</tool_call>\n'
        "Use only a listed tool and its declared arguments. Tool results will "
        "arrive in the next user message.\n"
        + "\n".join(descriptions)
    )

    flattened: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role in {"system", "user", "assistant"}:
            content = message.get("content", "")
            if not isinstance(content, str):
                raise ValueError("chat message content must be text")
            if role == "system":
                content += interface
            elif role == "assistant" and message.get("tool_calls"):
                content = "\n".join(_format_tool_call(item) for item in message["tool_calls"])
            flattened.append({"role": role, "content": content})
        elif role == "tool":
            name = message.get("name", "unknown")
            content = message.get("content", "")
            if not isinstance(name, str) or not isinstance(content, str):
                raise ValueError("tool result name and content must be text")
            flattened.append({
                "role": "user",
                "content": f"Tool result from {name}:\n{content}",
            })
        else:
            raise ValueError(f"unsupported chat role: {role!r}")
    if not flattened or flattened[0]["role"] != "system":
        raise ValueError("tool chat must begin with a system message")
    return flattened


def _format_tool_call(call: Mapping[str, Any]) -> str:
    function = call.get("function")
    if not isinstance(function, Mapping):
        raise ValueError("tool call must contain a function")
    name = function.get("name")
    arguments = function.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, (dict, list)):
        raise ValueError("tool call name or arguments are invalid")
    return "<tool_call>" + json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ) + "</tool_call>"
