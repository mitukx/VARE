from __future__ import annotations

import ast
import os
import sys
import types
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class _QualificationMMPayload(BaseModel):
    pass


def _qualification_return_mm_kwargs_alias(data):
    return data


def _load_anthropic_messages_request():
    source_path = Path(os.environ["VARE_SOURCE_FILE"])
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
            alias.name == "vllm.envs" for alias in node.names
        ):
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "vllm.entrypoints.generate.base.protocol"
        ):
            continue
        body.append(node)
    source_module = ast.Module(body=body, type_ignores=[])
    module = types.ModuleType("vllm_anthropic_protocol_isolated")
    module.__file__ = str(source_path)
    module.envs = SimpleNamespace(VLLM_MAX_STOP_STRINGS=4)
    module.MMPayload = _QualificationMMPayload
    module.resolve_return_mm_kwargs_alias = _qualification_return_mm_kwargs_alias
    sys.modules[module.__name__] = module
    exec(compile(source_module, str(source_path), "exec"), module.__dict__)
    return module.AnthropicMessagesRequest


AnthropicMessagesRequest = _load_anthropic_messages_request()

def _make_request(
    messages: list[dict],
    **kwargs,
) -> AnthropicMessagesRequest:
    return AnthropicMessagesRequest(
        model="test-model",
        max_tokens=128,
        messages=messages,
        **kwargs,
    )

class TestToolDefinitions:
    @pytest.mark.parametrize(
        "tool",
        [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
            {"type": "bash_20250124", "name": "bash"},
        ],
    )
    def test_anthropic_defined_tool_is_rejected_by_type(self, tool):
        with pytest.raises(ValidationError) as exc_info:
            _make_request([{"role": "user", "content": "Hi"}], tools=[tool])

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert f"{tool['type']!r} is not supported" in errors[0]["msg"]

    @pytest.mark.parametrize("extra", [{}, {"type": "custom"}])
    def test_custom_tool_is_accepted(self, extra):
        request = _make_request(
            [{"role": "user", "content": "Hi"}],
            tools=[{"name": "get_weather", "input_schema": {}, **extra}],
        )
        assert request.tools is not None
        assert request.tools[0].input_schema == {"type": "object"}

    def test_custom_tool_still_requires_input_schema(self):
        with pytest.raises(ValidationError, match="input_schema"):
            _make_request(
                [{"role": "user", "content": "Hi"}],
                tools=[{"type": "custom", "name": "get_weather"}],
            )
