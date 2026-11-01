"""Narrow, network-free public regression for vLLM #64674."""
from __future__ import annotations

import ast
import json
import random
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
import partial_json_parser  # noqa: E402
from partial_json_parser.core.options import Allow  # noqa: E402


def _top_level_bindings(tree: ast.Module) -> dict[str, ast.AST]:
    bindings: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node
    return bindings


def _load_relevant_definitions(path: Path, names: set[str], namespace: dict[str, Any]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bindings = _top_level_bindings(tree)
    pending = list(names)
    selected: dict[int, ast.AST] = {}
    while pending:
        name = pending.pop()
        node = bindings.get(name)
        if node is None or id(node) in selected:
            continue
        selected[id(node)] = node
        pending.extend(
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name)
            and child.id in bindings
            and child.id not in namespace
        )
    ordered = [node for node in tree.body if id(node) in selected]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *ordered,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def _load_agent_functions() -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "Any": Any,
        "json": json,
        "partial_json_parser": partial_json_parser,
        "Allow": Allow,
        "CUSTOM_TOOL_INPUT_KEY": "input",
        "SimpleStreamingState": object,
        "StreamingResponsesResponse": object,
        "ResponseCustomToolCallInputDeltaEvent": lambda **kwargs: kwargs,
        "ResponseFunctionCallArgumentsDeltaEvent": lambda **kwargs: kwargs,
    }
    _load_relevant_definitions(
        ROOT / "vllm/entrypoints/openai/responses/utils.py",
        {
            "encode_custom_tool_input",
            "_custom_tool_payload",
            "decode_custom_tool_input_prefix",
            "extend_custom_tool_input_prefix",
            "_CUSTOM_TOOL_INPUT_HEAD",
        },
        namespace,
    )
    _load_relevant_definitions(
        ROOT / "vllm/entrypoints/openai/responses/streaming_events.py",
        {
            "_emit_custom_tool_input_delta",
            "_custom_tool_input_prefix",
            "emit_simple_tool_call_delta",
        },
        namespace,
    )
    return namespace


def _stream(arguments: list[str], namespace: dict[str, Any]) -> list[types.SimpleNamespace]:
    state = types.SimpleNamespace(
        accumulated_text="",
        tool_call_payload="",
        tool_call_payload_end=0,
        tool_call_is_custom=True,
        output_index=0,
        current_item_id="test-item",
    )
    states = []
    for chunk in arguments:
        namespace["emit_simple_tool_call_delta"](state, chunk)
        states.append(types.SimpleNamespace(**vars(state)))
    return states


def test_streamed_payload_matches_prefix_decode() -> None:
    namespace = _load_agent_functions()
    rng = random.Random(0)
    alphabet = ["a", " ", '"', "\\", "/", "\n", "\t", "\x01", "\x1f", "\x85"]
    alphabet += ["\u00e9", "\u3000", "\U0001f600"]
    for ensure_ascii in (False, True):
        for _ in range(200):
            payload = "".join(rng.choices(alphabet, k=rng.randint(0, 40)))
            arguments = json.dumps({"input": payload}, ensure_ascii=ensure_ascii)
            cuts = sorted(
                rng.sample(range(1, len(arguments)), k=len(arguments) // 3)
            )
            chunks = [
                arguments[i:j]
                for i, j in zip([0, *cuts], [*cuts, None], strict=True)
            ]
            for state in _stream(chunks, namespace):
                expected = namespace["decode_custom_tool_input_prefix"](
                    state.accumulated_text
                )
                assert state.tool_call_payload == expected


def test_payload_deltas_do_not_reparse_the_input() -> None:
    namespace = _load_agent_functions()
    parser = namespace["partial_json_parser"]
    original_loads = parser.loads
    calls = 0

    def counting_loads(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original_loads(*args, **kwargs)

    parser.loads = counting_loads
    try:
        def full_decodes(lines: int) -> int:
            nonlocal calls
            calls = 0
            arguments = json.dumps({"input": 'echo "a > b"\n' * lines})
            chunks = [arguments[i : i + 4] for i in range(0, len(arguments), 4)]
            for _ in _stream(chunks, namespace):
                pass
            return calls

        assert full_decodes(lines=10) == full_decodes(lines=100)
    finally:
        parser.loads = original_loads


if __name__ == "__main__":
    test_streamed_payload_matches_prefix_decode()
    test_payload_deltas_do_not_reparse_the_input()
    print("2 vLLM #64674 public regression contracts passed")
