"""Isolate the exact upstream pure-function entrypoint for this task."""
import ast
import json
from pathlib import Path


def run(messages):
    source = Path("/subject/mlx_lm/server.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    definitions = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "process_message_content"
    ]
    if len(definitions) != 1 or definitions[0].decorator_list:
        raise ValueError("unsupported entrypoint shape")
    isolated = ast.Module(body=[definitions[0]], type_ignores=[])
    namespace = {"json": json}
    exec(compile(isolated, "<candidate-entrypoint>", "exec"), namespace)
    namespace["process_message_content"](messages)
    return messages
