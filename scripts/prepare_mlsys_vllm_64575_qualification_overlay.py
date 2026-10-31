#!/usr/bin/env python3
"""Extract the upstream Anthropic tool-definition regression into a small, trusted overlay."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import TypeVar

T = TypeVar("T", ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

SCAFFOLD = '''from __future__ import annotations

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

'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_source_segment(source: str, name: str, node_type: type[T]) -> str:
    tree = ast.parse(source)
    matches = [
        node for node in tree.body
        if isinstance(node, node_type) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one top-level {name!r} definition")
    segment = ast.get_source_segment(source, matches[0])
    if segment is None:
        raise ValueError(f"cannot extract source for {name!r}")
    return segment


def build_overlay(source: str) -> tuple[str, dict[str, str]]:
    helper = extract_source_segment(source, "_make_request", ast.FunctionDef)
    test_class = extract_source_segment(source, "TestToolDefinitions", ast.ClassDef)
    overlay = f"{SCAFFOLD}{helper}\n\n{test_class}\n"
    ast.parse(overlay)
    hashes = {
        "helper_sha256": sha256_bytes(helper.encode("utf-8")),
        "test_class_sha256": sha256_bytes(test_class.encode("utf-8")),
        "overlay_sha256": sha256_bytes(overlay.encode("utf-8")),
    }
    return overlay, hashes


def prepare(source_test: Path, output_dir: Path) -> dict[str, object]:
    source_test = source_test.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    source = source_test.read_text(encoding="utf-8")
    overlay, extracted_hashes = build_overlay(source)
    output_dir.mkdir(parents=True)
    upstream_copy = output_dir / "upstream_test_anthropic_messages_conversion.py"
    upstream_copy.write_bytes(source_test.read_bytes())
    overlay_path = output_dir / "test_anthropic_tool_definitions.py"
    overlay_path.write_text(overlay, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "upstream_test_path": str(source_test),
        "upstream_test_sha256": sha256_bytes(source_test.read_bytes()),
        "extracted_helper_sha256": extracted_hashes["helper_sha256"],
        "extracted_test_class_sha256": extracted_hashes["test_class_sha256"],
        "files": {
            upstream_copy.name: {
                "bytes": upstream_copy.stat().st_size,
                "sha256": sha256_bytes(upstream_copy.read_bytes()),
            },
            overlay_path.name: {
                "bytes": overlay_path.stat().st_size,
                "sha256": extracted_hashes["overlay_sha256"],
            },
        },
        "excluded_integration_test": {
            "name": "test_anthropic_server_tool_is_rejected",
            "reason": "Requires the repository server/client integration fixture; this CPU qualification isolates the request-model regression class.",
        },
    }
    manifest_path = output_dir / "overlay-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "output_dir": str(output_dir),
        "upstream_test_sha256": manifest["upstream_test_sha256"],
        "overlay_sha256": extracted_hashes["overlay_sha256"],
        "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_test, args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
