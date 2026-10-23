"""Candidate-side probe; expected tokenizer routes stay in the parent grader."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, "/subject/src")


def routes(request: dict[str, Any]) -> dict[str, Any]:
    from transformers.models.auto.tokenization_auto import (
        MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS,
        TOKENIZER_MAPPING_NAMES,
    )

    model_types = request["model_types"]
    source = Path("/subject/src/transformers/models/auto/tokenization_auto.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    registry: dict[str, str] = {}
    override_types: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TOKENIZER_MAPPING_NAMES"
            for target in node.targets
        ):
            registry = {
                ast.literal_eval(item.elts[0]): ast.dump(item.elts[1], include_attributes=False)
                for item in node.value.args[0].elts
            }
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS"
        ):
            override_types = sorted(ast.literal_eval(node.value))
    return {
        "routes": {
            name: TOKENIZER_MAPPING_NAMES.get(name) or "<unregistered>"
            for name in model_types
        },
        "override_types": sorted(set(model_types) & set(MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS)),
        "registry_ast_sha256": hashlib.sha256(
            json.dumps(registry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "all_override_types_sha256": hashlib.sha256(
            json.dumps(override_types, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "runtime_registry_sha256": hashlib.sha256(
            json.dumps(dict(TOKENIZER_MAPPING_NAMES), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "runtime_override_types_sha256": hashlib.sha256(
            json.dumps(sorted(MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
