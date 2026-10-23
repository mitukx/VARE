"""Candidate-side probe; expected tokenizer routes stay in the parent grader."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, "/subject/src")


def routes(request: dict[str, Any]) -> dict[str, Any]:
    from transformers.models.auto import tokenization_auto

    tokenizer_mapping_names: Any = getattr(tokenization_auto, "TOKENIZER_MAPPING_NAMES")
    override_registry: Any = getattr(
        tokenization_auto, "MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS", None
    )
    if not isinstance(override_registry, (set, frozenset)):
        raise ValueError("runtime tokenizer override registry is missing")

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
        ) and isinstance(node.value, ast.Call) and node.value.args and isinstance(
            node.value.args[0], (ast.List, ast.Tuple)
        ):
            registry = {
                ast.literal_eval(item.elts[0]): ast.dump(item.elts[1], include_attributes=False)
                for item in node.value.args[0].elts
                if isinstance(item, (ast.List, ast.Tuple)) and len(item.elts) == 2
            }
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS"
            and node.value is not None
        ):
            override_types = sorted(ast.literal_eval(node.value))
    return {
        "routes": {
            name: tokenizer_mapping_names.get(name) or "<unregistered>"
            for name in model_types
        },
        "override_types": sorted(set(model_types) & set(override_registry)),
        # Return source and runtime structures as data. The host-owned parent
        # compares them to its pinned expectation; no candidate-process hash
        # or JSON primitive participates in the integrity decision.
        "registry_ast": registry,
        "all_override_types": override_types,
        "runtime_registry": dict(tokenizer_mapping_names),
        "runtime_override_types": sorted(override_registry),
    }
