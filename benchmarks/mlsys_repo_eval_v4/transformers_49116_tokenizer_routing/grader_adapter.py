"""Candidate-side probe; expected tokenizer routes stay in the parent grader."""
from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "/subject/src")


def routes(request: dict[str, Any]) -> dict[str, Any]:
    from transformers.models.auto.tokenization_auto import (
        MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS,
        TOKENIZER_MAPPING_NAMES,
    )

    model_types = request["model_types"]
    return {
        "routes": {
            name: TOKENIZER_MAPPING_NAMES.get(name) or "<unregistered>"
            for name in model_types
        },
        "override_types": sorted(set(model_types) & set(MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS)),
    }
