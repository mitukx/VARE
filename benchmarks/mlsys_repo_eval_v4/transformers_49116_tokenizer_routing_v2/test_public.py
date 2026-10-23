from __future__ import annotations

import sys

sys.path.insert(0, "/workspace/src")

from transformers.models.auto.tokenization_auto import (  # noqa: E402
    MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS,
    TOKENIZER_MAPPING_NAMES,
)


def main() -> None:
    required = {"qwen3_5_moe_text", "qwen4_exp"}
    missing_override = required - set(MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS)
    wrong_mapping = {
        model_type: TOKENIZER_MAPPING_NAMES.get(model_type)
        for model_type in required
        if TOKENIZER_MAPPING_NAMES.get(model_type) != "Qwen3_5Tokenizer"
    }
    if missing_override or wrong_mapping:
        raise AssertionError({"missing_override": sorted(missing_override), "wrong_mapping": wrong_mapping})


if __name__ == "__main__":
    main()
