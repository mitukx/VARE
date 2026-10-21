"""Minimal candidate-side RPC worker; contains no expected outputs or test assertions."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def bloom_static_cache(
    config_values: dict[str, Any], tokens: list[list[int]], max_new_tokens: int, seed: int,
) -> dict[str, Any]:
    import torch
    from transformers import BloomConfig, BloomForCausalLM

    torch.manual_seed(seed)
    model = BloomForCausalLM(BloomConfig(**config_values)).eval()
    input_ids = torch.tensor(tokens, dtype=torch.long)
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            cache_implementation="static",
            pad_token_id=0,
        )
    return {"output_ids": output.tolist()}


def phi3_longrope_logits(
    config_values: dict[str, Any], tokens: list[list[int]], seed: int,
) -> dict[str, Any]:
    import torch
    from transformers import Phi3Config, Phi3ForCausalLM

    observations: list[dict[str, Any]] = []
    for sliding_window in (None, 5):
        values = dict(config_values)
        values["sliding_window"] = sliding_window
        torch.manual_seed(seed)
        model = Phi3ForCausalLM(Phi3Config(**values)).eval()
        input_ids = torch.tensor(tokens, dtype=torch.long)
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                max_new_tokens=4,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
            )
            comparisons = []
            for step, cached_logits in enumerate(generated.logits):
                full_prefix = generated.sequences[:, : input_ids.shape[1] + step]
                full_logits = model(full_prefix, use_cache=False).logits[:, -1]
                comparisons.append({
                    "cached": cached_logits.tolist(),
                    "full_prefix": full_logits.tolist(),
                })
        observations.append({"sliding_window": sliding_window, "comparisons": comparisons})
    return {"observations": observations}
