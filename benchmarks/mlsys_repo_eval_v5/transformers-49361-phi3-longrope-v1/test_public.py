from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from transformers import Phi3Config, Phi3ForCausalLM


def check(window):
    config = Phi3Config(
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        original_max_position_embeddings=8,
        max_position_embeddings=32,
        sliding_window=window,
        eos_token_id=None,
        pad_token_id=0,
        rope_parameters={
            "rope_type": "longrope",
            "rope_theta": 10000.0,
            "short_factor": [1.0] * 4,
            "long_factor": [2.0] * 4,
            "original_max_position_embeddings": 8,
            "factor": 4.0,
        },
    )
    model = Phi3ForCausalLM(config).eval()
    prefix = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    with torch.inference_mode():
        generated = model.generate(
            prefix,
            max_new_tokens=4,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )
        for step, logits in enumerate(generated.logits):
            full_prefix = generated.sequences[:, : prefix.shape[1] + step]
            expected = model(full_prefix, use_cache=False).logits[:, -1]
            torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)


for sliding_window in (None, 5):
    check(sliding_window)
