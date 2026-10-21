from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from transformers import BloomConfig, BloomForCausalLM


def grade():
    torch.manual_seed(11)
    config = BloomConfig(vocab_size=31, hidden_size=16, n_layer=1, n_head=2)
    model = BloomForCausalLM(config).eval()
    tokens = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        output = model.generate(
            tokens,
            max_new_tokens=3,
            do_sample=False,
            cache_implementation="static",
            pad_token_id=0,
        )
    return bool(output.shape == (1, 6) and torch.equal(output[:, :3], tokens))
