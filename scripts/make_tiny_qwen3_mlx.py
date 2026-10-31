#!/usr/bin/env python3
"""Create a tiny, randomly initialized, 4-bit Qwen3-architecture MLX model.

SYNTHETIC: used only to exercise the generate -> LoRA train -> adapter reload
-> evaluate plumbing on machines without the real model. Its outputs are
noise and are never evidence. The tokenizer files are copied from the pinned
real model revision so prompt-length handling matches.

  python scripts/make_tiny_qwen3_mlx.py OUT_DIR
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

REAL = ("mlx-community/Qwen3-4B-Instruct-2507-4bit", "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b")
TOKENIZER_FILES = ["tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
                   "generation_config.json", "chat_template.jinja"]


def main(out: str) -> None:
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm.models import qwen3
    from mlx_lm.utils import quantize_model, save_config, save_model

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(snapshot_download(REAL[0], revision=REAL[1], allow_patterns=TOKENIZER_FILES))
    for f in src.iterdir():
        shutil.copy(f, out_dir / f.name)
    cfg = {
        "model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "rms_norm_eps": 1e-6,
        "vocab_size": 151936, "max_position_embeddings": 8192, "rope_theta": 5000000.0,
        "tie_word_embeddings": True, "bos_token_id": 151643, "eos_token_id": 151645,
        "synthetic": True, "synthetic_note": "random weights; plumbing tests only",
    }
    mx.random.seed(0)
    args = qwen3.ModelArgs.from_dict(cfg)
    model = qwen3.Model(args)
    mx.eval(model.parameters())
    model, cfg = quantize_model(model, cfg, group_size=64, bits=4)
    save_model(out_dir, model)
    save_config(cfg, out_dir / "config.json")
    print(json.dumps({"out": str(out_dir), "files": sorted(p.name for p in out_dir.iterdir())}))


if __name__ == "__main__":
    main(sys.argv[1])
