"""Candidate-visible public workload adapter; contains no expected answers."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


def run(case: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import BertTokenizer, DataCollatorWithPadding

    q_len = int(case["q_len"])
    kv_len = int(case.get("kv_len", q_len))
    vocab_dir = Path(tempfile.mkdtemp(prefix="vare-bert-vocab-"))
    vocab_file = vocab_dir / "vocab.txt"
    vocab_file.write_text("[UNK]\n[CLS]\n[SEP]\n[PAD]\n[MASK]\n", encoding="utf-8")
    tokenizer = BertTokenizer(str(vocab_file))
    features = []
    for value in case["values"]:
        shape = (1, q_len, kv_len)
        if case["input_kind"] == "torch":
            mask = torch.full(shape, float(value), dtype=torch.float32)
        elif case["input_kind"] == "numpy":
            mask = np.full(shape, float(value), dtype=np.float32)
        else:
            raise ValueError("unsupported input kind")
        features.append({"input_ids": list(range(q_len)), "attention_mask": mask})
    batch = DataCollatorWithPadding(tokenizer)(features)
    mask = batch["attention_mask"]
    return {"shape": list(mask.shape), "dtype": str(mask.dtype), "values": mask.tolist()}
