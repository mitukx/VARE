"""Candidate-side request adapters; they contain no checks or expected values."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def entropy_gradients(
    seed: int, rows: int, hidden_size: int, vocab_size: int,
    temperature: float, chunk_size: int,
) -> dict[str, Any]:
    import torch
    from trl.trainer.utils import _ChunkedLogProbFunction

    torch.manual_seed(seed)
    hidden = torch.randn(rows, hidden_size, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (rows,))
    _, entropy = _ChunkedLogProbFunction.apply(
        hidden, weight, labels, temperature, chunk_size,
    )
    entropy.sum().backward()
    return {"hidden": hidden.grad.tolist(), "weight": weight.grad.tolist()}
