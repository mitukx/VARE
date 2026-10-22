"""Candidate-side request adapters; they contain no checks or expected values."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def multi_cpu_barrier(local_process_index: int) -> list[dict[str, Any]]:
    import torch.distributed
    from accelerate import PartialState
    from accelerate.utils import DistributedType

    state = PartialState()
    state.__dict__["distributed_type"] = DistributedType.MULTI_CPU
    state.__dict__["local_process_index"] = local_process_index
    calls: list[dict[str, Any]] = []
    original = torch.distributed.barrier
    torch.distributed.barrier = lambda *args, **kwargs: calls.append({
        "args": list(args), "kwargs": kwargs,
    })
    try:
        state.wait_for_everyone()
    finally:
        torch.distributed.barrier = original
    return calls
