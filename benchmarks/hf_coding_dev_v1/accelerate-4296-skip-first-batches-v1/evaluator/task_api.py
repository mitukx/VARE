"""Candidate-side request adapters; they contain no checks or expected values."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def skip_first_batches_config(
    batch_size: int, drop_last: bool, non_blocking: bool, skipped: int,
) -> list[dict[str, Any]]:
    from accelerate.data_loader import DataLoaderDispatcher, DataLoaderShard, skip_first_batches

    dataset = list(range(batch_size * 4 + 1))
    result: list[dict[str, Any]] = []
    for loader_type in (DataLoaderShard, DataLoaderDispatcher):
        loader = loader_type(
            dataset, batch_size=batch_size,
            _drop_last=drop_last, _non_blocking=non_blocking,
        )
        resumed = skip_first_batches(loader, num_batches=skipped)
        result.append({
            "loader_type": loader_type.__name__,
            "drop_last": resumed._drop_last,
            "non_blocking": resumed._non_blocking,
        })
    return result
