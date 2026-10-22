"""Candidate-side request adapters; they contain no checks or expected values."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def split_tuple(values: list[int], num_processes: int, process_index: int) -> dict[str, Any]:
    from accelerate import PartialState

    state = PartialState()
    state.__dict__["num_processes"] = num_processes
    state.__dict__["process_index"] = process_index
    with state.split_between_processes(tuple(values), apply_padding=True) as result:
        return {"container": type(result).__name__, "values": list(result)}
