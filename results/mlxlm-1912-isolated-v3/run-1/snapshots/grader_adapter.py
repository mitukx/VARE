"""Candidate-visible invocation bridge; all grading predicates stay trusted."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def run(case: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx

    target = Path("/subject/mlx_lm/sample_utils.py")
    spec = importlib.util.spec_from_file_location("vare_candidate_sample_utils", target)
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    apply_top_p = module.apply_top_p

    logits = [
        [float("-inf") if value is None else float(value) for value in row]
        for row in case["logits"]
    ]
    values = mx.array(logits)
    if case.get("dtype"):
        values = values.astype(getattr(mx, case["dtype"]))
    filtered = apply_top_p(values, float(case["top_p"]))
    rows = filtered.tolist()
    kept_indices = [
        [index for index, value in enumerate(row) if value != float("-inf")]
        for row in rows
    ]
    kept_values = [
        [row[index] for index in indices]
        for row, indices in zip(rows, kept_indices, strict=True)
    ]
    return {"kept_indices": kept_indices, "kept_values": kept_values}
