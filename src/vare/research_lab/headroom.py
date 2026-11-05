"""Accelerator headroom gate.

The gate only reads memory and process command lines. It does not signal,
renice, or otherwise change another process. A passing decision is not
permission to start MLX; this slice has no Metal runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

from vare.learning_pipeline.resources import memory_snapshot


DEFAULT_MARKERS = ("grpo_mlx", "mlx_lm", "mlx.core")


@dataclass(frozen=True, slots=True)
class HeadroomDecision:
    allowed: bool
    reason: str
    available_bytes: int | None
    competing: tuple[str, ...]


def decide_accelerator_headroom(
    *,
    available_bytes: int | None,
    min_available_bytes: int,
    cmdlines: Sequence[str],
    markers: Sequence[str] = DEFAULT_MARKERS,
) -> HeadroomDecision:
    if min_available_bytes <= 0:
        raise ValueError("min_available_bytes must be positive")
    competing = tuple(sorted({line for line in cmdlines if any(marker in line for marker in markers)}))
    if competing:
        return HeadroomDecision(False, "competing_accelerator_job", available_bytes, competing)
    if available_bytes is None or available_bytes < min_available_bytes:
        return HeadroomDecision(False, "insufficient_memory", available_bytes, ())
    return HeadroomDecision(True, "headroom_ok", available_bytes, ())


def local_cmdlines() -> list[str]:
    """Command lines of other processes. This process is excluded."""
    own = str(os.getpid())
    found: list[str] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit() or entry.name == own:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        found.append(raw.replace(b"\x00", b" ").decode("utf-8", "replace"))
    return found


def local_available_bytes() -> int | None:
    snapshot = memory_snapshot()
    available = snapshot.get("available")
    if isinstance(available, int):
        return available
    return None
