"""T4-compatible, protocol-derived VRAM eligibility and reserved-memory limits."""

ABSOLUTE_RESERVED_VRAM_LIMIT_BYTES = 14 * 1024**3


def maximum_reserved_vram_bytes(total_vram_bytes: int, maximum_fraction: float = 0.90) -> int:
    if total_vram_bytes <= 0:
        raise ValueError("total_vram_bytes must be positive")
    if not 0 < maximum_fraction <= 1:
        raise ValueError("maximum_fraction must be in (0, 1]")
    return min(ABSOLUTE_RESERVED_VRAM_LIMIT_BYTES, int(maximum_fraction * total_vram_bytes))
