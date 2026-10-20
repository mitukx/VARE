"""Hidden short-window checks for gradient accumulation."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    workspace = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(workspace))
    from grad_accum import average_microbatch_grads

    tail = [[[2.0, -4.0], [6.0, 8.0]]]
    snapshot = [row[:] for row in tail[0]]
    averaged = average_microbatch_grads(tail, accumulation_steps=4)
    if averaged != [[2.0, -4.0], [6.0, 8.0]]:
        raise AssertionError(f"short window was scaled by the nominal step count: {averaged}")
    if tail[0] != snapshot:
        raise AssertionError("input gradient was mutated")

    mixed = [
        [[1.0, 1.0, 1.0]],
        [[3.0, 5.0, 7.0]],
        [[5.0, 9.0, 13.0]],
    ]
    mixed_mean = average_microbatch_grads(mixed, accumulation_steps=8)
    if mixed_mean != [[3.0, 5.0, 7.0]]:
        raise AssertionError(mixed_mean)

    try:
        average_microbatch_grads([], 2)
    except ValueError:
        pass
    else:
        raise AssertionError("empty window must fail")


if __name__ == "__main__":
    main()
