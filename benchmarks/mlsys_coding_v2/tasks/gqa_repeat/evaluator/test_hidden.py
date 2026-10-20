"""Hidden head-order checks for grouped-query key/value repeat."""

from __future__ import annotations

import sys
from pathlib import Path

_PACK = Path(__file__).resolve().parents[4] / "mlsys_coding_v1"
if str(_PACK) not in sys.path:
    sys.path.insert(0, str(_PACK))

from harness_adapter import HarnessRequest, run_harness


def _reference(keys, n_rep):
    if type(n_rep) is not int or n_rep < 1:
        raise ValueError("n_rep must be an integer >= 1")
    return [row[:] for row in keys for _ in range(n_rep)]


def _generator(index: int):
    if index == 0:
        return (([[1.0, 2.0], [3.0, 4.0]], 2), {})
    if index == 1:
        return (([[5.0, 6.0, 7.0]], 3), {})
    return (([[1.0, 2.0]], 0), {})


def _checker(actual, expected, positional, _keywords, args, _kwargs):
    if actual != expected:
        raise AssertionError(f"head order diverged: {actual}")
    if len(actual) != len(args[0]) * args[1]:
        raise AssertionError("head count changed")
    if positional[0] != args[0]:
        raise AssertionError("keys were mutated")


def main() -> None:
    report = run_harness(
        HarnessRequest(
            workspace=Path(sys.argv[1]).resolve(),
            entrypoint="gqa_repeat:repeat_kv_heads",
            task_type="correctness",
            trusted_task_path=Path(__file__).resolve().parents[1] / "task.json",
            input_generator=_generator,
            reference=_reference,
            checker=_checker,
            repeats=3,
            timeout_s=10,
        )
    )
    if not report.passed:
        raise SystemExit(report.error or "hidden checks failed")


if __name__ == "__main__":
    main()
