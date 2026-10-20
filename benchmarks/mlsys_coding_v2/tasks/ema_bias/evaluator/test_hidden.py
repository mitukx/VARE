"""Hidden checks for EMA bias correction."""

from __future__ import annotations

import math
import sys
from pathlib import Path

_PACK = Path(__file__).resolve().parents[4] / "mlsys_coding_v1"
if str(_PACK) not in sys.path:
    sys.path.insert(0, str(_PACK))

from harness_adapter import HarnessRequest, run_harness


def _reference(previous, observation, beta, step):
    if type(step) is not int or step < 1 or not 0.0 <= float(beta) < 1.0:
        raise ValueError("invalid ema arguments")
    raw = float(beta) * float(previous) + (1.0 - float(beta)) * float(observation)
    return raw / (1.0 - float(beta) ** step)


def _generator(index: int):
    cases = (
        (0.0, 10.0, 0.9, 1),
        (1.0, 10.0, 0.9, 2),
        (0.0, 4.0, 0.0, 1),
        (1.0, 1.0, 0.5, 0),
    )
    return (cases[index], {})


def _checker(actual, expected, _positional, _keywords, _args, _kwargs):
    if not math.isfinite(actual) or abs(actual - expected) > 1e-12:
        raise AssertionError(f"{actual} != {expected}")


def main() -> None:
    report = run_harness(
        HarnessRequest(
            workspace=Path(sys.argv[1]).resolve(),
            entrypoint="ema_bias:bias_corrected_ema",
            task_type="correctness",
            trusted_task_path=Path(__file__).resolve().parents[1] / "task.json",
            input_generator=_generator,
            reference=_reference,
            checker=_checker,
            repeats=4,
            timeout_s=10,
        )
    )
    if not report.passed:
        raise SystemExit(report.error or "hidden checks failed")


if __name__ == "__main__":
    main()
