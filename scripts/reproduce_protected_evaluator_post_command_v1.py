#!/usr/bin/env python3
"""Reproduce protected-file integrity checks around test and metric commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

from vare.environments import (
    CommandSpec,
    EngineeringTaskSpec,
    ExecutableEvaluator,
    MetricSpec,
    ProtectedPath,
    RepositorySource,
    ResourceLimits,
)


LIMITS = ResourceLimits(wall_time_s=3, cpu_time_s=None, memory_mb=None, max_output_bytes=4096)


def _spec(
    protected_name: str,
    original: bytes,
    tests: tuple[CommandSpec, ...],
    metrics: tuple[MetricSpec, ...] = (),
) -> EngineeringTaskSpec:
    return EngineeringTaskSpec(
        id="protected-post-command-integrity-v1",
        title="Recheck protected evaluator inputs after commands",
        prompt="Keep protected evaluation inputs unchanged.",
        family="integrity",
        source=RepositorySource(local_path="."),
        tests=tests,
        metrics=metrics,
        protected_paths=(
            ProtectedPath(
                path=protected_name,
                sha256=hashlib.sha256(original).hexdigest(),
            ),
        ),
        limits=LIMITS,
    )


def reproduce() -> dict[str, object]:
    records: dict[str, object] = {"python": sys.version}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        original = b"original evaluator bytes\n"
        (root / "protected-test.py").write_bytes(original)
        mutation = (
            "from pathlib import Path; "
            "Path('protected-test.py').write_bytes(b'changed evaluator bytes\\n')"
        )
        result = ExecutableEvaluator(
            _spec(
                "protected-test.py",
                original,
                (CommandSpec(argv=(sys.executable, "-c", mutation), name="mutate-protected-test"),),
            )
        ).evaluate(root)
        records["test_mutation"] = {
            "passed": result.passed,
            "integrity_ok": result.integrity_ok,
            "score": result.score,
            "test_commands_passed": [command.passed for command in result.tests],
            "integrity_failures": result.integrity_failures,
        }
        assert not result.passed and not result.integrity_ok and result.score == 0.0
        assert any("protected file modified" in item for item in result.integrity_failures)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        original = b"original metric bytes\n"
        (root / "protected-metric.py").write_bytes(original)
        later_metric_marker = root / "later-metric-ran"
        mutation = (
            "from pathlib import Path; import json; "
            "Path('protected-metric.py').write_bytes(b'changed metric bytes\\n'); "
            "print(json.dumps({'work_units': 1}))"
        )
        later_metric = (
            f"from pathlib import Path; Path({str(later_metric_marker)!r}).write_text('ran'); "
            "print('{\"work_units\":1}')"
        )
        metrics = (
            MetricSpec(
                name="work_units",
                command=CommandSpec(argv=(sys.executable, "-c", mutation), name="mutate-protected-metric"),
                json_key="work_units",
            ),
            MetricSpec(
                name="later_metric",
                command=CommandSpec(argv=(sys.executable, "-c", later_metric), name="later-metric"),
                json_key="work_units",
            ),
        )
        result = ExecutableEvaluator(
            _spec(
                "protected-metric.py",
                original,
                (CommandSpec(argv=(sys.executable, "-c", "pass"), name="passing-test"),),
                metrics,
            )
        ).evaluate(root, baseline_metrics={"work_units": 1.0, "later_metric": 1.0})
        records["metric_mutation"] = {
            "passed": result.passed,
            "integrity_ok": result.integrity_ok,
            "score": result.score,
            "metric_count": len(result.metrics),
            "later_metric_executed": later_metric_marker.exists(),
            "first_metric_is_nan": bool(result.metrics and math.isnan(result.metrics[0].value)),
            "integrity_failures": result.integrity_failures,
        }
        assert not result.passed and not result.integrity_ok and result.score == 0.0
        assert len(result.metrics) == 1 and math.isnan(result.metrics[0].value)
        assert not later_metric_marker.exists()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        original = b"unchanged evaluator bytes\n"
        (root / "protected-test.py").write_bytes(original)
        result = ExecutableEvaluator(
            _spec(
                "protected-test.py",
                original,
                (CommandSpec(argv=(sys.executable, "-c", "pass"), name="passing-test"),),
            )
        ).evaluate(root)
        records["untampered_control"] = {
            "passed": result.passed,
            "integrity_ok": result.integrity_ok,
            "score": result.score,
        }
        assert result.passed and result.integrity_ok and result.score > 0.0

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = reproduce()
    payload = json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
