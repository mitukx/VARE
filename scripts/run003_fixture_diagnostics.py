#!/usr/bin/env python3
"""Run count-only Run-003 diagnostics on committed public pytest fixtures."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import run003_failure_forensics as ff

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "run003_failure_forensics"
DEFAULT_OUTPUT = ROOT / "results" / "run003-failure-forensics-public" / "report.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pytest_version(python: str) -> str:
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="")
    result = subprocess.run([python, "-m", "pytest", "--version"], capture_output=True,
                            text=True, timeout=30, check=True, env=env)
    return (result.stdout or result.stderr).strip().splitlines()[0]


def python_version(python: str) -> str:
    result = subprocess.run([python, "-c", "import platform; print(platform.python_version())"],
                            capture_output=True, text=True, timeout=30, check=True)
    return result.stdout.strip()


def run_case(label: str, python: str, report_dir: Path) -> dict[str, object]:
    fixture = FIXTURES / label
    junit = report_dir / f"{label}.xml"
    if label == "requested_node_absent":
        target = str(FIXTURES / "passed" / "test_case.py") + "::test_missing"
        requested = 1
    elif label == "no_tests_collected":
        target = str(FIXTURES / "no_tests")
        requested = 0
    elif label == "supervisor_timeout":
        target = str(FIXTURES / "timeout")
        requested = 1
    else:
        target = str(fixture)
        requested = 1
    command = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--junitxml", str(junit), target]
    env = dict(os.environ)
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="")
    started = time.perf_counter()
    timed_out = False
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                timeout=0.25 if label == "supervisor_timeout" else 30)
        exit_code: int | None = result.returncode
        stdout = result.stdout + result.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = None
        stdout = (exc.stdout or b"") + (exc.stderr or b"")
    elapsed = time.perf_counter() - started

    summary = None
    junit_ok = True
    if not timed_out:
        try:
            summary = ff.summarize_junit(junit)
        except (OSError, ValueError):
            junit_ok = False
    else:
        junit.unlink(missing_ok=True)

    # Exact absent-node evidence is supplied only for this controlled fixture:
    # the requested function is intentionally absent from the known fixture file.
    absent = int(label == "requested_node_absent")
    observation = ff.Observation(
        environment_constructed=True,
        sandbox_violation=False,
        source_integrity_ok=True,
        timed_out=timed_out,
        pytest_exit=exit_code,
        requested_count=requested,
        collected_count=summary.collected_count if summary else 0,
        passed_count=summary.passed_count if summary else 0,
        failed_count=summary.failed_count if summary else 0,
        skipped_count=summary.skipped_count if summary else 0,
        collection_error_count=summary.collection_error_count if summary else 0,
        requested_absent_count=absent,
        junit_integrity_ok=junit_ok,
    )
    record = ff.sanitized_record(observation, legacy_reason=None,
                                 junit_sha256=summary.sha256 if summary else None)
    if label == "requested_node_absent":
        source_file = FIXTURES / "passed" / "test_case.py"
    elif label == "no_tests_collected":
        source_file = FIXTURES / "no_tests" / "test_placeholder.py"
    elif label == "supervisor_timeout":
        source_file = FIXTURES / "timeout" / "test_case.py"
    else:
        source_file = next(iter(fixture.glob("test_*.py")), None)
    record.update({
        "fixture": label,
        "wall_s": round(elapsed, 3),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "fixture_source_sha256": sha256_file(source_file) if source_file else None,
    })
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter with pytest installed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    labels = ["passed", "failed", "import_error", "no_tests_collected",
              "requested_node_absent", "supervisor_timeout"]
    with tempfile.TemporaryDirectory(prefix="vare-run003-junit-") as temp:
        temp_path = Path(temp)
        records = [run_case(label, args.python, temp_path) for label in labels]
    report = {
        "schema": "vare.run003.public_fixture_report.v1",
        "purpose": "public synthetic fixtures validate diagnostic distinctions; no repository or held-out tasks were read",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "system": platform.system(), "machine": platform.machine(),
            "python": python_version(args.python), "python_executable_name": Path(args.python).name,
            "pytest": pytest_version(args.python),
        },
        "diagnostic_source_sha256": sha256_file(ROOT / "scripts" / "run003_failure_forensics.py"),
        "fixture_runner_sha256": sha256_file(Path(__file__)),
        "records": records,
        "aggregate": ff.aggregate(records),
        "eligibility_effect": "none",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_out = args.output.with_suffix(args.output.suffix + ".tmp")
    temp_out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp_out.replace(args.output)
    print(json.dumps({"output": str(args.output), "aggregate": report["aggregate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
