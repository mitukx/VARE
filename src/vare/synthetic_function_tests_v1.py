"""Bounded execution of small synthetic function tests in a separate interpreter.

Used by the v9/v10 multi-turn controller benchmark for candidate-visible
public-test feedback and by its separate grader for grader-only checks. The
candidate module and the test functions run in an isolated (``-I``) child
interpreter with a CPU limit, wall-clock timeout, empty environment, and a
temporary working directory. This is process isolation for synthetic fixtures,
not a container boundary for untrusted upstream repositories.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MAX_DETAIL_CHARS = 200
_RUNNER = r'''
import importlib.util, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
results = []
try:
    spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
except BaseException as exc:
    print(json.dumps({"import_error": f"{type(exc).__name__}: {exc}"[:200]}))
    raise SystemExit(0)
namespace = {"candidate": module}
exec(compile(open(sys.argv[2], encoding="utf-8").read(), "tests.py", "exec"), namespace)
for name in sorted(key for key in namespace if key.startswith("test_")):
    try:
        namespace[name]()
        results.append({"name": name, "outcome": "passed"})
    except AssertionError as exc:
        results.append({"name": name, "outcome": "failed", "detail": f"AssertionError: {exc}"[:200]})
    except BaseException as exc:
        results.append({"name": name, "outcome": "error", "detail": f"{type(exc).__name__}: {exc}"[:200]})
print(json.dumps({"tests": results}))
'''


def run_function_tests(source_text: str, test_source: str, *, timeout_s: float = 20.0) -> dict[str, Any]:
    """Run every ``test_*`` function against ``source_text``; fail closed on any harness error."""
    with tempfile.TemporaryDirectory(prefix="vare-synthetic-tests-") as temp:
        root = Path(temp)
        candidate = root / "candidate.py"
        tests = root / "tests.py"
        candidate.write_text(source_text, encoding="utf-8")
        tests.write_text(test_source, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER, str(candidate), str(tests)],
                cwd=root, env={"PATH": "/usr/bin:/bin"}, capture_output=True,
                text=True, timeout=timeout_s, check=False,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "status": "timeout", "tests": []}
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict):
        return {"passed": False, "status": "harness_error",
                "detail": (completed.stderr or "")[-MAX_DETAIL_CHARS:], "tests": []}
    if "import_error" in payload:
        return {"passed": False, "status": "import_error",
                "detail": str(payload["import_error"])[:MAX_DETAIL_CHARS], "tests": []}
    tests_run = payload.get("tests")
    if not isinstance(tests_run, list) or not tests_run:
        return {"passed": False, "status": "no_tests_executed", "tests": []}
    passed = all(item.get("outcome") == "passed" for item in tests_run)
    failed = sum(1 for item in tests_run if item.get("outcome") != "passed")
    return {
        "passed": passed,
        "status": "completed",
        "summary": f"{len(tests_run) - failed} passed, {failed} failed",
        "tests": tests_run,
    }
