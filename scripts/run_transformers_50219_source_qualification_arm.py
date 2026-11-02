#!/usr/bin/env python3
"""Run the frozen Transformers #50219 source-only regression checks for one arm."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import platform
import resource
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any


TASK_ID = "transformers-50219-tests-fetcher-posix-paths"
SOURCE_RELATIVE_PATH = pathlib.Path("utils/tests_fetcher.py")
UPSTREAM_TEST_NAME = "test_no_windows_path_patterns_in_fetcher"
BEHAVIOR_FUNCTIONS = ("get_repo_utils_tests", "get_conftest_tests")
MAX_WALL_SECONDS = 600
MAX_CPU_SECONDS = 540
MAX_RSS_BYTES = 4 * 1024**3


class WindowsRelativePath:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value

    def as_posix(self) -> str:
        return self.value.replace("\\", "/")


class WindowsPathStub:
    def __init__(self, value: str) -> None:
        self.value = value.rstrip("\\")

    def __truediv__(self, child: str) -> "WindowsPathStub":
        return WindowsPathStub(self.value + "\\" + child)

    def is_dir(self) -> bool:
        return True

    def glob(self, pattern: str) -> list["WindowsPathStub"]:
        if pattern != "test_*.py":
            raise AssertionError(f"unexpected test glob: {pattern}")
        return [
            WindowsPathStub(self.value + "\\test_alpha.py"),
            WindowsPathStub(self.value + "\\test_beta.py"),
        ]

    def relative_to(self, root: "WindowsPathStub") -> WindowsRelativePath:
        prefix = root.value + "\\"
        if not self.value.startswith(prefix):
            raise ValueError("path is not relative to the fake Windows repository root")
        return WindowsRelativePath(self.value[len(prefix) :])


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_function(source: str, name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(source)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        ),
        None,
    )
    if function is None:
        raise ValueError(f"required source function is absent: {name}")
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), f"<source:{name}>", "exec"), namespace)
    return namespace[name]


def run_upstream_regression(test_path: pathlib.Path, workspace: pathlib.Path) -> dict[str, Any]:
    tree = ast.parse(test_path.read_text(encoding="utf-8"))
    test_class = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TestFetcherTester"),
        None,
    )
    if test_class is None:
        raise ValueError("frozen upstream test class is missing")
    method = next(
        (
            node
            for node in test_class.body
            if isinstance(node, ast.FunctionDef) and node.name == UPSTREAM_TEST_NAME
        ),
        None,
    )
    if method is None:
        raise ValueError("frozen upstream regression test method is missing")
    namespace: dict[str, Any] = {
        "Path": pathlib.Path,
        "REPO_PATH": str(workspace),
    }
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<upstream-regression>", "exec"), namespace)
    try:
        namespace[UPSTREAM_TEST_NAME](unittest_case())
    except AssertionError as exc:
        return {"test_id": UPSTREAM_TEST_NAME, "status": "FAIL", "error": str(exc)}
    return {"test_id": UPSTREAM_TEST_NAME, "status": "PASS", "error": None}


def unittest_case() -> Any:
    import unittest

    return unittest.TestCase()


def run_windows_path_behavior(source: str) -> list[dict[str, Any]]:
    results = []
    for function_name in BEHAVIOR_FUNCTIONS:
        namespace: dict[str, Any] = {
            "PATH_TO_REPO": WindowsPathStub(r"C:\repo"),
            "PATH_TO_TESTS": WindowsPathStub(r"C:\repo\tests"),
        }
        function = extract_function(source, function_name, namespace)
        actual = function()
        expected = [
            "tests/" + ("repo_utils" if function_name == "get_repo_utils_tests" else "conftest_tests") + "/test_alpha.py",
            "tests/" + ("repo_utils" if function_name == "get_repo_utils_tests" else "conftest_tests") + "/test_beta.py",
        ]
        results.append(
            {
                "test_id": f"windows_posix_outputs.{function_name}",
                "status": "PASS" if actual == expected else "FAIL",
                "expected": expected,
                "actual": actual,
                "error": None,
            }
        )
    return results


def set_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("baseline", "wrong-mutant", "upstream-fixed"), required=True)
    parser.add_argument("--workspace", type=pathlib.Path, required=True)
    parser.add_argument("--trusted-test", type=pathlib.Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-test-sha256", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve(strict=True)
    source_path = workspace / SOURCE_RELATIVE_PATH
    test_path = args.trusted_test.resolve(strict=True)
    output_path = args.output.resolve()
    if not source_path.is_file() or source_path.is_symlink():
        raise SystemExit("candidate workspace must contain only a regular allowlisted source file")
    if test_path.is_relative_to(workspace):
        raise SystemExit("grader test must remain outside the candidate workspace")
    if not output_path.parent.is_dir():
        raise SystemExit("precreated arm output directory is missing")
    source_text = source_path.read_text(encoding="utf-8")
    actual_source_hash = sha256(source_path)
    actual_test_hash = sha256(test_path)
    if actual_source_hash != args.expected_source_sha256:
        raise SystemExit("candidate source hash differs from the frozen execution addendum")
    if actual_test_hash != args.expected_test_sha256:
        raise SystemExit("trusted test hash differs from the frozen execution addendum")

    set_limits()
    signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(TimeoutError("wall limit exceeded")))
    signal.alarm(MAX_WALL_SECONDS)
    started = time.monotonic()
    execution_error = None
    upstream = None
    behavioral = []
    try:
        upstream = run_upstream_regression(test_path, workspace)
        behavioral = run_windows_path_behavior(source_text)
    except BaseException as exc:
        execution_error = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
    elapsed = time.monotonic() - started
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(usage if sys.platform == "darwin" else usage * 1024)
    resource_breach = peak_rss_bytes > MAX_RSS_BYTES
    result = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "arm": args.arm,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_source_path": SOURCE_RELATIVE_PATH.as_posix(),
        "source_sha256": actual_source_hash,
        "trusted_test_sha256": actual_test_hash,
        "trusted_test_selector": f"TestFetcherTester.{UPSTREAM_TEST_NAME}",
        "upstream_regression": upstream,
        "independent_windows_path_behavior": behavioral,
        "execution_error": execution_error,
        "resource_breach": resource_breach,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss_bytes,
        "limits": {
            "cpu_seconds": MAX_CPU_SECONDS,
            "wall_seconds": MAX_WALL_SECONDS,
            "observed_peak_rss_bytes_max": MAX_RSS_BYTES,
            "rss_enforcement": "post-run measurement; no RLIMIT_AS, RLIMIT_RSS, or RLIMIT_DATA limit is claimed",
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"arm": args.arm, "upstream_regression": upstream, "behavior": behavioral, "execution_error": execution_error}))
    return 0 if execution_error is None and not resource_breach else 2


if __name__ == "__main__":
    raise SystemExit(main())
