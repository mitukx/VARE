#!/usr/bin/env python3
"""Run one pinned unittest method and emit each unittest subTest outcome as JSON."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path


class CaseAwareResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subtest_records = []

    def addSubTest(self, test, subtest, err):
        params = getattr(subtest, "params", {})
        parameters = {str(key): str(value) for key, value in params.items()}
        if err is None:
            outcome = "passed"
            exception_type = None
            message = None
        else:
            exception_type = err[0].__name__
            message = str(err[1])
            outcome = "error" if issubclass(err[0], test.failureException) is False else "failed"
        self.subtest_records.append({
            "id": subtest.id(),
            "parameters": parameters,
            "outcome": outcome,
            "exception_type": exception_type,
            "message": message,
        })
        super().addSubTest(test, subtest, err)


def main(argv):
    if len(argv) != 4:
        raise SystemExit("usage: case_runner.py TEST_FILE TEST_CLASS TEST_METHOD")
    test_path = Path(argv[1]).resolve()
    class_name, method_name = argv[2], argv[3]
    spec = importlib.util.spec_from_file_location("vare_pinned_test_module", test_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import test module from {test_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    test_class = getattr(module, class_name)
    test_case = test_class(method_name)
    suite = unittest.TestSuite([test_case])
    result = unittest.TextTestRunner(verbosity=2, resultclass=CaseAwareResult).run(suite)
    summary = {
        "test_file": str(test_path),
        "test_id": f"{class_name}.{method_name}",
        "tests_run": result.testsRun,
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "skip_count": len(result.skipped),
        "subtests": result.subtest_records,
        "was_successful": result.wasSuccessful(),
    }
    print("VARE_CASE_RUNNER_RESULT=" + json.dumps(summary, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
