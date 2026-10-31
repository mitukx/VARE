"""Run-002 R7 build check: report hygiene (no network, no model)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("run002_build_check", ROOT / "scripts" / "run002_build_check.py")
assert _spec and _spec.loader
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)


def test_summarize_drops_test_names_and_output() -> None:
    grade = {"passed": False, "reason": "tests_failed", "f2p_passed": 0, "f2p_total": 2,
             "p2p_passed": 3, "p2p_total": 3, "pytest_exit": 1, "stdout_sha256": "0" * 64,
             "seconds": 1.0, "pytest_seconds": 0.9, "missing": ["tests/x.py::secret_name"],
             "tests": {"tests/x.py::other_name": "FAILED"}, "stdout": "secret output",
             "apply_stderr": "secret"}
    out = bc.summarize(grade)
    text = json.dumps(out)
    assert "secret" not in text and "::" not in text
    assert out["missing_count"] == 1 and out["f2p_total"] == 2


def test_sanitize_removes_host_paths(tmp_path: Path) -> None:
    cache, data = tmp_path / "cache", tmp_path / "data"
    text = f"error in {cache}/repos/x and {data}/shard and {Path.home()}/y"
    out = bc.sanitize(text, cache, data)
    assert str(tmp_path) not in out and str(Path.home()) not in out


def test_report_settings_mismatch_refuses_with_exit_2(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    argv = ["--report", str(report), "--cache", str(tmp_path / "c"), "--dataset-cache", str(tmp_path / "d")]
    args = bc.build_parser().parse_args(argv)
    checker = bc.Checker(args)
    rep = checker.load_report({"a": "1"}, {"tasks_per_repo": 8})
    checker.save(rep)
    with pytest.raises(SystemExit) as exc:
        checker.load_report({"a": "1"}, {"tasks_per_repo": 7})
    assert exc.value.code == 2
