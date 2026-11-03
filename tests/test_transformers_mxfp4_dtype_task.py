from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "benchmarks/mlsys_repo_eval_v1/transformers_mxfp4_dtype"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grader = _load_module("transformers_mxfp4_task_grader", TASK_ROOT / "evaluator/grade.py")
calibrator = _load_module(
    "transformers_mxfp4_calibrator",
    REPO_ROOT / "scripts/calibrate_transformers_mxfp4_dtype_grader.py",
)


def test_frozen_protocol_hashes_match_all_task_files() -> None:
    protocol = grader._verify_protocol_lock()
    assert protocol["files"]["grade.py"] == hashlib.sha256(
        (TASK_ROOT / "evaluator/grade.py").read_bytes()
    ).hexdigest()


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 12),
    reason="restricted AST digests are frozen against the task's CPython 3.12 runtime",
)
def test_locked_source_forms_include_baseline_both_fixes_and_wrong_mutants() -> None:
    baseline = (TASK_ROOT / "calibration_inputs/mxfp4.py.baseline").read_bytes()
    known_fix = (TASK_ROOT / "calibration_inputs/mxfp4.py.known_fix").read_bytes()
    variants = calibrator._make_variants(baseline, known_fix)
    for name, source in variants.items():
        accepted = grader.source_contract_bytes(source)["accepted"]
        assert accepted is (name != "import_time_report_forgery"), name


def test_unreviewed_executable_delta_is_rejected_before_import() -> None:
    baseline = (TASK_ROOT / "calibration_inputs/mxfp4.py.baseline").read_bytes()
    edited = baseline + b"\nopen('/tmp/vare-mxfp4-should-not-run', 'w').write('bad')\n"
    result = grader.source_contract_bytes(edited)
    assert result["accepted"] is False
    assert result["reason"] == "unsupported_patch_contract"


def test_candidate_source_symlink_is_not_followed(tmp_path: Path) -> None:
    target = TASK_ROOT / "calibration_inputs/mxfp4.py.known_fix"
    link = tmp_path / "candidate.py"
    link.symlink_to(target)
    with pytest.raises(OSError):
        grader._read_regular_file(link)


def test_fp4_reference_decodes_low_then_high_nibbles() -> None:
    blocks = [[[[0x21]]]]
    scales = [[[127]]]
    output = grader.independent_reference(blocks, scales, "float64")
    assert output == [[[0.5], [1.0]]]


def test_reference_rounds_to_low_precision_destination_formats() -> None:
    assert grader._cast_reference(1.0001, "float16") == 1.0
    assert grader._cast_reference(1.0001, "bfloat16") == 1.0


def test_git_path_inventory_includes_staged_untracked_and_ignored_files(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "config", "user.name", "VARE Test"], check=True)
    subprocess.run(["git", "-C", str(workspace), "config", "user.email", "vare-test@example.invalid"], check=True)
    (workspace / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (workspace / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(workspace), "add", "."], check=True)
    subprocess.run(["git", "-C", str(workspace), "commit", "-qm", "base"], check=True)
    (workspace / "tracked.txt").write_text("staged edit\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(workspace), "add", "tracked.txt"], check=True)
    (workspace / "loose.txt").write_text("untracked\n", encoding="utf-8")
    (workspace / "ignored.txt").write_text("ignored\n", encoding="utf-8")

    assert grader._changed_paths(workspace) == ("ignored.txt", "loose.txt", "tracked.txt")


def test_git_preflight_does_not_run_configured_fsmonitor_hook(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "config", "user.name", "VARE Test"], check=True)
    subprocess.run(["git", "-C", str(workspace), "config", "user.email", "vare-test@example.invalid"], check=True)
    marker = tmp_path / "fsmonitor-ran"
    hook = tmp_path / "fsmonitor"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\nprintf ''\n", encoding="utf-8")
    hook.chmod(0o755)
    (workspace / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(workspace), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(workspace), "commit", "-qm", "base"], check=True)
    subprocess.run(["git", "-C", str(workspace), "config", "core.fsmonitor", str(hook)], check=True)
    (workspace / "tracked.txt").write_text("edited\n", encoding="utf-8")

    assert grader._changed_paths(workspace) == ("tracked.txt",)
    assert not marker.exists()
