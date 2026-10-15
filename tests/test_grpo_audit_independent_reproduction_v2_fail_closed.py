from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_RELATIVE_PATH = Path(
    "scripts/reproduce_grpo_audit_independent_reproduction_v2.py"
)
LOCK_RELATIVE_PATH = Path("protocols/grpo_audit_independent_reproduction_v2.lock.json")


def make_isolated_repository(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    lock = json.loads((ROOT / LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
    runner_path = tmp_path / RUNNER_RELATIVE_PATH
    runner_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / RUNNER_RELATIVE_PATH, runner_path)

    lock_path = tmp_path / LOCK_RELATIVE_PATH
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    source_artifacts = lock["source_artifacts"]
    assert isinstance(source_artifacts, dict)
    for artifact in source_artifacts.values():
        assert isinstance(artifact, dict)
        relative_path = Path(str(artifact["path"]))
        target_path = tmp_path / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative_path, target_path)
    return runner_path, lock


@pytest.mark.parametrize("optimized", [False, True], ids=["normal", "optimized"])
@pytest.mark.parametrize("mutation", ["input-hash", "exact-check"])
def test_runner_rejects_tampered_inputs_and_failed_checks(
    tmp_path: Path, optimized: bool, mutation: str
) -> None:
    runner_path, lock = make_isolated_repository(tmp_path)
    if mutation == "input-hash":
        source_artifacts = lock["source_artifacts"]
        assert isinstance(source_artifacts, dict)
        report = source_artifacts["report"]
        assert isinstance(report, dict)
        report_path = tmp_path / str(report["path"])
        report_path.write_bytes(report_path.read_bytes() + b"tampered\n")
        expected_error = "source hash mismatch"
    else:
        runner = runner_path.read_text(encoding="utf-8")
        original = "Exact(Fraction(-3, 8), Fraction(1, 4))"
        mutated = "Exact(Fraction(-2, 8), Fraction(1, 4))"
        assert original in runner
        runner_path.write_text(runner.replace(original, mutated, 1), encoding="utf-8")
        expected_error = "exact reproduction checks failed"

    output_path = tmp_path / "result.json"
    command = [sys.executable]
    if optimized:
        command.append("-O")
    command.extend((str(runner_path), "--output", str(output_path)))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode != 0
    assert expected_error in completed.stderr
    assert not output_path.exists()
