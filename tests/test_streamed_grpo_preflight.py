import json
import subprocess
import sys


def command(output):
    return [sys.executable, "scripts/validate_streamed_grpo_cpu.py", "--output", str(output),
            "--seed", "999", "--model-dir", "/nonexistent-model", "--rvl-source", "/nonexistent-rvl"]


def test_existing_run_directory_is_never_modified(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    marker = root / "failure.json"
    marker.write_text("original evidence")
    run = subprocess.run(command(root), capture_output=True, text=True, timeout=10)
    assert run.returncode != 0 and marker.read_text() == "original evidence"
    assert list(root.iterdir()) == [marker]


def test_failed_preflight_retains_owned_failure_before_model_import(tmp_path):
    root = tmp_path / "new"
    run = subprocess.run(command(root), capture_output=True, text=True, timeout=10)
    assert run.returncode != 0
    failure = json.loads((root / "failure.json").read_text())
    assert failure["type"] == "RuntimeError" and "undeclared seed" in failure["message"]
    assert not (root / "trajectories.db").exists()
