from __future__ import annotations

import hashlib
import shutil
import subprocess

import pytest

from scripts.run_coding_agent_dev_study_v1 import create_docker_shared_workspace
from scripts.run_local_agent import run_public_tests


IMAGE = (
    "vare/transformers-phase-a-arm64:1@sha256:"
    "7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"
)


def _colima_image_available() -> bool:
    if shutil.which("docker") is None:
        return False
    info = subprocess.run(
        ["docker", "--context", "colima", "info"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if info.returncode:
        return False
    image = subprocess.run(
        ["docker", "--context", "colima", "image", "inspect", IMAGE],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return image.returncode == 0


@pytest.mark.skipif(
    not _colima_image_available(),
    reason="requires a running Colima Docker context with the frozen v6 image",
)
def test_colima_public_runner_executes_baseline_failure_and_known_fix(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_CONTEXT", "colima")
    workspace = create_docker_shared_workspace("public-test-colima-regression")
    try:
        workspace.mkdir(mode=0o755)
        target = workspace / "src" / "module.py"
        target.parent.mkdir()
        test_file = workspace / "test_public.py"
        test_file.write_text(
            "from pathlib import Path\n"
            "namespace = {}\n"
            "exec(Path('src/module.py').read_text(), namespace)\n"
            "def test_repaired_behavior():\n"
            "    assert namespace['answer']() == 42\n",
            encoding="utf-8",
        )
        target.write_text("def answer():\n    return 0\n", encoding="utf-8")
        test_hash = hashlib.sha256(test_file.read_bytes()).hexdigest()

        allowlist = {"src/module.py", "test_public.py"}
        baseline = run_public_tests(
            workspace, IMAGE, test_hash, memory_mb=512,
            expected_workspace_files=allowlist,
        )
        assert baseline["classification"] == "assertion_failure"
        assert baseline["returncode"] == 1
        assert baseline["passed"] is False

        target.write_text("def answer():\n    return 42\n", encoding="utf-8")
        fixed = run_public_tests(
            workspace, IMAGE, test_hash, memory_mb=512,
            expected_workspace_files=allowlist,
        )
        assert fixed["classification"] == "passed"
        assert fixed["returncode"] == 0
        assert fixed["passed"] is True
    finally:
        shutil.rmtree(workspace.parent)
