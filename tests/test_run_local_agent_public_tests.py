from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from typing import cast

import pytest

from vare.container_isolation import IsolationError, IsolationResult
from vare.agent_structured_edit import StructuredEditController, StructuredEditError


agent = importlib.import_module("scripts.run_local_agent")
IMAGE = (
    "vare/transformers-phase-a-arm64:1@sha256:"
    "7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"
)


@pytest.mark.parametrize(
    ("returncode", "stderr", "expected"),
    [
        (0, "", "passed"),
        (1, "AssertionError: expected value", "assertion_failure"),
        (1, "ModuleNotFoundError: missing dependency", "evaluator_unavailable"),
        (125, "docker: Error response from daemon: bind source path does not exist", "container_launch_failure"),
        (125, "candidate test exited with code 125", "container_launch_failure"),
        (126, "exec: /usr/local/bin/python: permission denied", "container_launch_failure"),
        (127, "exec: \"python\": executable file not found in $PATH", "container_launch_failure"),
        (137, "", "container_resource_failure"),
        (1, "Traceback (most recent call last):\n  File \"/workspace/src/value.py\", line 1\nModuleNotFoundError: No module named 'missing_dep'", "test_execution_error"),
        (1, "Traceback (most recent call last):\n  File \"/workspace/test_public.py\", line 1\nModuleNotFoundError: No module named 'grader_dep'", "evaluator_unavailable"),
    ],
)
def test_public_test_exit_classification(
    returncode: int, stderr: str, expected: str,
) -> None:
    assert agent.classify_public_test_returncode(returncode, stderr) == expected


def _workspace(root: Path, return_value: int) -> Path:
    workspace = root / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "value.py").write_text(
        f"def value():\n    return {return_value}\n", encoding="utf-8",
    )
    (workspace / "test_public.py").write_text(
        "from value import value\n\ndef test_value():\n    assert value() == 42\n",
        encoding="utf-8",
    )
    return workspace


def test_docker_125_is_unavailable_not_a_candidate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_root = tmp_path / "host-root"
    host_root.mkdir()
    staging_root = tmp_path / "owned-cache"
    staging_root.mkdir(mode=0o700)
    staging_root.chmod(0o700)
    workspace = _workspace(host_root, 41)
    expected = hashlib.sha256((workspace / "test_public.py").read_bytes()).hexdigest()
    seen: dict[str, object] = {}

    monkeypatch.setattr(agent, "_public_test_staging_root", lambda: staging_root)

    async def docker_125(self, limits, *, allowed_workspace_root, workspace, command,
                         readonly_workspace=False, stdin_data=None):
        seen.update(
            workspace=Path(workspace), root=Path(allowed_workspace_root),
            readonly=readonly_workspace, command=tuple(command),
        )
        return IsolationResult(
            125, "", "docker: Error response from daemon: invalid mount config: bind source path does not exist",
        )

    monkeypatch.setattr(
        "vare.container_isolation.DockerIsolationRunner.run_candidate", docker_125,
    )
    result = agent.run_public_tests(
        workspace, IMAGE, expected,
        expected_workspace_files={"src/value.py", "test_public.py"},
    )

    assert result["passed"] is None
    assert result["gradeable"] is False
    assert result["classification"] == "container_launch_failure"
    visible = agent.candidate_visible_tool_result("run_public_tests", result, opaque_handles=True)
    assert "stderr" not in visible
    assert cast(Path, seen["workspace"]).parent == cast(Path, seen["root"])
    assert seen["readonly"] is True
    assert not list(staging_root.iterdir())

    controller = StructuredEditController(workspace, {"src/value.py"})
    feedback = controller.record_public_test_result(result)
    assert feedback["classification"] == "container_launch_failure"
    assert feedback["passed"] is None
    assert "stop this rollout" in feedback["next_action"]
    with pytest.raises(StructuredEditError, match="until public tests pass"):
        controller.finalize_for_independent_grading()


def test_public_test_timeout_is_not_a_candidate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_root = tmp_path / "host-root"
    host_root.mkdir()
    staging_root = tmp_path / "owned-cache"
    staging_root.mkdir(mode=0o700)
    staging_root.chmod(0o700)
    workspace = _workspace(host_root, 41)
    expected = hashlib.sha256((workspace / "test_public.py").read_bytes()).hexdigest()
    monkeypatch.setattr(agent, "_public_test_staging_root", lambda: staging_root)

    async def timeout(self, *_args, **_kwargs):
        raise IsolationError("container exceeded wall-time limit")

    monkeypatch.setattr(
        "vare.container_isolation.DockerIsolationRunner.run_candidate", timeout,
    )
    result = agent.run_public_tests(
        workspace, IMAGE, expected,
        expected_workspace_files={"src/value.py", "test_public.py"},
    )
    assert result["classification"] == "candidate_timeout"
    assert result["passed"] is False
    assert result["gradeable"] is True
    assert not list(staging_root.iterdir())


def test_raw_staging_exception_is_retained_but_redacted_from_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path, 42)
    expected = hashlib.sha256((workspace / "test_public.py").read_bytes()).hexdigest()

    class BrokenStaging:
        def __init__(self, _workspace):
            pass

        def __enter__(self):
            raise PermissionError("private staging path denied")

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(agent, "PublicTestWorkspaceStaging", BrokenStaging)
    result = agent.run_public_tests(
        workspace, IMAGE, expected,
        expected_workspace_files={"src/value.py", "test_public.py"},
    )

    assert result["classification"] == "evaluator_unavailable"
    assert result["passed"] is None
    assert result["error"] == "PermissionError: private staging path denied"
    visible = agent.candidate_visible_tool_result("run_public_tests", result, opaque_handles=True)
    assert "private staging path denied" not in json.dumps(visible)
    assert "/" not in visible.get("feedback", "")


@pytest.mark.parametrize(
    ("stderr", "classification", "passed", "gradeable"),
    [
        (
            'Traceback (most recent call last):\n  File "/workspace/src/value.py", line 1\n'
            "ModuleNotFoundError: No module named 'candidate_dep'",
            "test_execution_error", False, True,
        ),
        (
            'Traceback (most recent call last):\n  File "/workspace/test_public.py", line 1\n'
            "ModuleNotFoundError: No module named 'grader_dep'",
            "evaluator_unavailable", None, False,
        ),
    ],
)
def test_missing_import_is_attributed_by_traceback_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stderr: str,
    classification: str, passed: bool | None, gradeable: bool,
) -> None:
    workspace = _workspace(tmp_path, 41)
    expected = hashlib.sha256((workspace / "test_public.py").read_bytes()).hexdigest()
    staging_root = tmp_path / "owned-cache"
    staging_root.mkdir(mode=0o700)
    staging_root.chmod(0o700)
    monkeypatch.setattr(agent, "_public_test_staging_root", lambda: staging_root)

    async def missing_import(self, *_args, **_kwargs):
        return IsolationResult(1, "", stderr)

    monkeypatch.setattr(
        "vare.container_isolation.DockerIsolationRunner.run_candidate", missing_import,
    )
    result = agent.run_public_tests(
        workspace, IMAGE, expected,
        expected_workspace_files={"src/value.py", "test_public.py"},
    )
    assert result["classification"] == classification
    assert result["passed"] is passed
    assert result["gradeable"] is gradeable

    if classification == "evaluator_unavailable":
        visible = agent.candidate_visible_tool_result(
            "run_public_tests", result, opaque_handles=True,
        )
        assert "grader_dep" not in json.dumps(visible)
        assert "/workspace" not in json.dumps(visible)


def _colima_image_ready() -> bool:
    if not shutil.which("docker"):
        return False
    context = subprocess.run(
        ["docker", "context", "show"], capture_output=True, text=True, check=False,
        env={**os.environ, "DOCKER_CONTEXT": "colima"}, timeout=10,
    )
    image = subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True, text=True,
        check=False, env={**os.environ, "DOCKER_CONTEXT": "colima"}, timeout=10,
    )
    return context.returncode == 0 and context.stdout.strip() == "colima" and image.returncode == 0


@pytest.mark.skipif(not _colima_image_ready(), reason="Colima and pinned image unavailable")
def test_colima_public_tests_stage_unshared_host_workspace_and_run_baseline_and_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCKER_CONTEXT", "colima")
    staging_root = agent._public_test_staging_root()
    before = set(staging_root.iterdir())
    baseline = _workspace(tmp_path / "baseline", 41)
    fixed = _workspace(tmp_path / "fixed", 42)
    for workspace in (baseline, fixed):
        workspace.resolve().relative_to(tmp_path.resolve())
        assert not str(workspace).startswith(str(Path.home() / ".cache"))

    baseline_hash = hashlib.sha256((baseline / "test_public.py").read_bytes()).hexdigest()
    fixed_hash = hashlib.sha256((fixed / "test_public.py").read_bytes()).hexdigest()
    allowlist = {"src/value.py", "test_public.py"}
    baseline_result = agent.run_public_tests(
        baseline, IMAGE, baseline_hash, expected_workspace_files=allowlist,
    )
    fixed_result = agent.run_public_tests(
        fixed, IMAGE, fixed_hash, expected_workspace_files=allowlist,
    )

    assert baseline_result["classification"] == "assertion_failure"
    assert baseline_result["passed"] is False
    assert baseline_result["gradeable"] is True
    assert fixed_result["classification"] == "passed"
    assert fixed_result["passed"] is True
    assert fixed_result["gradeable"] is True
    assert baseline_result["workspace_staging"] == "owned-cache-bind-mount"
    assert fixed_result["workspace_staging"] == "owned-cache-bind-mount"
    assert set(staging_root.iterdir()) == before
    assert stat.S_IMODE(staging_root.stat().st_mode) == 0o700


def test_public_test_runner_rejects_extra_candidate_files_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path, 42)
    (workspace / "sitecustomize.py").write_text("raise RuntimeError('injected')\n")
    expected = hashlib.sha256((workspace / "test_public.py").read_bytes()).hexdigest()

    async def should_not_run(self, *_args, **_kwargs):
        raise AssertionError("Docker must not start for an out-of-allowlist file")

    monkeypatch.setattr(
        "vare.container_isolation.DockerIsolationRunner.run_candidate", should_not_run,
    )
    result = agent.run_public_tests(
        workspace, IMAGE, expected,
        expected_workspace_files={"src/value.py", "test_public.py"},
    )
    assert result["classification"] == "runner_integrity_error"
    assert result["gradeable"] is False
    assert "stderr" not in result
