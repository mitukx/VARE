"""No-daemon regression for readonly public tests on the frozen Docker runner."""

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts import run_local_agent as agent
from vare import container_isolation
from vare.container_isolation import IsolationResult

IMAGE = "local/vare-test@sha256:" + "a" * 64


def test_public_test_uses_existing_readonly_candidate_argv(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "task"
    workspace.mkdir(parents=True)
    source = workspace / "test_public.py"
    source.write_text("assert 1 + 1 == 2\\n", encoding="utf-8")
    identity = hashlib.sha256(source.read_bytes()).hexdigest()
    calls: list[tuple[str, object]] = []

    class Runner:
        def candidate_argv(self, limits, *, name, allowed_workspace_root, workspace, command,
                           readonly_workspace=False):
            assert readonly_workspace is True
            assert allowed_workspace_root == workspace.parent
            assert name.startswith("vare-agent-")
            assert list(command) == ["python", "/workspace/test_public.py"]
            calls.append(("candidate_argv", name))
            return ["docker", "run", "--mount",
                    f"type=bind,source={workspace},target=/workspace,readonly"]

        async def execute(self, argv, limits):
            assert "--mount" in argv
            calls.append(("execute", argv))
            return IsolationResult(0, "ok", "")

    monkeypatch.setattr(container_isolation, "DockerIsolationRunner", Runner)
    report = agent.run_public_tests(workspace, IMAGE, identity)
    assert report["passed"] is True
    assert report["stdout"] == "ok"
    assert [x[0] for x in calls] == ["candidate_argv", "execute"]


def test_public_test_refuses_changed_source_before_isolation(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "test_public.py").write_text("assert False\\n", encoding="utf-8")
    monkeypatch.setattr(
        container_isolation, "DockerIsolationRunner",
        lambda: (_ for _ in ()).throw(AssertionError("never start Docker")),
    )
    result = agent.run_public_tests(workspace, IMAGE, "0" * 64)
    assert result["passed"] is False
    assert "missing or changed" in result["error"]
