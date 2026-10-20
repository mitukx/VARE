"""CPU-only contracts for repository test execution off the controller host.

Fake Docker verifies command construction and fail-closed output handling; the
small local worker smoke uses only an operator-authored harmless command.
These fixtures do not establish kernel/VM containment.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json

import pytest

from vare.container_isolation import (
    DockerIsolationRunner, IsolationError, IsolationLimits, IsolationResult,
)
from vare.environments.catalog import (
    CatalogEntry, CatalogEnvironmentVerifier, TaskCatalog,
)
from vare.environments.runner import (
    CommandResult, EnvironmentResult, ExecutableEvaluator,
)
from vare.environments.spec import (
    CommandSpec, EngineeringTaskSpec, ProtectedPath, RepositorySource,
)
from vare.evidence import sha256_json
from vare.sandboxed_repository import DockerRepositoryEvaluator, _decode, _worker
from vare.types import Attempt, Task


IMAGE = "local/repository-grader@sha256:" + "b" * 64


def fixture(tmp_path):
    root = tmp_path / "operator" / "task"
    root.mkdir(parents=True)
    candidate_root = tmp_path / "campaign" / "workspaces"
    workspace = candidate_root / "attempt-01"
    workspace.mkdir(parents=True)
    protected = workspace / "important.txt"
    protected.write_bytes(b"unchanged")
    spec = EngineeringTaskSpec(
        id="safe-task",
        title="Trusted CPU smoke",
        prompt="No untrusted code for this fixture",
        family="fixtures",
        source=RepositorySource(local_path="."),
        tests=(CommandSpec(("true",), name="trusted-true"),),
        protected_paths=(ProtectedPath("important.txt", hashlib.sha256(b"unchanged").hexdigest()),),
        metadata={"grader_visibility": "public"},
    )
    spec_path = root / "task.json"
    spec.write(spec_path)
    return root, candidate_root, workspace, spec_path, spec


def test_repository_grader_command_mounts_ro_candidate_and_grader_only(tmp_path):
    root, candidate_root, workspace, spec_path, spec = fixture(tmp_path)
    backend = DockerRepositoryEvaluator(
        runner=DockerIsolationRunner(),
        limits=IsolationLimits(IMAGE),
        allowed_workspace_root=candidate_root,
    )
    argv = backend._argv(
        spec_path=spec_path, workspace=workspace, spec_digest=sha256_json(spec.to_dict()),
        policy_id="incumbent", policy_version=4,
        baseline_metrics=None, name="vare-repo-" + "a" * 24,
    )
    mounts = [argv[i + 1] for i, flag in enumerate(argv[:-1]) if flag == "--mount"]
    assert mounts == [
        f"type=bind,source={workspace},target=/candidate,readonly",
        f"type=bind,source={root},target=/grader,readonly",
    ]
    assert "--network=none" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges:true" in argv
    assert "--read-only" in argv
    assert not any(",target=/tmp" in item for item in mounts)
    assert "--scratch" in argv and "/scratch" in argv
    assert "--expected-spec-sha256" in argv
    assert "--policy-id" in argv and "incumbent" in argv
    assert "--policy-version" in argv and "4" in argv
    assert "--user" in argv and "65534:65534" in argv

    alias = candidate_root / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(IsolationError, match="non-symlink"):
        backend._argv(
            spec_path=spec_path, workspace=alias,
            spec_digest=sha256_json(spec.to_dict()),
            policy_id="incumbent", policy_version=4,
            baseline_metrics={}, name="vare-repo-" + "a" * 24,
        )
    with pytest.raises(IsolationError, match="escapes"):
        backend._argv(
            spec_path=spec_path, workspace=root,
            spec_digest=sha256_json(spec.to_dict()),
            policy_id="incumbent", policy_version=4,
            baseline_metrics={}, name="vare-repo-" + "a" * 24,
        )


def _trusted_worker_result(tmp_path):
    root, candidate_root, workspace, spec_path, spec = fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # Operator-controlled benign test only. Host command exercise does NOT
    # claim hostile-code isolation, which requires an actual Docker daemon.
    args = argparse.Namespace(
        worker=True, spec=str(spec_path), candidate=str(workspace),
        scratch=str(scratch), expected_spec_sha256=sha256_json(spec.to_dict()),
        policy_id="incumbent", policy_version=4, baseline_json="{}",
    )
    return root, candidate_root, workspace, spec_path, spec, args


def test_worker_preserves_scoring_contract_and_rejects_spec_mutation(tmp_path, capsys):
    root, candidate_root, workspace, spec_path, spec, args = _trusted_worker_result(tmp_path)
    assert _worker(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["task_id"] == "safe-task"
    assert result["passed"]
    assert result["correctness"] == 1.0
    assert result["provenance"]["policy_version"] == 4
    assert result["provenance"]["task_spec_sha256"] == sha256_json(spec.to_dict())
    assert result["tests"][0]["returncode"] == 0

    # Tweak the manifest without changing its pinned digest.
    modified = json.loads(spec_path.read_text())
    modified["title"] = "Tampered"
    spec_path.write_text(json.dumps(modified))
    with pytest.raises(RuntimeError, match="modified"):
        _worker(args)


def test_worker_cannot_promote_modified_protected_file(tmp_path, capsys):
    _root, _candidate_root, workspace, _spec_path, _spec, args = _trusted_worker_result(tmp_path)
    (workspace / "important.txt").write_text("tampered")
    assert _worker(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is False
    assert result["integrity_ok"] is False
    assert "protected file modified: important.txt" in result["integrity_failures"]


def test_signed_style_provenance_and_score_validation_fail_closed(tmp_path):
    spec_digest = "a" * 64
    report = EnvironmentResult(
        task_id="task", passed=True, score=0.9, correctness=1.0,
        tests=[CommandResult("test", ("true",), 0, 0.1, "", "")],
        provenance={"task_spec_sha256": spec_digest, "policy_id": "incumbent", "policy_version": 4},
    )
    original = asdict(report)
    assert _decode(
        json.dumps(report.to_dict()), task_id="task", spec_digest=spec_digest,
        policy_id="incumbent", policy_version=4,
    ).passed
    invalid = [
        {**original, "task_id": "other"},
        {**original, "score": 10.0},
        {**original, "score": "NaN"},
        {**original, "score": float("nan")},
        {**original, "passed": True, "correctness": 0.0},
        {**original, "provenance": {**original["provenance"], "task_spec_sha256": "c" * 64}},
        {**original, "provenance": {**original["provenance"], "policy_version": 5}},
        {**original, "tests": []},
        {**original, "integrity_ok": False},
    ]
    for case in invalid:
        with pytest.raises(IsolationError):
            _decode(json.dumps(case), task_id="task", spec_digest=spec_digest,
                    policy_id="incumbent", policy_version=4)


def test_catalog_verifier_never_falls_back_to_host_when_backend_fails(monkeypatch, tmp_path):
    _root, _candidate_root, workspace, spec_path, spec = fixture(tmp_path)
    catalog = TaskCatalog([CatalogEntry(spec_path, spec)])
    task = Task(spec.id, spec.prompt, family=spec.family)
    attempt = Attempt(
        task, "attempt", "incumbent", 4, 1,
        metadata={"workspace_path": str(workspace), "engineering_task_id": task.id},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("host ExecutableEvaluator must NEVER be called")

    monkeypatch.setattr(ExecutableEvaluator, "evaluate", forbidden)

    class RejectBackend:
        async def evaluate(self, **kwargs):
            raise IsolationError("Docker is absent or refused")

    verifier = CatalogEnvironmentVerifier(catalog, isolated_backend=RejectBackend())
    with pytest.raises(IsolationError, match="Docker"):
        asyncio.run(verifier.verify(attempt))


def test_catalog_accepts_only_isolated_backend_result_with_matching_task(monkeypatch, tmp_path):
    _root, _candidate_root, workspace, spec_path, spec = fixture(tmp_path)
    catalog = TaskCatalog([CatalogEntry(spec_path, spec)])
    task = Task(spec.id, spec.prompt, family=spec.family)
    attempt = Attempt(task, "fixture", "p0", 0, 0,
                      metadata={"workspace_path": str(workspace)})

    class WrongBackend:
        async def evaluate(self, **kwargs):
            return EnvironmentResult(
                "forged-other-task", True, 1.0, 1.0,
                tests=[CommandResult("ok", ("true",), 0, 0.1, "", "")],
            )

    with pytest.raises(ValueError, match="identity mismatch"):
        asyncio.run(CatalogEnvironmentVerifier(
            catalog, isolated_backend=WrongBackend(),
        ).verify(attempt))


def test_repository_backend_rejects_nonzero_worker_and_fake_signed_score(tmp_path):
    root, candidate_root, workspace, spec_path, spec = fixture(tmp_path)

    class FakeRunner(DockerIsolationRunner):
        async def execute(self, argv, limits):
            return IsolationResult(8, "", "worker crashed")

    backend = DockerRepositoryEvaluator(
        runner=FakeRunner(), limits=IsolationLimits(IMAGE),
        allowed_workspace_root=candidate_root,
        allow_public_grader_visible_execution=True,
    )
    with pytest.raises(IsolationError, match="failed closed"):
        asyncio.run(backend.evaluate(
            spec_path=spec_path, workspace=workspace,
            spec_digest=sha256_json(spec.to_dict()),
            policy_id="p0", policy_version=0,
        ))


def test_cli_disallows_agent_only_and_evaluator_only_container_modes(tmp_path):
    from vare.cli import _cmd_env_campaign

    root, _, _, _, _ = fixture(tmp_path)
    for agent_image, evaluator_image in ((IMAGE, None), (None, IMAGE)):
        args = argparse.Namespace(
            catalog_root=str(root), agent_argv_json='["true"]',
            isolated_agent_image=agent_image, isolated_evaluator_image=evaluator_image,
        )
        with pytest.raises(ValueError, match="require both"):
            _cmd_env_campaign(args)


def test_repository_evaluator_fails_closed_without_public_task_opt_in(tmp_path):
    _root, candidate_root, workspace, spec_path, spec = fixture(tmp_path)

    class NeverRunRunner:
        called = False

        async def execute(self, *args, **kwargs):
            self.called = True
            raise AssertionError("candidate evaluator must not launch")

    runner = NeverRunRunner()
    backend = DockerRepositoryEvaluator(
        runner=runner, limits=IsolationLimits(IMAGE),
        allowed_workspace_root=candidate_root,
    )
    with pytest.raises(IsolationError, match="explicit public-task opt-in"):
        asyncio.run(backend.evaluate(
            spec_path=spec_path, workspace=workspace,
            spec_digest=sha256_json(spec.to_dict()),
            policy_id="p0", policy_version=0,
        ))
    assert runner.called is False


def test_repository_evaluator_rejects_nonpublic_task_even_after_opt_in(tmp_path):
    _root, candidate_root, workspace, spec_path, spec = fixture(tmp_path)
    spec.metadata["grader_visibility"] = "confidential"
    spec.write(spec_path)

    class NeverRunRunner:
        called = False

        async def execute(self, *args, **kwargs):
            self.called = True
            raise AssertionError("candidate evaluator must not launch")

    runner = NeverRunRunner()
    backend = DockerRepositoryEvaluator(
        runner=runner, limits=IsolationLimits(IMAGE),
        allowed_workspace_root=candidate_root,
        allow_public_grader_visible_execution=True,
    )
    with pytest.raises(IsolationError, match="grader_visibility=public"):
        asyncio.run(backend.evaluate(
            spec_path=spec_path, workspace=workspace,
            spec_digest=sha256_json(spec.to_dict()),
            policy_id="p0", policy_version=0,
        ))
    assert runner.called is False


def test_cli_requires_explicit_public_grader_opt_in(tmp_path):
    from vare.cli import _cmd_env_campaign

    root, _, _, _, _ = fixture(tmp_path)
    args = argparse.Namespace(
        catalog_root=str(root), agent_argv_json='["true"]',
        isolated_agent_image=IMAGE, isolated_evaluator_image=IMAGE,
        allow_public_grader_visible_execution=False,
    )
    with pytest.raises(ValueError, match="disabled by default"):
        _cmd_env_campaign(args)
