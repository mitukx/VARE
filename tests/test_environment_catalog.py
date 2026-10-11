from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from vare.environments import (
    CatalogEnvironmentVerifier,
    CommandSpec,
    ExecutableEvaluator,
    TaskCatalog,
    Workspace,
    WorkspaceAgentRollout,
)
from vare.environments.catalog import CatalogEntry
from vare.types import Attempt, Task


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/fixtures/oracle/fix_stable_logsumexp.py"


class OracleAgent:
    async def run(self, *, prompt: str, workspace: Path, step: int) -> str:
        subprocess.run(["python", str(ORACLE), str(workspace)], check=True)
        return "patched"


def test_catalog_rollout_and_dynamic_verifier(tmp_path):
    import asyncio

    catalog = TaskCatalog.discover(ROOT / "benchmarks/smoke")
    task = catalog.as_vare_tasks()[0]
    rollout = WorkspaceAgentRollout(catalog, OracleAgent(), tmp_path / "runs")
    attempt = asyncio.run(rollout.rollout(task, "policy", 3, 9))
    verifier = CatalogEnvironmentVerifier(catalog, version=4)
    verdict = asyncio.run(verifier.verify(attempt))
    assert verdict.passed
    assert verdict.trusted
    assert verdict.verifier_version == 4
    assert Path(attempt.metadata["workspace_path"]).exists()


def test_rollout_rejects_task_metadata_override(tmp_path):
    import asyncio

    catalog = TaskCatalog.discover(ROOT / "benchmarks/smoke")
    source_entry = catalog.get("smoke-stable-logsumexp")
    always_pass = replace(
        source_entry.spec,
        id="always-pass",
        title="Always passing task used only for identity regression",
        prompt="No-op fixture.",
        tests=(CommandSpec(argv=("python", "-c", "pass"), name="always-pass"),),
    )
    catalog = TaskCatalog(
        [source_entry, CatalogEntry(source_entry.spec_path, always_pass)]
    )
    challenge = catalog.as_vare_tasks()[0]
    spoofed = Task(
        id=challenge.id,
        prompt=challenge.prompt,
        family=challenge.family,
        metadata={"engineering_task_id": "always-pass"},
    )
    class NoopAgent:
        called = False

        async def run(self, *, prompt: str, workspace: Path, step: int) -> str:
            self.called = True
            return "no edit"

    agent = NoopAgent()
    run_root = tmp_path / "runs"
    rollout = WorkspaceAgentRollout(catalog, agent, run_root)

    with pytest.raises(ValueError, match="task identity mismatch"):
        asyncio.run(rollout.rollout(spoofed, "policy", 0, 1))

    assert not agent.called
    assert list(run_root.iterdir()) == []


def test_verifier_rejects_attempt_task_id_override(tmp_path):
    import asyncio

    catalog = TaskCatalog.discover(ROOT / "benchmarks/smoke")
    challenge = catalog.get("smoke-stable-logsumexp")
    always_pass = replace(
        challenge.spec,
        id="always-pass",
        title="Always passing task used only for identity regression",
        prompt="No-op fixture.",
        tests=(CommandSpec(argv=("python", "-c", "pass"), name="always-pass"),),
    )
    catalog = TaskCatalog(
        [challenge, CatalogEntry(challenge.spec_path, always_pass)]
    )
    workspace = tmp_path / "workspace"
    Workspace(challenge.spec, root=workspace)
    # The alternate task's own checks accept this workspace, so selecting it
    # through caller-controlled metadata would produce a false task pass.
    assert ExecutableEvaluator(
        always_pass, evaluator_root=challenge.spec_path.parent
    ).evaluate(workspace).passed
    task = Task(
        id=challenge.spec.id,
        prompt=challenge.spec.prompt,
        family=challenge.spec.family,
        metadata={"engineering_task_id": challenge.spec.id},
    )
    attempt = Attempt(
        task=task,
        output="",
        policy_id="policy",
        policy_version=0,
        created_step=0,
        metadata={
            "workspace_path": str(workspace),
            "engineering_task_id": "always-pass",
        },
    )

    with pytest.raises(ValueError, match="task identity mismatch"):
        asyncio.run(CatalogEnvironmentVerifier(catalog).verify(attempt))

    task_claim_spoof = replace(
        attempt,
        task=replace(task, metadata={"engineering_task_id": "always-pass"}),
        metadata={"workspace_path": str(workspace), "engineering_task_id": challenge.spec.id},
    )
    with pytest.raises(ValueError, match="task identity mismatch"):
        asyncio.run(CatalogEnvironmentVerifier(catalog).verify(task_claim_spoof))
