from __future__ import annotations

from pathlib import Path
import subprocess

from vare.environments import CatalogEnvironmentVerifier, TaskCatalog, WorkspaceAgentRollout


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
