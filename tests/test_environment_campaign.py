from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess

from vare.environments import EnvironmentCampaignRunner, TaskCatalog


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/fixtures/oracle/fix_stable_logsumexp.py"


class OracleAgent:
    async def run(self, *, prompt: str, workspace: Path, step: int) -> str:
        subprocess.run(["python", str(ORACLE), str(workspace)], check=True)
        return "ok"


def test_campaign_retains_hash_linked_raw_records(tmp_path):
    catalog = TaskCatalog.discover(ROOT / "benchmarks/smoke")
    runner = EnvironmentCampaignRunner(catalog, OracleAgent(), run_root=tmp_path / "campaign")
    summary = asyncio.run(runner.run(repeats=2))
    assert summary.runs == 2
    assert summary.success_rate == 1.0
    assert len(summary.records_sha256) == 64
    assert (tmp_path / "campaign/records.jsonl").exists()
    assert (tmp_path / "campaign/summary.json").exists()


def test_campaign_normalizes_relative_run_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    catalog = TaskCatalog.discover(ROOT / "benchmarks/smoke")
    runner = EnvironmentCampaignRunner(catalog, OracleAgent(), run_root="relative-campaign")
    assert runner.run_root.is_absolute()
    summary = asyncio.run(runner.run(repeats=1))
    assert summary.success_rate == 1.0


def test_command_agent_resolves_existing_relative_arguments(monkeypatch, tmp_path):
    from vare.environments import CommandWorkspaceAgent
    script = tmp_path / "agent.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(tmp_path)
    agent = CommandWorkspaceAgent(("python", "agent.py"))
    output = asyncio.run(agent.run(prompt="x", workspace=workspace, step=0))
    assert '"returncode": 0' in output
