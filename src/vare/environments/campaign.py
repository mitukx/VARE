from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Sequence

from ..evidence import sha256_json
from .catalog import CatalogEnvironmentVerifier, TaskCatalog, WorkspaceAgent, WorkspaceAgentRollout


@dataclass(frozen=True, slots=True)
class CommandWorkspaceAgent:
    """Adapter for an external repository-editing agent command.

    The command vector is configured by the experimenter. Placeholders
    `{workspace}`, `{prompt}`, and `{step}` are substituted without invoking a
    shell. The external program is responsible for editing the workspace.
    """

    argv: tuple[str, ...]
    timeout_s: float = 1800.0
    launch_root: Path = field(default_factory=lambda: Path.cwd().resolve())

    async def run(self, *, prompt: str, workspace: Path, step: int) -> str:
        if not self.argv:
            raise ValueError("agent command must not be empty")
        values = {
            "{workspace}": str(workspace),
            "{prompt}": prompt,
            "{step}": str(step),
        }
        argv: list[str] = []
        for item in self.argv:
            value = item
            had_placeholder = any(token in value for token in values)
            for token, replacement in values.items():
                value = value.replace(token, replacement)
            if not had_placeholder and not Path(value).is_absolute():
                candidate = (self.launch_root / value).resolve()
                if candidate.exists():
                    value = str(candidate)
            argv.append(value)

        def execute() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                argv,
                cwd=workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )

        proc = await asyncio.to_thread(execute)
        return json.dumps(
            {
                "argv": argv,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-100_000:],
                "stderr": proc.stderr[-100_000:],
            },
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class CampaignRecord:
    task_id: str
    family: str
    repeat: int
    step: int
    passed: bool
    score: float
    correctness: float
    verifier_version: int
    wall_s: float
    workspace_path: str
    diff_sha256: str
    agent_output_sha256: str
    result_sha256: str


@dataclass(frozen=True, slots=True)
class CampaignSummary:
    runs: int
    tasks: int
    successes: int
    success_rate: float
    mean_score: float
    mean_wall_s: float
    by_family: dict[str, dict[str, float | int]]
    records_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


class EnvironmentCampaignRunner:
    """Fixed-budget repository-agent campaign with raw per-run evidence."""

    def __init__(
        self,
        catalog: TaskCatalog,
        agent: WorkspaceAgent,
        *,
        run_root: str | Path,
        verifier_version: int = 0,
    ) -> None:
        self.catalog = catalog
        self.rollout = WorkspaceAgentRollout(catalog, agent, Path(run_root) / "workspaces")
        self.verifier = CatalogEnvironmentVerifier(catalog, version=verifier_version)
        self.run_root = Path(run_root).resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)

    async def run(self, *, repeats: int = 1, task_ids: Sequence[str] | None = None) -> CampaignSummary:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        tasks = self.catalog.as_vare_tasks()
        if task_ids is not None:
            selected = set(task_ids)
            tasks = [task for task in tasks if task.id in selected]
            missing = selected - {task.id for task in tasks}
            if missing:
                raise KeyError(f"unknown task IDs: {sorted(missing)}")
        if not tasks:
            raise ValueError("campaign requires at least one task")

        records: list[CampaignRecord] = []
        raw_path = self.run_root / "records.jsonl"
        if raw_path.exists():
            raw_path.unlink()
        step = 0
        for repeat in range(repeats):
            for task in tasks:
                started = time.perf_counter()
                attempt = await self.rollout.rollout(task, "campaign-agent", 0, step)
                verdict = await self.verifier.verify(attempt)
                wall_s = time.perf_counter() - started
                result = dict(verdict.metadata["environment_result"])
                diff = str(result.get("workspace_diff", ""))
                result_sha = sha256_json(result)
                record = CampaignRecord(
                    task_id=task.id,
                    family=task.family,
                    repeat=repeat,
                    step=step,
                    passed=verdict.passed,
                    score=verdict.score,
                    correctness=float(result.get("correctness", 0.0)),
                    verifier_version=verdict.verifier_version,
                    wall_s=wall_s,
                    workspace_path=str(attempt.metadata["workspace_path"]),
                    diff_sha256=hashlib.sha256(diff.encode("utf-8")).hexdigest(),
                    agent_output_sha256=hashlib.sha256(attempt.output.encode("utf-8")).hexdigest(),
                    result_sha256=result_sha,
                )
                with raw_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(record), sort_keys=True) + "\n")
                records.append(record)
                step += 1

        families: dict[str, list[CampaignRecord]] = {}
        for record in records:
            families.setdefault(record.family, []).append(record)
        by_family = {
            family: {
                "runs": len(rows),
                "successes": sum(row.passed for row in rows),
                "success_rate": sum(row.passed for row in rows) / len(rows),
                "mean_score": sum(row.score for row in rows) / len(rows),
            }
            for family, rows in sorted(families.items())
        }
        raw_bytes = raw_path.read_bytes()
        summary = CampaignSummary(
            runs=len(records),
            tasks=len(tasks),
            successes=sum(row.passed for row in records),
            success_rate=sum(row.passed for row in records) / len(records),
            mean_score=sum(row.score for row in records) / len(records),
            mean_wall_s=sum(row.wall_s for row in records) / len(records),
            by_family=by_family,
            records_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )
        (self.run_root / "summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
