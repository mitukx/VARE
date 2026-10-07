from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import shutil

from ..lifecycle import FailureDrivenTaskGenerator
from ..types import Attempt, Experience, FailureCluster, Task, Verification
from .runner import ExecutableEvaluator, Workspace
from .spec import EngineeringTaskSpec


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    spec_path: Path
    spec: EngineeringTaskSpec


class TaskCatalog:
    """Collection of executable engineering task packs.

    Task packs remain data: a JSON task specification plus repository reference.
    The catalog can therefore mix local development fixtures and external Git
    revisions without copying upstream source into this repository.
    """

    def __init__(self, entries: Iterable[CatalogEntry]) -> None:
        entries = tuple(entries)
        ids = [entry.spec.id for entry in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate engineering task IDs")
        self._entries = {entry.spec.id: entry for entry in entries}

    @classmethod
    def discover(cls, root: str | Path) -> "TaskCatalog":
        base = Path(root)
        entries = [CatalogEntry(path, EngineeringTaskSpec.load(path)) for path in sorted(base.rglob("task.json"))]
        if not entries:
            raise ValueError(f"no task.json files found under {base}")
        return cls(entries)

    def get(self, task_id: str) -> CatalogEntry:
        try:
            return self._entries[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown engineering task: {task_id}") from exc

    def specs(self) -> tuple[EngineeringTaskSpec, ...]:
        return tuple(entry.spec for entry in self._entries.values())

    def as_vare_tasks(self) -> list[Task]:
        return [
            Task(
                id=entry.spec.id,
                prompt=entry.spec.prompt,
                family=entry.spec.family,
                metadata={
                    "engineering_task_id": entry.spec.id,
                    "task_spec_path": str(entry.spec_path),
                },
            )
            for entry in self._entries.values()
        ]


class WorkspaceAgent(Protocol):
    async def run(self, *, prompt: str, workspace: Path, step: int) -> str: ...


class WorkspaceAgentRollout:
    """Materialize one isolated repository workspace and hand it to an agent.

    Workspaces are intentionally retained under `run_root` after verification so
    diffs and artifacts can be inspected. A run-level cleanup policy may delete
    them later; the source repository is never modified.
    """

    def __init__(self, catalog: TaskCatalog, agent: WorkspaceAgent, run_root: str | Path) -> None:
        self.catalog = catalog
        self.agent = agent
        self.run_root = Path(run_root).resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)

    async def rollout(self, task: Task, policy_id: str, policy_version: int, step: int) -> Attempt:
        task_id = str(task.metadata.get("engineering_task_id", task.id))
        entry = self.catalog.get(task_id)
        workspace_path = self.run_root / f"{step:08d}-{task_id}"
        if workspace_path.exists():
            shutil.rmtree(workspace_path)
        Workspace(entry.spec, root=workspace_path)  # retained intentionally
        output = await self.agent.run(prompt=entry.spec.prompt, workspace=workspace_path, step=step)
        return Attempt(
            task=task,
            output=output,
            policy_id=policy_id,
            policy_version=policy_version,
            created_step=step,
            metadata={
                "workspace_path": str(workspace_path),
                "engineering_task_id": task_id,
                "task_spec_path": str(entry.spec_path),
            },
        )


class CatalogEnvironmentVerifier:
    name = "executable-environment-catalog"
    trusted = True

    def __init__(self, catalog: TaskCatalog, *, version: int = 0, baseline_metrics: dict[str, dict[str, float]] | None = None) -> None:
        self.catalog = catalog
        self.version = version
        self.baseline_metrics = dict(baseline_metrics or {})

    async def verify(self, attempt: Attempt) -> Verification:
        import asyncio

        task_id = str(attempt.metadata.get("engineering_task_id", attempt.task.id))
        entry = self.catalog.get(task_id)
        workspace = attempt.metadata.get("workspace_path")
        if not workspace:
            raise ValueError("engineering attempt missing workspace_path")
        result = await asyncio.to_thread(
            ExecutableEvaluator(entry.spec, evaluator_root=entry.spec_path.parent).evaluate,
            Path(str(workspace)),
            baseline_metrics=self.baseline_metrics.get(task_id),
            provenance={
                "policy_id": attempt.policy_id,
                "policy_version": attempt.policy_version,
                "task_spec_path": str(entry.spec_path),
            },
        )
        return Verification(
            score=result.score,
            passed=result.passed,
            confidence=1.0,
            verifier_version=self.version,
            verifier_name=self.name,
            trusted=True,
            metadata={"environment_result": result.to_dict()},
        )


class FailureMatchedEnvironmentGenerator(FailureDrivenTaskGenerator):
    """Turn observed failure families into more executable catalog pressure."""

    def __init__(self, catalog: TaskCatalog) -> None:
        self.catalog = catalog

    async def generate(
        self,
        failures: Sequence[FailureCluster],
        experiences: Sequence[Experience],
        limit: int,
    ) -> list[Task]:
        if limit <= 0:
            return []
        ranked_families: list[str] = []
        for cluster in sorted(failures, key=lambda x: (-x.count, x.mean_score, x.label)):
            ranked_families.extend(cluster.task_families)
            ranked_families.append(cluster.label)
        candidates = self.catalog.as_vare_tasks()
        chosen: list[Task] = []
        seen: set[str] = set()
        for family in ranked_families:
            for task in candidates:
                if task.id in seen:
                    continue
                if task.family == family or family in task.family or task.family in family:
                    chosen.append(task)
                    seen.add(task.id)
                    if len(chosen) >= limit:
                        return chosen
        # If exact failure labels do not align with task-family names, use a
        # deterministic catalog fallback so curriculum expansion never depends
        # on a language-model interpretation of labels.
        for task in candidates:
            if task.id not in seen:
                chosen.append(task)
                seen.add(task.id)
                if len(chosen) >= limit:
                    break
        return chosen
