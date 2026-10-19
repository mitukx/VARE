from __future__ import annotations

import asyncio
from collections.abc import Awaitable
import signal
import struct
import subprocess
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Sequence

from ..evidence import sha256_json
from .catalog import CatalogEnvironmentVerifier, RepositoryEvaluatorBackend, TaskCatalog, WorkspaceAgent, WorkspaceAgentRollout
from .runner import resolve_launch_argv


async def _run_owned_command(
    argv: list[str], *, cwd: Path, env: dict[str, str], timeout_s: float,
    output_limit_bytes: int = 262_144,
) -> tuple[int, str, str]:
    """Run one externally configured tool agent; never abandon a live child.

    Retain at most 100 KB per stream and reject excessive *total* output.
    The guard's separate four-byte status pipe avoids conflating its exit with
    the agent's return code. A closed owner pipe terminates the entire ordinary
    process group, including on uncatchable controller death.
    """
    if os.name != "posix":
        raise RuntimeError("owned command agents require POSIX process groups")
    if not argv or timeout_s <= 0 or output_limit_bytes <= 0:
        raise ValueError("invalid bounded agent execution arguments")

    status_read, status_write = os.pipe()
    guard = Path(__file__).resolve().parents[1] / "_grader_guard.py"
    proc: asyncio.subprocess.Process | None = None
    pumps: list[asyncio.Task[None]] = []
    status_task: asyncio.Task[bytes] | None = None
    excess_task: asyncio.Task[bool] | None = None
    outputs = [bytearray(), bytearray()]
    total = 0
    overflow = asyncio.Event()

    async def pump(stream: asyncio.StreamReader, index: int) -> None:
        nonlocal total
        while block := await stream.read(16_384):
            total += len(block)
            outputs[index].extend(block)
            if len(outputs[index]) > 100_000:
                del outputs[index][:-100_000]
            if total > output_limit_bytes:
                overflow.set()

    try:
        # A cancel during process creation must still acquire its handle, so
        # it can close the guardian's owner pipe rather than orphaning work.
        launch = asyncio.create_task(asyncio.create_subprocess_exec(
            sys.executable, "-B", str(guard), str(status_write), *argv,
            cwd=cwd, env=env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            pass_fds=(status_write,), start_new_session=True,
        ))
        cancelled = False
        while True:
            try:
                proc = await asyncio.shield(launch)
                break
            except asyncio.CancelledError:
                cancelled = True
                if launch.cancelled():
                    raise
        os.close(status_write)
        status_write = -1
        if cancelled:
            raise asyncio.CancelledError()
        assert proc.stdout is not None and proc.stderr is not None
        pumps = [asyncio.create_task(pump(proc.stdout, 0)),
                 asyncio.create_task(pump(proc.stderr, 1))]
        status_task = asyncio.create_task(asyncio.to_thread(os.read, status_read, 4))
        excess_task = asyncio.create_task(overflow.wait())
        done, _ = await asyncio.wait(
            (status_task, excess_task), timeout=timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if excess_task in done:
            raise RuntimeError("agent exceeded bounded output budget")
        if status_task not in done:
            raise subprocess.TimeoutExpired(argv, timeout_s)
        raw = status_task.result()
        if len(raw) != 4:
            raise RuntimeError("agent ownership guard exited without status")
        returncode = struct.unpack("!i", raw)[0]
    finally:
        if status_write >= 0:
            os.close(status_write)
        if proc is not None:
            assert proc.stdin is not None
            proc.stdin.close()  # EOF kills ordinary descendants, including after cancellation.

            async def finish() -> None:
                pending: list[Awaitable[object]] = [proc.wait(), *pumps]
                if status_task is not None:
                    pending.append(status_task)
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True), timeout=3,
                    )
                except asyncio.TimeoutError:
                    # Guard failure fallback; do not abandon a running session.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    if proc.returncode is None:
                        proc.kill()
                    for task in pumps:
                        task.cancel()
                    if status_task is not None:
                        status_task.cancel()
                    await asyncio.gather(proc.wait(), *pumps, return_exceptions=True)

            closing = asyncio.create_task(finish())
            interrupted = False
            while True:
                try:
                    await asyncio.shield(closing)
                    break
                except asyncio.CancelledError:
                    interrupted = True
            if excess_task is not None:
                excess_task.cancel()
                await asyncio.gather(excess_task, return_exceptions=True)
            os.close(status_read)
            if interrupted:
                raise asyncio.CancelledError()
        else:
            os.close(status_read)

    if overflow.is_set():
        raise RuntimeError("agent exceeded bounded output budget")
    return (
        returncode,
        outputs[0].decode("utf-8", errors="replace"),
        outputs[1].decode("utf-8", errors="replace"),
    )


@dataclass(frozen=True, slots=True)
class CommandWorkspaceAgent:
    """Adapter for an external repository-editing agent command.

    The command vector is configured by the experimenter. Placeholders
    `{workspace}`, `{prompt}`, and `{step}` in that vector are substituted
    without invoking a shell. Text inserted for a placeholder is not scanned
    again, so a prompt that contains `{step}` stays literal. The external
    program is responsible for editing the workspace.
    """

    argv: tuple[str, ...]
    timeout_s: float = 1800.0
    launch_root: Path = field(default_factory=lambda: Path.cwd().resolve())
    # Prevent accidental inheritance of controller API keys/tokens. This is
    # hygiene only: a same-UID subprocess is NOT a security boundary.
    environment_allowlist: tuple[str, ...] = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR")

    async def run(self, *, prompt: str, workspace: Path, step: int) -> str:
        if not self.argv:
            raise ValueError("agent command must not be empty")
        values = {
            "{workspace}": str(workspace),
            "{prompt}": prompt,
            "{step}": str(step),
        }
        expanded = resolve_launch_argv(self.argv, values)
        argv: list[str] = []
        for original, value in zip(self.argv, expanded, strict=True):
            had_placeholder = any(
                token in original for token in (*values, "{python}")
            )
            if (
                not had_placeholder
                and value != sys.executable
                and not Path(value).is_absolute()
            ):
                candidate = (self.launch_root / value).resolve()
                if candidate.exists():
                    value = str(candidate)
            argv.append(value)

        if (
            type(self.environment_allowlist) is not tuple
            or any(not isinstance(key, str) or not key.isidentifier() for key in self.environment_allowlist)
        ):
            raise ValueError("environment_allowlist must contain valid variable names")
        child_env = {key: os.environ[key] for key in self.environment_allowlist if key in os.environ}

        code, stdout, stderr = await _run_owned_command(
            argv, cwd=workspace, env=child_env, timeout_s=self.timeout_s,
        )
        return json.dumps(
            {
                "argv": argv,
                "returncode": code,
                "stdout": stdout,
                "stderr": stderr,
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
        isolated_evaluator: RepositoryEvaluatorBackend | None = None,
    ) -> None:
        self.catalog = catalog
        self.rollout = WorkspaceAgentRollout(catalog, agent, Path(run_root) / "workspaces")
        self.verifier = CatalogEnvironmentVerifier(catalog, version=verifier_version, isolated_backend=isolated_evaluator)
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
