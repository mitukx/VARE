"""Docker-backed roles for candidate execution and trusted CPU evaluation.

Security boundary is the independently administered Docker daemon/VM, NOT the
Python launcher. Only operator-owned code may configure images, mounts, keys,
and launch authority. No candidate checkout ever receives the Docker socket.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import secrets
from typing import Sequence

from .attestation import EvidenceTrustRoot
from .types import EvaluationReport


_IMAGE = re.compile(r"[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}\Z")
_ALLOWED_OUTPUT = 512_000


class IsolationError(RuntimeError):
    """Container startup, containment assumptions, or cleanup failed."""


def _image_pinned(image: str) -> None:
    if not isinstance(image, str) or _IMAGE.fullmatch(image) is None:
        raise ValueError("isolation image must be pinned by sha256 digest")


def _mount_source(path: Path, *, directory: bool) -> Path:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink():
        raise IsolationError("mount source must be an absolute, non-symlink operator path")
    # --mount takes comma-separated CSV; refuse ambiguous sources.
    if any(c in str(source) for c in (",", "\n", "\r")):
        raise IsolationError("unsafe Docker mount source")
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise IsolationError("missing Docker mount source") from exc
    if resolved != source or (not resolved.is_dir() if directory else not resolved.is_file()):
        raise IsolationError("Docker mount source changed or has unexpected type")
    return resolved


@dataclass(frozen=True, slots=True)
class IsolationLimits:
    """Per-container process, memory, CPU, timeout, and output limits."""

    image: str
    timeout_s: float = 120.0
    memory_mb: int = 512
    cpus: float = 1.0
    pids_limit: int = 64
    uid: int = 65534
    gid: int = 65534
    max_output_bytes: int = 100_000

    def __post_init__(self) -> None:
        _image_pinned(self.image)
        if (
            not isinstance(self.timeout_s, (int, float))
            or isinstance(self.timeout_s, bool)
            or not math.isfinite(float(self.timeout_s))
            or not 0 < self.timeout_s <= 3600
            or type(self.memory_mb) is not int or not 32 <= self.memory_mb <= 65536
            or not isinstance(self.cpus, (int, float))
            or isinstance(self.cpus, bool)
            or not math.isfinite(float(self.cpus))
            or not 0 < self.cpus <= 32
            or type(self.pids_limit) is not int or not 1 <= self.pids_limit <= 4096
            or type(self.uid) is not int or not 1 <= self.uid <= 65534
            or type(self.gid) is not int or not 1 <= self.gid <= 65534
            or type(self.max_output_bytes) is not int
            or not 1 <= self.max_output_bytes <= _ALLOWED_OUTPUT
        ):
            raise ValueError("invalid or unsafe isolation resource limits")


@dataclass(frozen=True, slots=True)
class IsolationResult:
    returncode: int
    stdout: str
    stderr: str


class DockerIsolationRunner:
    """Trusted Docker client with bounded output, timeout, and forced cleanup.

    Docker credentials and socket stay on the operator side. This does not
    verify daemon/rootless/host security configuration; assess that separately.
    """

    def __init__(self, *, docker_binary: str = "docker") -> None:
        if not isinstance(docker_binary, str) or not docker_binary:
            raise ValueError("docker_binary required")
        self.docker_binary = docker_binary
        # Freeze the daemon/context selection for run and cleanup alike.
        self._client_env = {
            k: os.environ[k]
            for k in ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "XDG_RUNTIME_DIR")
            if k in os.environ
        }

    def _common(self, limits: IsolationLimits, name: str) -> list[str]:
        return [
            self.docker_binary, "run", "--rm", "--pull=never", "--init",
            "--name", name, "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
            "--pids-limit", str(limits.pids_limit),
            "--memory", f"{limits.memory_mb}m",
            "--cpus", str(limits.cpus),
            "--user", f"{limits.uid}:{limits.gid}",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=16m",
        ]

    @staticmethod
    def _workspace(root: Path, workspace: Path) -> Path:
        allowed = _mount_source(root, directory=True)
        actual = _mount_source(workspace, directory=True)
        if actual == allowed or allowed not in actual.parents:
            raise IsolationError("candidate workspace escapes its dedicated root")
        return actual

    def candidate_argv(
        self, limits: IsolationLimits, *, name: str,
        allowed_workspace_root: Path, workspace: Path, command: Sequence[str],
    ) -> list[str]:
        if not command or not all(isinstance(x, str) and x for x in command):
            raise ValueError("candidate command must be a nonempty string vector")
        checked = self._workspace(allowed_workspace_root, workspace)
        return [
            *self._common(limits, name),
            "--mount", f"type=bind,source={checked},target=/workspace",
            "--workdir=/workspace",
            limits.image, *command,
        ]

    def evaluator_argv(
        self, limits: IsolationLimits, *, name: str, policy_id: str,
        checkpoint_store: Path, heldout_set: Path, key_file: Path,
        trust: EvidenceTrustRoot,
    ) -> list[str]:
        if not trust.strict_checkpoint_binding or not trust.promotion_nonce:
            raise ValueError("isolated grader requires strict checkpoint-bound evidence")
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("policy_id required")
        store = _mount_source(checkpoint_store, directory=True)
        heldout = _mount_source(heldout_set, directory=False)
        secret = _mount_source(key_file, directory=False)
        # Defend against accidentally mounting the key/dataset within a
        # checkpoint store that could also be used as a candidate source.
        if (store in heldout.parents or store in secret.parents
                or heldout == secret):
            raise IsolationError("grader mounts must use separate controlled roots")
        return [
            *self._common(limits, name),
            "--mount", f"type=bind,source={store},target=/operator/checkpoints,readonly",
            "--mount", f"type=bind,source={heldout},target=/operator/heldout.json,readonly",
            "--mount", f"type=bind,source={secret},target=/operator/key,readonly",
            "--workdir=/",
            limits.image, "python", "-m", "vare.trusted_evaluation",
            "--checkpoint-store", "/operator/checkpoints",
            "--policy-id", policy_id,
            "--heldout-set", "/operator/heldout.json",
            "--heldout-sha256", str(trust.evaluation_set_sha256),
            "--key-file", "/operator/key",
            "--protocol-sha256", trust.protocol_sha256,
            "--run-id", trust.run_id,
            "--promotion-nonce", trust.promotion_nonce,
        ]

    async def execute(self, argv: Sequence[str], limits: IsolationLimits) -> IsolationResult:
        """Supervise one container, including cancellation and output-overflow.

        A unique name is generated by the caller only through our role
        functions; arbitrary argv is not exposed as an end-user control.
        """
        # Role-specific argv includes a freshly minted container name.
        try:
            index = argv.index("--name")
            name = argv[index + 1]
        except (ValueError, IndexError) as exc:
            raise IsolationError("missing managed container identity") from exc
        if not re.fullmatch(r"vare-[a-z]+-[0-9a-f]{24}", name):
            raise IsolationError("invalid managed container identity")
        # Prevent arbitrary command substitution; caller uses only our builders.
        env = self._client_env
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=env,
            )
        except OSError as exc:
            raise IsolationError("Docker could not be started; fail closed") from exc

        exceeded = asyncio.Event()
        kept = {"stdout": bytearray(), "stderr": bytearray()}
        total = 0

        async def drain(stream: asyncio.StreamReader, stream_name: str) -> None:
            nonlocal total
            while chunk := await stream.read(65536):
                total += len(chunk)
                room = max(0, limits.max_output_bytes - sum(map(len, kept.values())))
                kept[stream_name].extend(chunk[:room])
                if total > limits.max_output_bytes:
                    exceeded.set()

        assert proc.stdout is not None and proc.stderr is not None
        readers = [
            asyncio.create_task(drain(proc.stdout, "stdout")),
            asyncio.create_task(drain(proc.stderr, "stderr")),
        ]
        waiter = asyncio.create_task(proc.wait())
        alarm = asyncio.create_task(exceeded.wait())
        cleanup_failure: Exception | None = None
        try:
            done, _ = await asyncio.wait(
                (waiter, alarm), timeout=limits.timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if alarm in done and exceeded.is_set():
                raise IsolationError("container exceeded output byte limit")
            if waiter not in done:
                raise IsolationError("container exceeded wall-time limit")
            await asyncio.wait_for(asyncio.gather(*readers), timeout=3.0)
            if exceeded.is_set():
                raise IsolationError("container exceeded output byte limit")
            return IsolationResult(
                int(proc.returncode if proc.returncode is not None else 125),
                kept["stdout"].decode("utf-8", errors="replace"),
                kept["stderr"].decode("utf-8", errors="replace"),
            )
        finally:
            # Always ask the daemon to remove the named container. Killing
            # just the local CLI may leave the container running remotely.
            cleanup_task = asyncio.create_task(self._force_remove(name))
            # A cancellation must not abandon an in-flight daemon cleanup.
            # Repeat the shielded wait until the bounded cleanup task exits.
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    continue
                except Exception as exc:
                    cleanup_failure = exc
                    break
            if cleanup_task.done() and cleanup_failure is None:
                try:
                    cleanup_task.result()
                except Exception as exc:
                    cleanup_failure = exc
            if proc.returncode is None:
                proc.kill()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    pass
            for task in (*readers, waiter, alarm):
                if not task.done():
                    task.cancel()
            await asyncio.gather(*readers, waiter, alarm, return_exceptions=True)
            if cleanup_failure is not None:
                raise IsolationError("could not confirm Docker container cleanup") from cleanup_failure

    async def _force_remove(self, name: str) -> None:
        env = self._client_env
        proc = await asyncio.create_subprocess_exec(
            self.docker_binary, "rm", "-f", name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            _out, err = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise IsolationError("Docker forced cleanup timed out") from exc
        if proc.returncode != 0 and b"No such container" not in err and b"No such object" not in err:
            raise IsolationError("Docker daemon did not confirm forced cleanup")

    async def run_candidate(
        self, limits: IsolationLimits, *, allowed_workspace_root: Path,
        workspace: Path, command: Sequence[str],
    ) -> IsolationResult:
        name = "vare-agent-" + secrets.token_hex(12)
        args = self.candidate_argv(
            limits, name=name, allowed_workspace_root=allowed_workspace_root,
            workspace=workspace, command=command,
        )
        return await self.execute(args, limits)

    async def run_evaluator(
        self, limits: IsolationLimits, *, policy_id: str,
        checkpoint_store: Path, heldout_set: Path, key_file: Path,
        trust: EvidenceTrustRoot,
    ) -> EvaluationReport:
        name = "vare-grader-" + secrets.token_hex(12)
        args = self.evaluator_argv(
            limits, name=name, policy_id=policy_id,
            checkpoint_store=checkpoint_store, heldout_set=heldout_set,
            key_file=key_file, trust=trust,
        )
        result = await self.execute(args, limits)
        if result.returncode != 0:
            raise IsolationError("trusted evaluator failed without a valid signed report")
        try:
            raw = json.loads(result.stdout)
            if type(raw) is not dict or set(raw) != {
                "policy_id", "primary", "slices", "cost",
                "verifier_disagreement", "n", "metadata",
            }:
                raise ValueError("unexpected evaluation output shape")
            report = EvaluationReport(**raw)
            if report.policy_id != policy_id or not trust.verifies(report):
                raise ValueError("evaluator identity or attestation mismatch")
            return report
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise IsolationError("invalid or unsigned evaluator response") from exc


@dataclass(frozen=True, slots=True)
class DockerWorkspaceAgent:
    """Optional replacement for CommandWorkspaceAgent in environment campaigns."""

    limits: IsolationLimits
    allowed_workspace_root: Path
    command: tuple[str, ...]
    runner: DockerIsolationRunner

    async def run(self, *, prompt: str, workspace: Path, step: int) -> str:
        if not self.command:
            raise ValueError("isolated agent command required")
        values = {"{workspace}": "/workspace", "{prompt}": prompt, "{step}": str(step)}
        pattern = re.compile(r"\{workspace\}|\{prompt\}|\{step\}")
        argv = [pattern.sub(lambda match: values[match.group(0)], arg) for arg in self.command]
        result = await self.runner.run_candidate(
            self.limits, allowed_workspace_root=self.allowed_workspace_root,
            workspace=workspace, command=argv,
        )
        return json.dumps({
            "argv": argv,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }, sort_keys=True)
