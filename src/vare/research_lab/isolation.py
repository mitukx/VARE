"""Isolation selection for candidate commands.

``process_group`` reuses the POSIX owner in ``process_owner``. It is not a
secure sandbox: the candidate keeps the caller's uid and can touch any path
that uid can touch. Network denial, when the kernel allows an unprivileged
user and network namespace, only removes network access.

``docker`` reuses ``vare.container_isolation.DockerIsolationRunner`` and does
not fall back to the process group. If the daemon or the pinned image is
unavailable, execution fails closed.
"""
from __future__ import annotations

from pathlib import Path
import secrets
from typing import Sequence

from vare.container_isolation import DockerIsolationRunner, IsolationError, IsolationLimits, IsolationResult

RESIDUAL_RISK = (
    "process_group mode is a resource and bookkeeping boundary, not a secure sandbox. "
    "The candidate runs as the same user. Protected-path and source fingerprints are "
    "checked after the fact and fail closed; they do not prevent a write. A child that "
    "creates a new session can leave the guarded process group. Docker mode is a separate "
    "boundary only when a digest-pinned image is actually launched, and it never falls "
    "back to process_group."
)


def docker_argv(
    *,
    image: str,
    workspace: Path,
    command: Sequence[str],
    wall_time_s: float,
    memory_mb: int,
    max_output_bytes: int,
) -> tuple[IsolationLimits, list[str], str]:
    limits = IsolationLimits(
        image=image,
        timeout_s=wall_time_s,
        memory_mb=memory_mb,
        max_output_bytes=min(max_output_bytes, 512_000),
    )
    name = f"vare-lab-{secrets.token_hex(12)}"
    argv = DockerIsolationRunner().candidate_argv(
        limits,
        name=name,
        allowed_workspace_root=workspace.parent,
        workspace=workspace,
        command=list(command),
    )
    return limits, argv, name


async def docker_execute(argv: Sequence[str], limits: IsolationLimits) -> IsolationResult:
    try:
        return await DockerIsolationRunner().execute(argv, limits)
    except OSError as exc:
        raise IsolationError("Docker could not be started; fail closed") from exc
