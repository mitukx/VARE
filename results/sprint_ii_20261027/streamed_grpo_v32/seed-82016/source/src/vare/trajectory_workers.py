"""Bounded single-host async workers over the durable trajectory queues."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import time

from .durable import FenceError
from .trajectory_store import AgentStep, TrajectoryStore
from .types import Verification


async def owned_io(fn, *args, **kwargs):
    """Drain a submitted DB operation before cancellation releases ownership."""
    task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
        except BaseException:
            break
    result = task.result()
    if cancelled is not None:
        raise cancelled
    return result


async def _heartbeat(store, lease, seconds):
    while True:
        await asyncio.sleep(seconds / 3)
        await owned_io(store.heartbeat, lease, lease_seconds=seconds)


async def _worker(store, stage, worker_id, operation, stop, *, lease_seconds, poll_seconds, stats):
    while not stop.is_set():
        lease = await owned_io(store.claim, stage, worker_id, lease_seconds=lease_seconds)
        if lease is None:
            started = time.perf_counter()
            await asyncio.sleep(poll_seconds)
            stats["idle_s"] = stats.get("idle_s", 0.) + time.perf_counter() - started
            continue
        started = time.perf_counter()
        stats["claims"] = stats.get("claims", 0) + 1
        heartbeat = asyncio.create_task(_heartbeat(store, lease, lease_seconds))
        body = asyncio.create_task(operation(lease))
        try:
            done, _ = await asyncio.wait((heartbeat, body), return_when=asyncio.FIRST_COMPLETED)
            # A dead/expired heartbeat cancels work; it must not authorize a commit.
            if heartbeat in done:
                heartbeat.result()
                raise RuntimeError("heartbeat stopped unexpectedly")
            body.result()
        except BaseException:
            body.cancel()
            await asyncio.gather(body, return_exceptions=True)
            # The cursor survived. Retry records the failed attempt; a stale
            # worker is already represented by its durable expiry/cancellation.
            try:
                await owned_io(store.retry, lease)
            except FenceError:
                pass
            raise
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            stats["owned_work_s"] = stats.get("owned_work_s", 0.) + time.perf_counter() - started


async def execute_trajectories(
    store: TrajectoryStore, worker_id: str,
    step_fn: Callable[[dict, str], Awaitable[AgentStep]], stop: asyncio.Event,
    *, lease_seconds=5, poll_seconds=.005, step_timeout=30, stats=None,
):
    """step_fn resumes bounded JSON state; its tools may execute again on retry.

    The worker terminates on an exception after persisting the retry transition.
    An operator/supervisor may restart it; failures are never silently swallowed.
    """
    if step_timeout <= 0 or step_timeout > 3600:
        raise ValueError("step timeout must be in (0,3600]")

    async def execute(lease):
        resumed = await owned_io(store.resume_state, lease.trajectory_id)
        while resumed["next_step"] < resumed["spec"]["expected_steps"]:
            async with asyncio.timeout(step_timeout):
                step = await step_fn(resumed, lease.trajectory_id)
            committed_head = await owned_io(
                store.append_step, lease, step, return_head=True
            )
            if committed_head is False:
                # An idempotent duplicate may follow an interrupted worker.
                resumed = await owned_io(store.resume_state, lease.trajectory_id)
            else:
                resumed = {
                    **resumed,
                    "next_step": step.index + 1,
                    "state": step.state,
                    "head": committed_head,
                }
        await owned_io(store.finish, lease)

    await _worker(store, "execute", worker_id, execute, stop,
                  lease_seconds=lease_seconds, poll_seconds=poll_seconds, stats={} if stats is None else stats)


async def verify_trajectories(
    store: TrajectoryStore, worker_id: str,
    verify_fn: Callable[[TrajectoryStore, str], Awaitable[Verification]], stop: asyncio.Event,
    *, lease_seconds=5, poll_seconds=.005, verification_timeout=30, stats=None,
):
    if verification_timeout <= 0 or verification_timeout > 3600:
        raise ValueError("verification timeout must be in (0,3600]")

    async def verify(lease):
        async with asyncio.timeout(verification_timeout):
            value = await verify_fn(store, lease.trajectory_id)
        await owned_io(store.record_verification, lease, value)

    await _worker(store, "verify", worker_id, verify, stop,
                  lease_seconds=lease_seconds, poll_seconds=poll_seconds, stats={} if stats is None else stats)
