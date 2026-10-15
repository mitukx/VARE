"""Trusted, explicitly mediated tool dispatch for bounded TASE experiments.

This is a cooperative adapter, not an OS sandbox. Any call bypassing this
adapter is outside its coverage; no untrusted code may access its recorder.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar
from uuid import uuid4

from .trajectory_safety import (
    TrajectoryEvent, TrajectoryPolicy, TrustedTrajectoryRecorder,
    egress_host_allowed as egress_host_allowed,
)

T = TypeVar("T")


class TrajectoryActionDenied(PermissionError):
    """The frozen policy rejected an attempted action before dispatch."""


class AuditedDispatcher:
    """Record policy decision before a side effect, then a result observation.

    The trusted caller owns close() and must keep the adapter/recorder out of
    reach of the candidate. Redirect hops require separate mediation.
    """

    def __init__(self, *, recorder: TrustedTrajectoryRecorder, policy: TrajectoryPolicy) -> None:
        self.recorder = recorder
        self.policy = policy
        self._events = 0
        self._closed = False
        self._prefix = uuid4().hex
        self._steps = 0
        self._inflight = 0
        self._capture_failed = False

    def _emit(self, *, kind: str, source_origin: str, authorization: str,
              tool: str = "", action: str = "", target: str = "",
              parent_event_id: str | None = None) -> str:
        self._steps += 1
        event_id = f"{self._prefix}-{self._steps}"
        try:
            self.recorder.record(TrajectoryEvent(
                event_id=event_id, kind=kind, source_origin=source_origin,
                authorization=authorization, tool=tool, action=action,
                target=target, parent_event_id=parent_event_id,
            ))
        except BaseException:
            # Persistence may have failed before or after publication. Never
            # certify a possibly incomplete capture or dispatch more effects.
            self._capture_failed = True
            raise
        self._events += 1
        return event_id

    async def execute(
        self,
        tool: str,
        call: Callable[[], Awaitable[T]],
        *,
        action: str = "",
        target: str = "",
        source_origin: str = "agent",
        network_url: str | None = None,
    ) -> T:
        """Execute only after synchronous allow/deny journal persistence.

        An external HTTP adapter must disable automatic redirects and call
        execute again for each redirected request. Direct process sockets,
        subprocess execution and other unmediated calls are not covered.
        """
        if self._closed:
            raise RuntimeError("dispatcher already closed")
        if self._capture_failed:
            raise RuntimeError("trajectory capture failed; retain evidence and use a new dispatcher")
        if not callable(call):
            raise TypeError("call must be async callable")
        self._inflight += 1
        try:
            allowed_tool = tool in self.policy.allowed_tools
            parent = self._emit(kind="tool_call", source_origin=source_origin,
                                authorization="allow" if allowed_tool else "deny",
                                tool=tool, action=action, target=target)
            if not allowed_tool:
                raise TrajectoryActionDenied(f"tool not authorized: {tool}")
            if network_url is not None:
                allowed_network = egress_host_allowed(network_url, self.policy.allowed_egress_hosts)
                self._emit(kind="network_request", source_origin=source_origin,
                           authorization="allow" if allowed_network else "deny",
                           tool=tool, action=action, target=network_url, parent_event_id=parent)
                if not allowed_network:
                    raise TrajectoryActionDenied("egress host not authorized")
            try:
                result = await call()
            except BaseException:
                self._emit(kind="tool_result", source_origin="tool_output",
                           authorization="not_applicable", action="error",
                           parent_event_id=parent)
                raise
            self._emit(kind="tool_result", source_origin="tool_output",
                       authorization="not_applicable", action="success",
                       parent_event_id=parent)
            return result
        finally:
            self._inflight -= 1

    def close(self, *, expected_event_count: int) -> None:
        if self._closed:
            raise RuntimeError("dispatcher already closed")
        if self._inflight:
            raise RuntimeError("cannot close trajectory while tool calls are pending")
        if self._capture_failed:
            raise RuntimeError("cannot close trajectory after capture failure")
        if type(expected_event_count) is not int or expected_event_count < 0:
            raise ValueError("expected_event_count must be nonnegative")
        # Must be supplied by an independent trusted runtime dispatch counter.
        # Never silently infer coverage from the recorder itself.
        try:
            self.recorder.close(expected_event_count=expected_event_count)
        except BaseException:
            self._capture_failed = True
            raise
        self._closed = True
