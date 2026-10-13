from __future__ import annotations

from typing import Protocol, Sequence

from .types import Attempt, EvaluationReport, Experience, Task, Verification


class Verifier(Protocol):
    name: str
    version: int
    trusted: bool

    async def verify(self, attempt: Attempt) -> Verification: ...


class LoopHooks(Protocol):
    """Policy operations with hook-owned resource cleanup.

    An evaluation coroutine must not terminate while its owned resource work
    continues, including after cancellation. Hooks must protect their own
    cleanup against accepted cancellation. A hook suppressing cancellation to
    resume ordinary work must also clear its task's cancellation request state;
    otherwise the engine treats it as already unwinding and waits for it.
    """

    def active_policy(self) -> tuple[str, int]: ...

    async def rollout(self, task: Task, policy_id: str, policy_version: int, step: int) -> Attempt: ...

    async def train_candidate(self, incumbent_id: str, experiences: Sequence[Experience]) -> str: ...

    async def evaluate(self, policy_id: str) -> EvaluationReport: ...

    async def promote(self, candidate_id: str) -> None: ...

    async def discard(self, candidate_id: str) -> None: ...
