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

    After promote() returns normally, active_policy() must acknowledge the
    installed candidate identity. A publication exception or acknowledgment
    mismatch requires external recovery, not automatic candidate discard.
    Accepted publication is owned by the engine until the hook is terminal and
    its acknowledgment is checked. With a durable promotion journal, caller
    cancellation after dispatch is deferred through commit and reported in the
    RoundResult. Without a journal, cancellation is delivered to the owned
    hook, drained, then propagated while the loop is fenced. A hook that
    suppresses its own cancellation must clear its task's cancellation state
    before it can acknowledge success.

    Loops with the same hooks object share a runtime coordinator. Distinct
    proxies for one mutable runtime must share runtime_coordinator explicitly
    or the same PromotionJournal database. All owned work must be terminal
    before a hook returns/raises; detached workers are outside this contract.
    """

    def active_policy(self) -> tuple[str, int]: ...

    async def rollout(self, task: Task, policy_id: str, policy_version: int, step: int) -> Attempt: ...

    async def train_candidate(self, incumbent_id: str, experiences: Sequence[Experience]) -> str: ...

    async def evaluate(self, policy_id: str) -> EvaluationReport: ...

    async def promote(self, candidate_id: str) -> None: ...

    async def discard(self, candidate_id: str) -> None: ...
