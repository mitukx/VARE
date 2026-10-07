from __future__ import annotations

from typing import Protocol, Sequence

from .types import Attempt, EvaluationReport, Experience, Task, Verification


class Verifier(Protocol):
    name: str
    version: int
    trusted: bool

    async def verify(self, attempt: Attempt) -> Verification: ...


class LoopHooks(Protocol):
    def active_policy(self) -> tuple[str, int]: ...

    async def rollout(self, task: Task, policy_id: str, policy_version: int, step: int) -> Attempt: ...

    async def train_candidate(self, incumbent_id: str, experiences: Sequence[Experience]) -> str: ...

    async def evaluate(self, policy_id: str) -> EvaluationReport: ...

    async def promote(self, candidate_id: str) -> None: ...

    async def discard(self, candidate_id: str) -> None: ...
