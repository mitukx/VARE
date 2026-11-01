from __future__ import annotations

from typing import Protocol, Sequence

from .types import Experience, FailureCluster, Task


class FailureDrivenTaskGenerator(Protocol):
    """Generate new tasks/environments targeted at observed failure clusters."""

    async def generate(
        self,
        failures: Sequence[FailureCluster],
        experiences: Sequence[Experience],
        limit: int,
    ) -> list[Task]: ...


class VerifierRefreshController(Protocol):
    """Own verifier fitting/recalibration and bump verifier versions on refresh."""

    async def maybe_refresh(
        self,
        failures: Sequence[FailureCluster],
        experiences: Sequence[Experience],
    ) -> bool: ...
