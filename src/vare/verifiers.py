from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .interfaces import Verifier
from .types import Attempt, Verification


@dataclass(frozen=True, slots=True)
class VerifierMember:
    verifier: Verifier
    weight: float = 1.0


class VerifierEnsemble:
    """Weighted verifier ensemble with explicit disagreement accounting."""

    def __init__(self, members: list[VerifierMember]) -> None:
        if not members:
            raise ValueError("VerifierEnsemble requires at least one member")
        if any(m.weight <= 0 for m in members):
            raise ValueError("verifier weights must be positive")
        self.members = members

    @property
    def active_version(self) -> int:
        return max(m.verifier.version for m in self.members)

    async def verify(self, attempt: Attempt) -> Verification:
        results = await asyncio.gather(*(m.verifier.verify(attempt) for m in self.members))
        total = sum(m.weight for m in self.members)
        score = sum(m.weight * r.score for m, r in zip(self.members, results)) / total
        pass_prob = sum(m.weight * float(r.passed) for m, r in zip(self.members, results)) / total
        confidence = sum(m.weight * r.confidence for m, r in zip(self.members, results)) / total
        disagreement = sum(m.weight * abs(r.score - score) for m, r in zip(self.members, results)) / total
        trusted_members = [(m, r) for m, r in zip(self.members, results) if r.trusted]
        trusted = bool(trusted_members)
        if trusted_members:
            # A trusted verifier owns terminal pass/fail; the ensemble still exposes disagreement.
            trusted_vote = sum(m.weight * float(r.passed) for m, r in trusted_members)
            trusted_total = sum(m.weight for m, _ in trusted_members)
            passed = trusted_vote >= 0.5 * trusted_total
        else:
            passed = pass_prob >= 0.5
        return Verification(
            score=score,
            passed=passed,
            confidence=confidence,
            verifier_version=self.active_version,
            verifier_name="ensemble[" + ",".join(r.verifier_name for r in results) + "]",
            disagreement=disagreement,
            trusted=trusted,
            metadata={"members": [r.metadata | {"name": r.verifier_name, "score": r.score} for r in results]},
        )


class FunctionalAttemptVerifier:
    """Dependency-free verifier wrapper over a synchronous score function."""

    def __init__(self, fn, *, name: str = "functional", version: int = 1, trusted: bool = False) -> None:
        self.fn = fn
        self.name = name
        self.version = version
        self.trusted = trusted

    async def verify(self, attempt: Attempt) -> Verification:
        value = self.fn(attempt)
        if isinstance(value, Verification):
            return value
        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError("functional verifier score must be in [0,1]")
        return Verification(
            score=score,
            passed=score >= 0.5,
            confidence=1.0,
            verifier_version=self.version,
            verifier_name=self.name,
            trusted=self.trusted,
        )
