import asyncio
import math
import pytest
from vare.types import Attempt, Task, Verification
from vare.verifiers import FunctionalAttemptVerifier, VerifierEnsemble, VerifierMember


class V:
    trusted = False
    def __init__(self, name, score, version=1, trusted=False):
        self.name, self.score, self.version, self.trusted = name, score, version, trusted
    async def verify(self, attempt):
        return Verification(self.score, self.score >= 0.5, 1.0, self.version, self.name, trusted=self.trusted)


def test_ensemble_reports_disagreement():
    e = VerifierEnsemble([VerifierMember(V("a", 1.0)), VerifierMember(V("b", 0.0))])
    a = Attempt(Task("t", "p"), "x", "p", 0, 0)
    r = asyncio.run(e.verify(a))
    assert r.score == 0.5
    assert r.disagreement == 0.5


@pytest.mark.parametrize(
    "changes",
    [
        {"score": 2.0},
        {"score": math.nan},
        {"confidence": -0.1},
        {"confidence": math.inf},
        {"disagreement": 1.1},
        {"passed": 1},
        {"trusted": "yes"},
        {"verifier_version": -1},
        {"verifier_version": True},
        {"verifier_name": "  "},
        {"metadata": []},
    ],
)
def test_ensemble_rejects_malformed_structured_verification(changes):
    verification = Verification(0.75, True, 1.0, 1, "fixture", trusted=True)
    for name, value in changes.items():
        setattr(verification, name, value)
    class ReturnsMalformed:
        name = "fixture"
        version = 1
        trusted = True

        async def verify(self, attempt):
            return verification

    attempt = Attempt(Task("t", "p"), "x", "p", 0, 0)
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(VerifierEnsemble([VerifierMember(ReturnsMalformed())]).verify(attempt))


def test_functional_verifier_snapshots_and_validates_mutable_result():
    verification = Verification(0.75, True, 1.0, 1, "fixture", trusted=True)

    def mutate_before_return(attempt):
        verification.score = 2.0
        return verification

    attempt = Attempt(Task("t", "p"), "x", "p", 0, 0)
    with pytest.raises(ValueError, match="score"):
        asyncio.run(FunctionalAttemptVerifier(mutate_before_return).verify(attempt))
