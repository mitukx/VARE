import asyncio
from vare.types import Attempt, Task, Verification
from vare.verifiers import VerifierEnsemble, VerifierMember


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
