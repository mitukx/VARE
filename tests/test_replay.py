from vare.config import ReplayConfig
from vare.replay import PrioritizedReplay
from vare.types import Attempt, Experience, Task, Verification


def make_exp(passed: bool, disagreement: float = 0.0):
    t = Task("t", "p")
    a = Attempt(t, "x", "p0", 0, 0)
    v = Verification(float(passed), passed, 1.0, 0, "v", disagreement=disagreement)
    return Experience(a, v, 0, 0, 0.0)


def test_failure_gets_higher_priority():
    replay = PrioritizedReplay(ReplayConfig(), seed=1)
    good, bad = make_exp(True), make_exp(False, disagreement=0.4)
    replay.add(good, 1.0)
    replay.add(bad, 1.0)
    assert bad.priority > good.priority
