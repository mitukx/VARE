import math
from subject import logsumexp


def test_small_values():
    got = logsumexp([0.0, 0.0])
    assert abs(got - math.log(2.0)) < 1e-12


def test_large_values_are_finite():
    got = logsumexp([1000.0, 1000.0])
    assert math.isfinite(got)
    assert abs(got - (1000.0 + math.log(2.0))) < 1e-9
