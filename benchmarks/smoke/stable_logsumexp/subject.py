import math


def logsumexp(xs):
    """Intentionally naive baseline used only to test the environment harness."""
    return math.log(sum(math.exp(x) for x in xs))
