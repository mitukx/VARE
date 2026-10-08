from collections import Counter
import math

from scripts.run_grpo_group_audit_identifiability_v1 import (
    binomial_half,
    clean_group,
    normalized_advantages,
)
import random


def test_two_worlds_have_same_single_item_law_but_different_group_signal():
    for world in ("A", "B"):
        possible = (
            ((0, 0), (1, 1)) if world == "A" else ((0, 1), (1, 0))
        )
        assert sum(pair[0] for pair in possible) / len(possible) == 0.5
        assert sum(pair[1] for pair in possible) / len(possible) == 0.5

    assert normalized_advantages((0, 0)) == (0.0, 0.0)
    assert normalized_advantages((1, 1)) == (0.0, 0.0)
    assert normalized_advantages((1, 0)) == (1.0, -1.0)
    assert normalized_advantages((0, 1)) == (-1.0, 1.0)


def test_item_only_audit_count_law_is_identical_and_group_audit_separates():
    worlds = {
        "A": ((0, 0), (1, 1)),
        "B": ((0, 1), (1, 0)),
    }
    item_laws = {}
    proxy_positive_laws = {}
    for world, pairs in worlds.items():
        item_laws[world] = Counter(
            (member, pair[member]) for pair in pairs for member in (0, 1)
        )
        proxy_positive_laws[world] = Counter((0, pair[0]) for pair in pairs)
    assert item_laws["A"] == item_laws["B"]
    assert proxy_positive_laws["A"] == proxy_positive_laws["B"]

    # Consequently, both worlds induce Binomial(200, 1/2) audited-positive counts.
    assert math.isclose(sum(binomial_half(200)), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    # A complete group is always tied in A and always discordant in B.
    for seed in range(100):
        assert clean_group(random.Random(seed), "A")[0] == clean_group(random.Random(seed), "A")[1]
        assert clean_group(random.Random(seed), "B")[0] != clean_group(random.Random(seed), "B")[1]
