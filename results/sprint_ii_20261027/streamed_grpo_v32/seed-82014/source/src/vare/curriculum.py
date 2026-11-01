from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from .types import FailureCluster, Task


class AdaptiveCurriculum:
    """Reweights future tasks toward families implicated by recent failures."""

    def __init__(self, seed: int = 0, floor: float = 1.0, failure_scale: float = 0.25) -> None:
        self._rng = random.Random(seed)
        self.floor = floor
        self.failure_scale = failure_scale
        self.family_weights: dict[str, float] = defaultdict(lambda: floor)

    def update(self, failures: Sequence[FailureCluster]) -> None:
        for cluster in failures:
            bump = self.failure_scale * cluster.count
            for family in cluster.task_families:
                self.family_weights[family] = max(self.floor, self.family_weights[family] + bump)

    def choose(self, tasks: Sequence[Task], n: int) -> list[Task]:
        if not tasks or n <= 0:
            return []
        weights = [self.family_weights[t.family] for t in tasks]
        return self._rng.choices(list(tasks), weights=weights, k=n)
