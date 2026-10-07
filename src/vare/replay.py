from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from itertools import count
from typing import Callable

from .config import ReplayConfig
from .types import Experience


@dataclass(order=True)
class _HeapItem:
    priority: float
    seq: int
    experience: Experience = field(compare=False)


class PrioritizedReplay:
    """Bounded replay that combines failure value, verifier uncertainty, and freshness."""

    def __init__(self, config: ReplayConfig, seed: int = 0) -> None:
        self.config = config
        self._heap: list[_HeapItem] = []
        self._seq = count()
        self._rng = random.Random(seed)

    def _priority(self, exp: Experience, freshness: float) -> float:
        failure = self.config.failure_bonus if not exp.verification.passed else 1.0
        uncertainty = 1.0 + self.config.disagreement_bonus * exp.verification.disagreement
        signal = max(0.05, abs(exp.verification.score - 0.5) * 2.0)
        p = failure * uncertainty * signal * max(self.config.freshness_floor, freshness)
        return max(self.config.min_priority, p)

    def add(self, exp: Experience, freshness: float) -> None:
        exp.priority = self._priority(exp, freshness)
        item = _HeapItem(exp.priority, next(self._seq), exp)
        if len(self._heap) < self.config.capacity:
            heapq.heappush(self._heap, item)
        elif item.priority > self._heap[0].priority:
            heapq.heapreplace(self._heap, item)

    def sample(self, n: int) -> list[Experience]:
        if not self._heap or n <= 0:
            return []
        items = list(self._heap)
        weights = [max(self.config.min_priority, x.priority) for x in items]
        if n >= len(items):
            return [x.experience for x in sorted(items, reverse=True)]
        chosen = self._rng.choices(items, weights=weights, k=n)
        # De-duplicate while preserving importance order as much as possible.
        seen: set[int] = set()
        out: list[Experience] = []
        for x in sorted(chosen, reverse=True):
            key = id(x.experience)
            if key not in seen:
                seen.add(key)
                out.append(x.experience)
        if len(out) < n:
            for x in sorted(items, reverse=True):
                key = id(x.experience)
                if key not in seen:
                    seen.add(key)
                    out.append(x.experience)
                    if len(out) == n:
                        break
        return out


    def sample_grouped(self, n: int, *, metadata_key: str = "vare_rollout_group") -> list[Experience]:
        """Sample whole rollout groups without breaking group-relative objectives.

        Groups are weighted by their strongest member. The returned batch may
        exceed ``n`` by at most one group so a GRPO/PPO-style objective never
        receives a silently truncated comparison group.
        """
        if not self._heap or n <= 0:
            return []
        groups: dict[str, list[_HeapItem]] = {}
        for item in self._heap:
            key = item.experience.attempt.metadata.get(metadata_key)
            if key is None:
                key = item.experience.attempt.task.metadata.get(metadata_key)
            if key is None:
                key = f"ungrouped:{id(item.experience)}"
            groups.setdefault(str(key), []).append(item)
        remaining = dict(groups)
        out: list[Experience] = []
        while remaining and len(out) < n:
            keys = list(remaining)
            weights = [max(x.priority for x in remaining[k]) for k in keys]
            chosen_key = self._rng.choices(keys, weights=weights, k=1)[0]
            items = sorted(remaining.pop(chosen_key), reverse=True)
            out.extend(x.experience for x in items)
        return out


    def sample_current(
        self,
        n: int,
        *,
        freshness_fn: Callable[[Experience], float | None],
        grouped: bool = True,
        metadata_key: str = "vare_rollout_group",
    ) -> list[Experience]:
        """Sample using freshness recomputed against the *current* policy/verifier.

        ``freshness_fn`` returns ``None`` for currently inadmissible experience.
        Dynamic priorities are recomputed rather than reusing the priority from
        insertion time, preventing a once-fresh trajectory from remaining hot
        after policy or verifier versions advance.
        """
        if not self._heap or n <= 0:
            return []
        eligible: list[tuple[float, _HeapItem]] = []
        for item in self._heap:
            freshness = freshness_fn(item.experience)
            if freshness is None:
                continue
            eligible.append((self._priority(item.experience, freshness), item))
        if not eligible:
            return []
        if not grouped:
            items = [item for _, item in eligible]
            weights = [priority for priority, _ in eligible]
            if n >= len(items):
                return [x.experience for _, x in sorted(eligible, key=lambda z: z[0], reverse=True)]
            chosen = self._rng.choices(range(len(items)), weights=weights, k=n)
            order = sorted(set(chosen), key=lambda i: weights[i], reverse=True)
            out = [items[i].experience for i in order]
            if len(out) < n:
                for i in sorted(range(len(items)), key=lambda i: weights[i], reverse=True):
                    if items[i].experience not in out:
                        out.append(items[i].experience)
                        if len(out) == n:
                            break
            return out

        groups: dict[str, list[tuple[float, _HeapItem]]] = {}
        for priority, item in eligible:
            key = item.experience.attempt.metadata.get(metadata_key)
            if key is None:
                key = item.experience.attempt.task.metadata.get(metadata_key)
            if key is None:
                key = f"ungrouped:{id(item.experience)}"
            groups.setdefault(str(key), []).append((priority, item))
        remaining = dict(groups)
        out: list[Experience] = []
        while remaining and len(out) < n:
            keys = list(remaining)
            weights = [max(priority for priority, _ in remaining[k]) for k in keys]
            chosen_key = self._rng.choices(keys, weights=weights, k=1)[0]
            members = sorted(remaining.pop(chosen_key), key=lambda z: z[0], reverse=True)
            out.extend(item.experience for _, item in members)
        return out

    def __len__(self) -> int:
        return len(self._heap)
