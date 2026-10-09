from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from itertools import count
from typing import Callable, Sequence

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

    @staticmethod
    def group_has_homogeneous_provenance(experiences: Sequence[Experience]) -> bool:
        if not experiences:
            return False
        first = experiences[0]
        provenance = (
            first.attempt.policy_id,
            first.attempt.policy_version,
            first.verification.verifier_version,
        )
        return all(
            (
                exp.attempt.policy_id,
                exp.attempt.policy_version,
                exp.verification.verifier_version,
            ) == provenance
            for exp in experiences[1:]
        )

    @staticmethod
    def _group_is_complete(items: list[_HeapItem]) -> bool:
        if not items:
            return False
        first_prompt = items[0].experience.attempt.task.prompt
        if (not isinstance(first_prompt, str)
                or any(item.experience.attempt.task.prompt != first_prompt for item in items)):
            return False

        if not PrioritizedReplay.group_has_homogeneous_provenance(
            [item.experience for item in items]
        ):
            return False

        declared_sizes = set()
        for item in items:
            attempt = item.experience.attempt
            size = attempt.metadata.get("vare_rollout_group_size")
            if size is None:
                size = attempt.task.metadata.get("vare_rollout_group_size")
            if size is None:
                continue
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                return False
            declared_sizes.add(size)
        if len(declared_sizes) > 1:
            return False
        if not declared_sizes:
            return True
        return len(items) == next(iter(declared_sizes))

    def add(self, exp: Experience, freshness: float) -> None:
        exp.priority = self._priority(exp, freshness)
        item = _HeapItem(exp.priority, next(self._seq), exp)
        if len(self._heap) < self.config.capacity:
            heapq.heappush(self._heap, item)
        elif item.priority > self._heap[0].priority:
            heapq.heapreplace(self._heap, item)

    @staticmethod
    def _group_id(exp: Experience) -> str | None:
        group_id = exp.attempt.metadata.get("vare_rollout_group")
        if group_id is None:
            group_id = exp.attempt.task.metadata.get("vare_rollout_group")
        if group_id is None:
            return None
        return str(group_id)

    @staticmethod
    def _declared_group_size(exp: Experience) -> int | None:
        size = exp.attempt.metadata.get("vare_rollout_group_size")
        if size is None:
            size = exp.attempt.task.metadata.get("vare_rollout_group_size")
        if size is None:
            return None
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            return -1
        return size

    def _remove_partial_declared_groups(self) -> None:
        groups: dict[str, list[_HeapItem]] = {}
        for item in self._heap:
            exp = item.experience
            group_id = self._group_id(exp)
            size = self._declared_group_size(exp)
            if group_id is not None and size is not None:
                groups.setdefault(group_id, []).append(item)

        partial_ids = {
            group_id for group_id, items in groups.items()
            if not self._group_is_complete(items)
        }
        if partial_ids:
            self._heap = [
                item for item in self._heap
                if self._group_id(item.experience) not in partial_ids
            ]
            heapq.heapify(self._heap)

    def add_group(
        self,
        experiences: Sequence[Experience],
        freshnesses: Sequence[float],
    ) -> bool:
        """Admit one complete declared rollout group atomically.

        If the group cannot be stored whole, the replay contents are left
        unchanged (apart from removing pre-existing partial declared groups).
        Complete groups are evicted as units; ungrouped legacy items remain
        individual eviction units.
        """
        self._remove_partial_declared_groups()

        if not experiences or len(experiences) != len(freshnesses):
            return False
        if len({id(exp) for exp in experiences}) != len(experiences):
            return False

        group_ids = {self._group_id(exp) for exp in experiences}
        declared_sizes = {self._declared_group_size(exp) for exp in experiences}
        if (len(group_ids) != 1 or None in group_ids
                or len(declared_sizes) != 1 or None in declared_sizes
                or -1 in declared_sizes):
            return False
        group_id = next(iter(group_ids))
        group_size = next(iter(declared_sizes))
        if not group_id.strip() or len(experiences) != group_size:
            return False
        if group_size > self.config.capacity:
            return False
        if any(
            self._group_id(item.experience) == group_id for item in self._heap
        ):
            return False
        existing_ids = {id(item.experience) for item in self._heap}
        if any(id(exp) in existing_ids for exp in experiences):
            return False

        new_items = [
            _HeapItem(self._priority(exp, freshness), -1, exp)
            for exp, freshness in zip(experiences, freshnesses, strict=True)
        ]
        if not self._group_is_complete(new_items):
            return False
        new_group_priority = max(item.priority for item in new_items)
        needed = max(0, len(self._heap) + len(new_items) - self.config.capacity)
        if needed:
            grouped: dict[str, list[_HeapItem]] = {}
            units: list[tuple[float, int, list[_HeapItem]]] = []
            for item in self._heap:
                exp = item.experience
                group_id_existing = self._group_id(exp)
                group_size_existing = self._declared_group_size(exp)
                if group_id_existing is not None and group_size_existing is not None:
                    grouped.setdefault(group_id_existing, []).append(item)
                else:
                    units.append((item.priority, item.seq, [item]))

            for members in grouped.values():
                units.append((
                    max(item.priority for item in members),
                    min(item.seq for item in members),
                    members,
                ))
            units.sort(key=lambda unit: (unit[0], unit[1]))

            victims: list[_HeapItem] = []
            freed = 0
            for priority, _, members in units:
                if freed >= needed:
                    break
                # Equal priority admits the newer group and prevents an
                # equal-priority stream from being permanently rejected.
                if new_group_priority < priority:
                    return False
                victims.extend(members)
                freed += len(members)
            if freed < needed:
                return False

            victim_ids = {id(item) for item in victims}
            self._heap = [item for item in self._heap if id(item) not in victim_ids]

        for item in new_items:
            item.seq = next(self._seq)
            item.experience.priority = item.priority
            self._heap.append(item)
        heapq.heapify(self._heap)
        return True

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
        groups = {key: members for key, members in groups.items()
                  if self._group_is_complete(members)}
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
        after policy or verifier versions advance. With grouped sampling, one
        inadmissible member excludes its whole comparison group.
        """
        if not self._heap or n <= 0:
            return []
        eligible: list[tuple[float, _HeapItem]] = []
        group_candidates: dict[str, list[tuple[float | None, _HeapItem]]] = {}
        for item in self._heap:
            freshness = freshness_fn(item.experience)
            if grouped:
                key = item.experience.attempt.metadata.get(metadata_key)
                if key is None:
                    key = item.experience.attempt.task.metadata.get(metadata_key)
                if key is None:
                    key = f"ungrouped:{id(item.experience)}"
                priority = (
                    None if freshness is None
                    else self._priority(item.experience, freshness)
                )
                group_candidates.setdefault(str(key), []).append((priority, item))
            elif freshness is not None:
                eligible.append((self._priority(item.experience, freshness), item))

        if not grouped:
            if not eligible:
                return []
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

        groups = {}
        for key, members in group_candidates.items():
            if (all(priority is not None for priority, _ in members)
                    and self._group_is_complete([item for _, item in members])):
                groups[key] = members
        if not groups:
            return []
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
