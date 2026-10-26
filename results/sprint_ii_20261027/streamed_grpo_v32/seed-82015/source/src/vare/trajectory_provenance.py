"""Replay explicit event ancestry without claiming semantic causality.

Only parent_event_id references emitted by a trusted recorder are observed
dependencies. This is not a taint-tracking runtime or authority delegation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .trajectory_safety import (
    TrajectoryEvent, TrajectoryPolicy, audit_trajectory,
)

_UNTRUSTED = frozenset({"retrieved_webpage", "tool_output", "unknown"})
_ACTIONS = frozenset({"tool_call", "network_request", "artifact_write"})


@dataclass(frozen=True, slots=True)
class ProvenanceNode:
    event_id: str
    kind: str
    source_origin: str
    parent_event_id: str | None
    untrusted_ancestor_ids: tuple[str, ...]
    # This means *recorded ancestry*, not proof of behavioral influence.
    has_untrusted_ancestry: bool


@dataclass(frozen=True, slots=True)
class ProvenanceReplay:
    run_id: str
    policy_id: str
    ledger_head_sha256: str
    nodes: tuple[ProvenanceNode, ...]
    actions_with_untrusted_ancestry: tuple[str, ...]
    actions_with_missing_parent: tuple[str, ...] = ()

    @property
    def observed_untrusted_action_count(self) -> int:
        return len(self.actions_with_untrusted_ancestry)


def replay_provenance(
    path: str | Path,
    *,
    run_id: str,
    policy_id: str,
    policy: TrajectoryPolicy,
) -> ProvenanceReplay:
    """Validate trace first, then reconstruct explicitly recorded single-parent DAG.

    External text can legitimately influence an authorized action. The action
    is *not* rejected merely for having untrusted ancestry. Tool privileges
    must always be independently enforced by the runtime.
    """
    audit = audit_trajectory(path, run_id=run_id, policy_id=policy_id, policy=policy)
    events: dict[str, ProvenanceNode] = {}
    flagged: list[str] = []
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    for row in rows[1:-1]:
        document = json.loads(row)
        payload = document["payload"]
        event = TrajectoryEvent(**payload)
        parent = events.get(event.parent_event_id) if event.parent_event_id else None
        seeds = set(parent.untrusted_ancestor_ids) if parent is not None else set()
        if event.source_origin in _UNTRUSTED:
            seeds.add(event.event_id)
        node = ProvenanceNode(
            event_id=event.event_id,
            kind=event.kind,
            source_origin=event.source_origin,
            parent_event_id=event.parent_event_id,
            untrusted_ancestor_ids=tuple(sorted(seeds)),
            has_untrusted_ancestry=bool(seeds),
        )
        events[event.event_id] = node
        if event.kind in _ACTIONS and node.has_untrusted_ancestry:
            flagged.append(event.event_id)
    return ProvenanceReplay(
        run_id=run_id,
        policy_id=policy_id,
        ledger_head_sha256=audit.ledger_head_sha256,
        nodes=tuple(events.values()),
        actions_with_untrusted_ancestry=tuple(flagged),
    )
