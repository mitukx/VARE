#!/usr/bin/env python3
"""Check that stale members invalidate their full comparison group."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.workspace.resolve() / "src"))

    from vare.config import ReplayConfig
    from vare.replay import PrioritizedReplay
    from vare.types import Attempt, Experience, Task, Verification

    replay = PrioritizedReplay(ReplayConfig(capacity=16), seed=0)
    for index in range(8):
        group = "group-a" if index < 4 else "group-b"
        task = Task(id=f"task-{index}", prompt="fixture")
        attempt = Attempt(
            task=task,
            output=f"response-{index}",
            policy_id="policy-v1",
            policy_version=1,
            created_step=0,
            metadata={"vare_rollout_group": group},
        )
        experience = Experience(
            attempt=attempt,
            verification=Verification(0.75, True, 1.0, 1, "fixture", trusted=True),
            policy_lag=0,
            verifier_lag=0,
            shift_score=0.0,
        )
        replay.add(experience, freshness=1.0)

    stale_ids = {"task-0"}
    freshness_fn = lambda item: None if item.attempt.task.id in stale_ids else 1.0
    selected = replay.sample_current(
        8,
        freshness_fn=freshness_fn,
        grouped=True,
    )
    counts = Counter(item.attempt.metadata["vare_rollout_group"] for item in selected)
    per_item = replay.sample_current(8, freshness_fn=freshness_fn, grouped=False)
    result = {
        "schema_version": 1,
        "task_id": "vare-replay-group-freshness-atomicity-v1",
        "passed": counts.get("group-a", 0) == 0 and counts.get("group-b", 0) == 4
                  and len(per_item) == 7,
        "group_members": 4,
        "stale_members": 1,
        "selected_count": len(selected),
        "selected_task_ids": sorted(item.attempt.task.id for item in selected),
        "group_counts": dict(counts),
        "per_item_selected_count": len(per_item),
        "claim_limit": "One deterministic local CPU replay-contract check; not model-learning or throughput evidence.",
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
