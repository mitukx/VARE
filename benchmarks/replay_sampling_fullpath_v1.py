"""Full entrypoint replay benchmark: before PR #182 versus candidate.

CPU-only: excludes inference, optimizer and end-to-end round.
Run: PYTHONPATH=src python benchmarks/replay_sampling_fullpath_v1.py --output out.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time

from vare.config import ReplayConfig
from vare.replay import PrioritizedReplay
from vare.types import Attempt, Experience, Task, Verification


def legacy_grouped(replay, n):
    # Exact grouped selection path in main@6373d8d (before PR #182).
    groups = {}
    for item in replay._heap:
        key = item.experience.attempt.metadata.get("vare_rollout_group")
        if key is None:
            key = item.experience.attempt.task.metadata.get("vare_rollout_group")
        if key is None:
            key = f"ungrouped:{id(item.experience)}"
        groups.setdefault(str(key), []).append(item)
    groups = {key: members for key, members in groups.items()
              if replay._group_is_complete(members)}
    remaining = dict(groups)
    out = []
    while remaining and len(out) < n:
        keys = list(remaining)
        weights = [max(x.priority for x in remaining[k]) for k in keys]
        chosen = replay._rng.choices(keys, weights=weights, k=1)[0]
        out.extend(x.experience for x in sorted(remaining.pop(chosen), reverse=True))
    return out


def legacy_current(replay, n, freshness_fn):
    # Exact grouped sample_current path in main@6373d8d.
    candidates = {}
    for item in replay._heap:
        fresh = freshness_fn(item.experience)
        key = item.experience.attempt.metadata.get("vare_rollout_group")
        if key is None:
            key = item.experience.attempt.task.metadata.get("vare_rollout_group")
        if key is None:
            key = f"ungrouped:{id(item.experience)}"
        priority = None if fresh is None else replay._priority(item.experience, fresh)
        candidates.setdefault(str(key), []).append((priority, item))
    complete = {}
    for key, members in candidates.items():
        if not replay._group_is_complete([item for _, item in members]):
            continue
        narrowed = []
        for priority, item in members:
            if priority is None:
                narrowed = []
                break
            narrowed.append((priority, item))
        if narrowed:
            complete[key] = narrowed
    if not complete:
        return []
    remaining = dict(complete)
    out = []
    while remaining and len(out) < n:
        keys = list(remaining)
        weights = [max(priority for priority, _ in remaining[k]) for k in keys]
        chosen = replay._rng.choices(keys, weights=weights, k=1)[0]
        ranked = sorted(remaining.pop(chosen), key=lambda z: z[0], reverse=True)
        out.extend(item.experience for _, item in ranked)
    return out


def build_replay(group_count):
    replay = PrioritizedReplay(ReplayConfig(capacity=group_count * 4), seed=0)
    for i in range(group_count):
        task = Task(f"task-{i}", "same benchmark prompt", "coding")
        metadata = {"vare_rollout_group": f"g{i}",
                    "vare_rollout_group_size": 4,
                    "policy_checkpoint_sha256": "a" * 64}
        for j in range(4):
            attempt = Attempt(task, f"response-{j}", "fixed-model", 1, i * 4 + j,
                              metadata=dict(metadata))
            verification = Verification(0.25 + (i % 5) * 0.12, bool(i % 2),
                                        0.95, 1, "trusted", trusted=True,
                                        metadata={"verifier_config_sha256": "b" * 64})
            replay.add(Experience(attempt, verification, 0, 0, 0.0), freshness=0.8)
    return replay


def measure(group_count, requested, repeats, mode):
    replay = build_replay(group_count)
    def fresh(exp):
        return None if int(exp.attempt.task.id[5:]) % 97 == 0 else 0.7
    arms = {
        "baseline": lambda: legacy_current(replay, requested, fresh)
                           if mode == "current" else legacy_grouped(replay, requested),
        "candidate": lambda: replay.sample_current(
            requested, freshness_fn=fresh, grouped=True)
                            if mode == "current" else replay.sample_grouped(requested),
    }
    samples = {"baseline": [], "candidate": []}
    for run in range(repeats + 3):
        order = ("baseline", "candidate") if run % 2 == 0 else ("candidate", "baseline")
        for arm in order:
            replay._rng = random.Random(7331 + run)
            start = time.perf_counter_ns()
            selected = arms[arm]()
            ms = (time.perf_counter_ns() - start) / 1e6
            groups = Counter(x.attempt.metadata["vare_rollout_group"] for x in selected)
            assert len(selected) == requested and len({id(x) for x in selected}) == requested
            assert all(v == 4 for v in groups.values())
            if mode == "current":
                assert all(int(x.attempt.task.id[5:]) % 97 != 0 for x in selected)
            if run >= 3:
                samples[arm].append(ms)
    def stats(values):
        return {"median_ms": statistics.median(values), "p95_ms": sorted(values)[int(0.95*(len(values)-1))],
                "raw_ms": values}
    baseline, candidate = stats(samples["baseline"]), stats(samples["candidate"])
    return {"groups": group_count, "stored_rows": group_count * 4,
            "requested_rows": requested, "mode": mode, "baseline": baseline,
            "candidate": candidate,
            "speedup": baseline["median_ms"] / candidate["median_ms"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 500:
        raise SystemExit("repeats must be in [3,500]")
    out = {
        "scope": "complete VARE replay sampling entrypoints only; no training",
        "baseline_main_sha": "6373d8d91ac76d5f3ba1bae53cdee002b59db16c",
        "platform": platform.platform(), "python": sys.version,
        "replay_sha256": hashlib.sha256(Path("src/vare/replay.py").read_bytes()).hexdigest(),
        "repeats": args.repeats, "results": []
    }
    for count, rows, mode in ((1000, 256, "current"), (4000, 256, "current"),
                              (12500, 256, "current"), (12500, 1024, "current"),
                              (12500, 1024, "grouped")):
        result = measure(count, rows, args.repeats, mode)
        out["results"].append(result)
        print(f"{mode} G={count} requested={rows} old={result['baseline']['median_ms']:.3f}ms "
              f"new={result['candidate']['median_ms']:.3f}ms speedup={result['speedup']:.3f}x",
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
