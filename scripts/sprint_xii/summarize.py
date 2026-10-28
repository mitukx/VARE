"""Recompute descriptive results from retained receipts, not model inference."""

import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(root):
    rows = json.loads((root / "summary.json").read_text())
    result = {}
    for arm in "ABCDE":
        cases = [x for x in rows if x["arm"] == arm]
        latency = []
        io = []
        for case in cases:
            path = root / f"seed-{case['seed']}" / arm
            latency.extend(
                json.loads((path / f"done-{i}.json").read_text())[
                    "queue_inclusive_seconds"
                ]
                for i in range(5)
            )
            io.extend(
                json.loads(p.read_text())["io_seconds"]
                for p in path.glob("checkpoint-*.pt.gz.json")
            )
        result[arm] = {
            "effective_commits": sum(x["effective_commits"] for x in cases),
            "contaminated_updates": sum(x["contaminated_updates"] for x in cases),
            "v1_reward_mean": statistics.mean(x["v1_reward"] for x in cases),
            "v1_reward_min": min(x["v1_reward"] for x in cases),
            "v1_reward_max": max(x["v1_reward"] for x in cases),
            "trajectories_s": statistics.mean(
                x["trajectories_per_second"] for x in cases
            ),
            "useful_trajectories_s": statistics.mean(
                x["useful_trajectories_per_second"] for x in cases
            ),
            "queue_p50_s": statistics.median(latency),
            "queue_p95_s": sorted(latency)[math.ceil(0.95 * len(latency)) - 1],
            "peak_process_rss": max(x["peak_child_rss"] for x in cases),
            "swap_growth": max(x["swap_growth"] for x in cases),
            "checkpoint_io_mean_s": statistics.mean(io),
            "total_wall_s": sum(x["wall_seconds"] for x in cases),
        }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    a = p.parse_args()
    print(json.dumps(summarize(a.run), indent=2))
