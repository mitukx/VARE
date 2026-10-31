"""Build the Train-pool task list (identifiers, hashes, base G=4 estimates; no output text).

Inputs: the frozen probe protocol, the group-informativeness v1 protocol (existing
first estimates) and its samples, the v1 Train manifest, and this directory's
tasks.jsonl. Writes train_pool_v1.json. Exploratory / diagnostic.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def classify(k: int, n: int) -> str:
    return "all_zero" if k == 0 else "all_one" if k == n else "mixed"


def main() -> None:
    probe_p = ROOT / "protocols/grpo_train_pool_probe_v1.json"
    prior_p = ROOT / "protocols/grpo_group_informativeness_v1.json"
    manifest_p = ROOT / "configs/learning_pipeline_v1/train.manifest.jsonl"
    study_samples = ROOT / "results/grpo-group-informativeness-v1/samples.jsonl"
    tasks_p = HERE / "tasks.jsonl"
    prior = json.loads(prior_p.read_text())["prior"]["per_task"]
    manifest = {t["instance_id"]: t for t in map(json.loads, manifest_p.read_text().splitlines()) if t}
    extra: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in map(json.loads, study_samples.read_text().splitlines()):
        extra[r["instance_id"]][0] += int(float(r["reward"]) > 0.5)
        extra[r["instance_id"]][1] += 1

    tasks = []
    for iid, e in sorted(prior.items()):
        m = manifest[iid]
        k2, n2 = e["k"] + extra[iid][0], e["n"] + extra[iid][1]
        tasks.append({"instance_id": iid, "task_group": m["upstream_repo"], "split": "train",
                      "origin": "v1_manifest", "estimate_source": "+".join(e["sources"]),
                      "patch_sha256": m["patch_sha256"], "problem_statement_sha256": m["problem_statement_sha256"],
                      "k": e["k"], "n": e["n"], "class": classify(e["k"], e["n"]),
                      "pooled_k": k2, "pooled_n": n2, "pooled_mixed_0.1_0.9": 0.1 <= k2 / n2 <= 0.9})
    gated: dict[str, int] = defaultdict(int)
    for r in map(json.loads, tasks_p.read_text().splitlines()):
        if r["gate"]:
            gated[r["gate"]] += 1
            continue
        tasks.append({"instance_id": r["instance_id"], "task_group": r["task_group"], "split": "train",
                      "origin": "pool_probe_v1", "estimate_source": "pool-probe-v1",
                      "patch_sha256": r["patch_sha256"], "problem_statement_sha256": r["problem_statement_sha256"],
                      "k": r["k"], "n": r["n"], "class": r["cls"], "pooled_k": r["k"], "pooled_n": r["n"],
                      "pooled_mixed_0.1_0.9": 0.1 <= r["k"] / r["n"] <= 0.9})
    by = lambda rows: {c: sum(t["class"] == c for t in rows) for c in ("mixed", "all_zero", "all_one")}  # noqa: E731
    out = {
        "schema": "vare.grpo_train_pool.v1",
        "label": "exploratory / diagnostic; Train tasks only; base policy G=4 T=0.7 top_p 0.95 estimates (approximate-ratio sampling, diagnostic only); rewards provisional (v1 native grader, pending trusted-grader requalification); no output text",
        "classification": "k of G=4 trusted passes: 0 all_zero, 4 all_one, 1-3 mixed",
        "inputs": {p.relative_to(ROOT).as_posix(): sha(p) for p in (probe_p, prior_p, manifest_p, study_samples, tasks_p)},
        "counts": {"all": by(tasks), "v1_manifest": by([t for t in tasks if t["origin"] == "v1_manifest"]),
                   "pool_probe_v1": by([t for t in tasks if t["origin"] == "pool_probe_v1"]),
                   "pooled_mixed_0.1_0.9": sum(t["pooled_mixed_0.1_0.9"] for t in tasks)},
        "probe_gated": dict(sorted(gated.items())),
        "mixed_instance_ids": sorted(t["instance_id"] for t in tasks if t["class"] == "mixed"),
        "tasks": tasks,
    }
    (HERE / "train_pool_v1.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("counts", "probe_gated")}, indent=1))


if __name__ == "__main__":
    main()
