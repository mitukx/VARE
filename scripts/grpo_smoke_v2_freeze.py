"""Freeze the Train-only GRPO smoke v2 protocol (new identity; directive C4).

Selects admitted v1 Train tasks whose pooled base-policy estimate (run-001 /
smoke-001 priors plus the group-informativeness v1 samples) is mixed with
pooled p-hat in [0.1, 0.9], orders them by a seeded hash, and keeps the first
``--tasks``. Sampling is untruncated (top_p = 1.0, exact-ratio objective).
Writes the protocol JSON; it never reads the held-out task payloads, only the
held-out manifest's identifiers and repositories for the disjointness check.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "protocols/coding_policy_learning_mlx_4b_v1.json"
PRIOR = "protocols/grpo_group_informativeness_v1.json"
STUDY_SAMPLES = "results/grpo-group-informativeness-v1/samples.jsonl"
ADMITTED = "results/learning-pipeline-v1/run-001/calibration/admitted.json"
SCRIPTS = ("scripts/grpo_vertical_slice.py", "src/vare/learning_pipeline/grpo_mlx.py",
           "src/vare/learning_pipeline/scoring.py", "scripts/grpo_smoke_v2_freeze.py")


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def rows(rel: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / rel).read_text().splitlines() if line.strip()]


def select(seed: int, n_tasks: int) -> dict:
    base = json.loads((ROOT / BASE).read_text())
    train_rel, heldout_rel = base["curriculum"]["train_manifest"], base["curriculum"]["heldout_manifest"]
    train = {t["instance_id"]: t for t in rows(train_rel)}
    heldout = rows(heldout_rel)
    heldout_ids = {t["instance_id"] for t in heldout}
    heldout_repos = {t["upstream_repo"] for t in heldout}
    admitted = set(json.loads((ROOT / ADMITTED).read_text())["train"])
    pooled: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for iid, est in json.loads((ROOT / PRIOR).read_text())["prior"]["per_task"].items():
        pooled[iid][0] += est["k"]
        pooled[iid][1] += est["n"]
    for r in rows(STUDY_SAMPLES):
        pooled[r["instance_id"]][0] += int(float(r["reward"]) > 0.5)
        pooled[r["instance_id"]][1] += 1
    eligible = []
    for iid, (k, n) in pooled.items():
        t = train.get(iid)
        if t is None or iid not in admitted or t["split"] != "train":
            continue
        if iid in heldout_ids or t["upstream_repo"] in heldout_repos:
            raise SystemExit(f"held-out overlap: {iid}")
        if 0 < k < n and 0.1 <= k / n <= 0.9:
            eligible.append(iid)
    key = {i: hashlib.sha256(f"grpo-smoke-v2:{seed}:{i}".encode()).hexdigest() for i in eligible}
    chosen = sorted(eligible, key=key.get)[:n_tasks]
    return {
        "eligible": len(eligible),
        "tasks": chosen,
        "task_details": [{"instance_id": i, "upstream_repo": train[i]["upstream_repo"], "pooled_k": pooled[i][0],
                          "pooled_n": pooled[i][1], "order_key": key[i]} for i in chosen],
        "train_repos": sorted({train[i]["upstream_repo"] for i in chosen}),
        "heldout_repos_excluded": sorted(heldout_repos),
        "inputs": {rel: sha(rel) for rel in (BASE, PRIOR, STUDY_SAMPLES, ADMITTED, train_rel)},
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="protocols/grpo_vertical_slice_smoke_v2.json")
    p.add_argument("--seed", type=int, default=20261104)
    p.add_argument("--tasks", type=int, default=8)
    a = p.parse_args(argv)
    sel = select(a.seed, a.tasks)
    proto = {
        "schema": "vare.grpo_vertical_slice.smoke_protocol/v2",
        "experiment_id": "grpo-vertical-slice-smoke-002",
        "label": "exploratory / diagnostic; Train tasks only; no held-out data; no capability claim",
        "supersedes_nothing": "smoke-001 (top_p 0.95, approximate-ratio evidence) is kept unchanged",
        "question": "Does the exact-ratio (top_p = 1.0) GRPO path run end to end on Qwen3-4B with real "
                    "policy-generated completions, rewards, updates, checkpoint and reload?",
        "seed": a.seed,
        "selection_rule": "admitted v1 Train tasks; pooled base estimate (prior + informativeness-v1 samples) "
                          "0 < k < n and 0.1 <= k/n <= 0.9; order by sha256('grpo-smoke-v2:{seed}:{iid}'); "
                          f"first {a.tasks}. Priors were measured at top_p 0.95 and are only a selection heuristic.",
        **sel,
        "sampling": {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
                     "max_tokens": "base protocol budgets.inference.max_tokens",
                     "ratio": "exact (untruncated softmax(logits / T); docs/grpo-exact-ratio-sampling-decision-v1.md)"},
        "args": {"group_size": 4, "steps": 3, "groups_per_step": 2, "lr": 1e-5, "clip": 0.2,
                 "temperature": 0.7, "top_p": 1.0, "seed": a.seed},
        "command": "python scripts/grpo_vertical_slice.py --run-dir ~/grpo-runs/smoke-002 --cache <cache> "
                   "--task-list protocols/grpo_vertical_slice_smoke_v2.json --group-size 4 --steps 3 "
                   "--groups-per-step 2 --lr 1e-5 --clip 0.2 --temperature 0.7 --top-p 1.0 "
                   f"--seed {a.seed}",
        "verifier": "v1 native grader via scoring.score_candidate; rewards provisional pending "
                    "trusted-grader requalification (directive D1)",
        "verify": ["policy-generated completions", "rewards per sample", "behavior checkpoint id per group",
                   "params changed after each update", "loss / approx_kl / max_abs_log_ratio per step",
                   "checkpoint digest equal after reload", "post-update generation and grading",
                   "MLX peak memory and swap"],
        "stop_rules": {"swap_used_fraction_max": 0.97, "memory_free_pct_min": 15, "wall_clock_s": 5100,
                       "free_disk_gib_min": 8,
                       "action": "operator stops only this run's process; partial results reported as partial"},
        "source_sha256": {rel: sha(rel) for rel in SCRIPTS if rel != "scripts/grpo_smoke_v2_freeze.py"},
    }
    out = ROOT / a.out
    out.write_text(json.dumps(proto, indent=1, sort_keys=True) + "\n")
    print(out.relative_to(ROOT), sel["eligible"], sel["tasks"])


if __name__ == "__main__":
    main()
