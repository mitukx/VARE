#!/usr/bin/env python3
"""Write a synthetic rollouts/graded.jsonl matching vare.learning_pipeline.rollout/v1.

Synthetic only (no model, no repository): used to exercise pilot_analysis.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SCHEMA = "vare.learning_pipeline.rollout/v1"


def make(n_tasks=78, G=4, n_repos=10, seed=0, p_mixed=0.35, p_fp_given_wrong=0.3, p_unapplied=0.25):
    rng = random.Random(seed)
    ckpt = "base:mlx-community/Qwen3-4B-Instruct-2507-4bit"
    rows = []
    for k in range(n_tasks):
        repo = f"example-org/repo{k % n_repos}"
        iid = f"example-org__repo{k % n_repos}.deadbeef.func_basic__{k:04d}"
        f2p = [f"tests/test_m.py::test_f{j}" for j in range(rng.randint(1, 4))]
        p2p = [f"tests/test_m.py::test_p{j}" for j in range(rng.randint(3, 12))]
        u = rng.random()
        p_solve = rng.uniform(0.2, 0.8) if u < p_mixed else (0.0 if u < 0.9 else 1.0)
        for s in range(G):
            text = f"candidate {k}-{s}-{rng.random()}"
            sha = hashlib.sha256(text.encode()).hexdigest()
            base = {"schema": SCHEMA, "stage": "rollout", "checkpoint_id": ckpt, "checkpoint_step": 0,
                    "group_id": f"{ckpt}|{iid}", "task_group": repo, "split": "train", "arm": "rft_self",
                    "policy_id": "synthetic", "instance_id": iid, "sample_index": s, "seed": rng.randrange(2**31),
                    "temperature": 0.8, "top_p": 0.95, "max_tokens": 512, "output_text": text,
                    "output_sha256": sha, "finish": "stop", "prompt_tokens": 2000, "completion_tokens": 300,
                    "generation_seconds": 30.0, "synthetic": True}
            if rng.random() < p_unapplied and p_solve < 1.0:
                base.update(apply="no_blocks", outcome="malformed_or_unapplied_edit", reward=0,
                            verifiers={v: {"passed": False, "reason": "no_blocks", "seconds": 0.0,
                                           "executed": False, "tests": {}} for v in ("cheap_f2p_v1", "trusted_full_v1")})
                rows.append(base)
                continue
            solved = rng.random() < p_solve
            f2p_ok = solved or rng.random() < p_fp_given_wrong
            fs = {t: "PASSED" if f2p_ok else rng.choice(["FAILED", "ERROR"]) for t in f2p}
            ps = {t: "PASSED" for t in p2p}
            if f2p_ok and not solved:
                ps[p2p[0]] = "FAILED"  # regression only P2P catches -> cheap false positive
            ct, tt = 4.0 + rng.random(), 9.0 + 3 * rng.random()
            cheap = {"passed": f2p_ok, "seconds": ct, "pytest_seconds": ct - 1, "tests": dict(fs)}
            trusted_ok = all(v == "PASSED" for v in {**fs, **ps}.values())
            trusted = {"passed": trusted_ok, "seconds": tt, "pytest_seconds": tt - 1, "tests": {**fs, **ps}}
            base.update(apply="ok", final_diff="--- a\n+++ b\n", reward=int(trusted_ok),
                        outcome="pass" if trusted_ok else "edit_applied_but_wrong",
                        cheap_false_positive=bool(f2p_ok and not trusted_ok),
                        verifiers={"cheap_f2p_v1": cheap, "trusted_full_v1": trusted})
            rows.append(base)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=78)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in make(n_tasks=a.tasks, seed=a.seed):
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
