"""Freeze the memory-bounded Train-only GRPO smoke v3 protocol (new identity).

smoke-002 (protocol v2) stopped at its swap rule during the step-0 update,
before any update completed. v3 keeps v2's 8 Train tasks, G, seed, sampling and
optimizer unchanged so the before/after systems comparison is matched, and only
adds memory bounds: an MLX buffer-cache limit, ``mx.clear_cache()`` after every
rollout and microbatch and between phases, an in-process swap stop rule, and a
start gate that refuses to load the model when swap is already high. Gradient
checkpointing was already on in v2 and stays on. Reads no held-out payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = "protocols/grpo_vertical_slice_smoke_v2.json"
V2_SHA256 = "609010224b46c7ed2cd8e1c6c736624d891dbc101736dda859a8cffe98615531"
SCRIPTS = ("scripts/grpo_vertical_slice.py", "src/vare/learning_pipeline/grpo_mlx.py",
           "src/vare/learning_pipeline/scoring.py", "scripts/grpo_smoke_v3_freeze.py")
MEMORY = {"mlx_cache_limit_gib": 1.0, "clear_cache": True, "max_swap_fraction": 0.97,
          "start_max_swap_fraction": 0.85, "grad_checkpointing": True, "max_tokens_unchanged": 512}


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def build() -> dict:
    if sha(V2) != V2_SHA256:
        raise SystemExit(f"{V2} changed since smoke-002 was frozen")
    v2 = json.loads((ROOT / V2).read_text())
    args = dict(v2["args"])
    out = "protocols/grpo_vertical_slice_smoke_v3.json"
    cmd = (f"python scripts/grpo_vertical_slice.py --run-dir ~/grpo-runs/smoke-003 --cache <cache> --task-list {out} "
           f"--group-size {args['group_size']} --steps {args['steps']} --groups-per-step {args['groups_per_step']} "
           f"--lr {args['lr']} --clip {args['clip']} --temperature {args['temperature']} --top-p {args['top_p']} "
           f"--seed {args['seed']} --mlx-cache-limit-gib {MEMORY['mlx_cache_limit_gib']} --clear-cache "
           f"--max-swap-fraction {MEMORY['max_swap_fraction']} "
           f"--start-max-swap-fraction {MEMORY['start_max_swap_fraction']}")
    return {
        "schema": "vare.grpo_vertical_slice.smoke_protocol/v3",
        "experiment_id": "grpo-vertical-slice-smoke-003",
        "label": "exploratory / diagnostic; Train tasks only; no held-out data; no capability claim",
        "supersedes_nothing": "smoke-001 and the partial smoke-002 are kept unchanged; v2 is consumed and not rerun",
        "question": "With bounded MLX memory, does the exact-ratio (top_p = 1.0) GRPO path complete end to end on "
                    "Qwen3-4B, and how do peak MLX memory, peak RSS, swap growth, tokens/s and s per update "
                    "step compare with smoke-002 on the same tasks and G?",
        "derived_from": {"protocol": V2, "sha256": V2_SHA256,
                         "unchanged": "tasks, group size, steps, groups per step, lr, clip, temperature, top_p, "
                                      "seed and max_tokens (matched comparison; seed reuse is deliberate)"},
        "tasks": v2["tasks"], "task_details": v2["task_details"], "train_repos": v2["train_repos"],
        "heldout_repos_excluded": v2["heldout_repos_excluded"],
        "sampling": v2["sampling"], "args": args, "memory_bounds": MEMORY, "command": cmd,
        "verifier": v2["verifier"],
        "verify": v2["verify"] + ["peak RSS", "swap growth", "generation tokens/s", "s per update step",
                                  "before/after table vs smoke-002"],
        "start_condition": "measure available memory and swap at start; do not start if swap used/total >= 0.85 "
                           "(report instead); never stop other processes",
        "stop_rules": {"swap_used_fraction_max": 0.97, "memory_free_pct_min": 15, "wall_clock_s": 5100,
                       "free_disk_gib_min": 8,
                       "action": "in-process swap rule stops before the next rollout or microbatch (no partial "
                                 "update); operator stops only this run's process; partial results reported"},
        "source_sha256": {rel: sha(rel) for rel in SCRIPTS if rel != "scripts/grpo_smoke_v3_freeze.py"},
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="protocols/grpo_vertical_slice_smoke_v3.json")
    a = p.parse_args(argv)
    out = ROOT / a.out
    out.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n")
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
