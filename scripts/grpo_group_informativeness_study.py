"""Exploratory / diagnostic: how much GRPO group signal does a fixed rollout
budget buy on Train tasks? (plan section 6.3)

Three conditions share one rollout budget and the same base policy
(Qwen3-4B 4-bit, no adapter), prompt, decoding and verifier:

  uniform_g4  uniformly drawn Train tasks, G = 4
  uniform_g8  the first half of the same draw, G = 8
  prior_g4    G = 4 on the Train tasks ranked highest by a transparent prior:
              posterior-predictive probability that a G-sample group is mixed,
              from earlier base-policy Train rollouts (k passes of n) under a
              uniform Beta(1, 1) prior; ties broken by a seeded hash.

``freeze`` writes the protocol (inputs, selections, stop rules) before any
rollout; ``run`` executes it and verifies the protocol hash first. Held-out
tasks are never loaded: the pool is restricted to admitted Train tasks.

  python scripts/grpo_group_informativeness_study.py freeze \
      --run001-graded <run-001 rollouts/graded.jsonl>
  python scripts/grpo_group_informativeness_study.py run \
      --run-dir ~/grpo-runs/informativeness-v1 --cache ~/.cache/vare-learning-pipeline-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import exp, lgamma
import platform
import re
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

PROTOCOL = ROOT / "protocols/grpo_group_informativeness_v1.json"
BASE_PROTOCOL = ROOT / "protocols/coding_policy_learning_mlx_4b_v1.json"
ADMITTED = ROOT / "results/learning-pipeline-v1/run-001/calibration/admitted.json"
SMOKE = ROOT / "results/grpo-vertical-slice/smoke-001/samples.jsonl"
SCHEMA = "vare.grpo_group_informativeness.protocol/v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lbeta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def p_mixed(k: int, n: int, g: int) -> float:
    """Posterior-predictive P(group of g is neither all-pass nor all-fail)
    with p ~ Beta(1 + k, 1 + n - k)."""
    a, b = 1 + k, 1 + n - k
    all_one = exp(_lbeta(a + g, b) - _lbeta(a, b))
    all_zero = exp(_lbeta(a, b + g) - _lbeta(a, b))
    return 1.0 - all_one - all_zero


def swap_fraction() -> float | None:
    """macOS used/total swap (the total grows on demand); None elsewhere."""
    if platform.system() != "Darwin":
        return None
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"total = ([\d.]+)M\s+used = ([\d.]+)M", out)
    return float(m.group(2)) / float(m.group(1)) if m and float(m.group(1)) > 0 else None


def tie_key(seed: int, iid: str) -> str:
    return hashlib.sha256(f"{seed}:{iid}".encode()).hexdigest()


def wilson(x: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(c - h, 4), round(c + h, 4)]


# ------------------------------------------------------------------ freeze
def freeze(a) -> None:
    train = set(json.loads(ADMITTED.read_text())["train"])
    priors: dict[str, dict[str, Any]] = {}

    def add(rows, source):
        for r in rows:
            if r.get("split") != "train" or r["instance_id"] not in train:
                raise SystemExit(f"non-Train row in prior source {source}: {r['instance_id']}")
            e = priors.setdefault(r["instance_id"], {"k": 0, "n": 0, "sources": []})
            e["k"] += int(float(r["reward"]) > 0.5)
            e["n"] += 1
            if source not in e["sources"]:
                e["sources"].append(source)

    graded = Path(a.run001_graded).expanduser()
    add([json.loads(x) for x in graded.read_text().splitlines() if x.strip()], "run-001/rollouts")
    add([r for r in map(json.loads, SMOKE.read_text().splitlines()) if r["stage"] == "survey"], "smoke-001/survey")
    pool = sorted(priors)
    seed, budget = a.seed, a.budget
    g = 4
    n_a = budget // g
    uniform = random.Random(seed).sample(pool, n_a)
    ranked = sorted(pool, key=lambda i: (-round(p_mixed(priors[i]["k"], priors[i]["n"], g), 12), tie_key(seed, i)))
    for i in pool:
        e = priors[i]
        e["p_hat"] = round(e["k"] / e["n"], 4)
        e["p_mixed_g4"] = round(p_mixed(e["k"], e["n"], 4), 6)
    protocol = {
        "schema": SCHEMA,
        "label": "exploratory / diagnostic; Train tasks only; no held-out data; no capability claim",
        "question": "Mixed-group fraction and wasted generations per rollout budget under three Train task/group-size samplers",
        "base_protocol": {"path": str(BASE_PROTOCOL.relative_to(ROOT)), "sha256": sha256_file(BASE_PROTOCOL)},
        "policy": "base model, no adapter (identical to LoRA B = 0)",
        "decoding": {"temperature": 0.7, "top_p": 0.95, "max_tokens": "base protocol budgets.inference.max_tokens"},
        "verifier": "trusted_full_v1 via scoring.score_candidate, timeout from base protocol",
        "rollout_budget_per_condition": budget,
        "seed": seed,
        "prior": {
            "rule": "score = posterior-predictive P(mixed group of G=4) under Beta(1+k, 1+n-k); rank descending; ties by sha256(f'{seed}:{instance_id}')",
            "sources": {
                "run-001/rollouts": {"file": "run-001 rollouts/graded.jsonl (not committed; contains completion text)",
                                     "sha256": sha256_file(graded), "rows": sum(1 for x in graded.read_text().splitlines() if x.strip())},
                "smoke-001/survey": {"file": str(SMOKE.relative_to(ROOT)), "sha256": sha256_file(SMOKE)},
            },
            "per_task": priors,
        },
        "pool": {"rule": "admitted Train tasks with a prior estimate", "size": len(pool)},
        "conditions": {
            "uniform_g4": {"group_size": 4, "tasks": uniform},
            "uniform_g8": {"group_size": 8, "tasks": uniform[: budget // 8]},
            "prior_g4": {"group_size": 4, "tasks": ranked[:n_a]},
        },
        "schedule": "groups interleaved across conditions round-robin (uniform_g4, uniform_g8, prior_g4), each condition in its listed order",
        "rollout_seed": "int(sha256(f'study-v1:{condition}:{instance_id}:{k}')[:8], 16)",
        "metrics": ["mixed_group_fraction (Wilson 95% interval)", "all_zero / all_one groups",
                    "wasted_generations (in zero-variance groups) and informative_generation_fraction",
                    "sec_per_rollout (generation + grading)", "MLX peak, available memory, swap, memory_pressure, thermal"],
        "stop_rules": {"wall_clock_s": a.wall_s, "swap_used_fraction_max": 0.97, "memory_free_pct_min": 15,
                       "action": "stop before the next group; record the reason; report partial conditions as such"},
        "analysis": "descriptive only; small n; no significance claims. Regression to the mean is expected for prior_g4 because priors come from the same small samples used to rank.",
    }
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(PROTOCOL, sha256_file(PROTOCOL))


# --------------------------------------------------------------------- run
def run(a) -> None:
    import mlx.core as mx
    import grpo_vertical_slice as gvs
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    from vare.learning_pipeline import curriculum, scoring
    from vare.learning_pipeline.prompting import PROMPT_VERSION, build_messages
    from vare.learning_pipeline.resources import GiB, memory_snapshot

    proto = json.loads(PROTOCOL.read_text())
    proto_sha = sha256_file(PROTOCOL)
    if a.expect_sha and proto_sha != a.expect_sha:
        raise SystemExit("protocol hash mismatch")
    base = json.loads(BASE_PROTOCOL.read_text())
    if sha256_file(BASE_PROTOCOL) != proto["base_protocol"]["sha256"]:
        raise SystemExit("base protocol changed since freeze")
    if sha256_file(SMOKE) != proto["prior"]["sources"]["smoke-001/survey"]["sha256"]:
        raise SystemExit("prior source changed since freeze")
    run_dir = Path(a.run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    events, samples_log = run_dir / "events.jsonl", run_dir / "samples.jsonl"
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    gvs.jdump(run_dir / "run_meta.json", {"schema": "vare.grpo_group_informativeness.run/v1", "protocol_sha256": proto_sha,
                                          "git_head": head, "label": proto["label"], "heldout_manifest_loaded": False,
                                          "mlx": mx.__version__, "model": base["model"]})

    def note(event, **info):
        row = {"event": event, **info, **gvs.telemetry()}
        gvs.append(events, row)
        print(json.dumps(row, default=str)[:400], flush=True)
        return row

    train_ids = set(json.loads(ADMITTED.read_text())["train"])
    manifest = {t["instance_id"]: t for t in curriculum.load_manifest(ROOT / base["curriculum"]["train_manifest"])}
    conds = proto["conditions"]
    needed = sorted({i for c in conds.values() for i in c["tasks"]})
    assert all(i in train_ids and manifest[i]["split"] == "train" for i in needed)
    tasks = {t["instance_id"]: t for t in curriculum.attach_payloads([manifest[i] for i in needed], Path(a.cache).expanduser())}
    envs: dict[str, Any] = {}

    def env(t):
        e = envs.get(t["mirror"])
        if e is None:
            from vare.learning_pipeline.repo_env import env_for
            e = env_for(t, Path(a.cache).expanduser())
            if not e.python.exists() or not (e.path / ".git").exists():
                e.materialize()
            envs[t["mirror"]] = e
        return e

    for t in tasks.values():
        t["_buggy"] = env(t).buggy_file_text(t["_patch"], t["target_file"])
    note("start", memory=memory_snapshot(), protocol_sha256=proto_sha)
    m = base["model"]
    path = snapshot_download(m["id"], revision=m["revision"], allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt",
                                                                             "tokenizer.model", "*.tiktoken", "*.jinja"])
    mx.reset_peak_memory()
    model, tok = load(path)
    note("loaded")
    order = ["uniform_g4", "uniform_g8", "prior_g4"]
    queues = {c: list(conds[c]["tasks"]) for c in order}
    schedule = []
    while any(queues.values()):
        for c in order:
            if queues[c]:
                schedule.append((c, queues[c].pop(0)))
    t_all = time.time()
    stopped = None
    groups: list[dict[str, Any]] = []
    for c, iid in schedule:
        tel = gvs.telemetry()
        swap_frac = swap_fraction()
        free_pct = int(str(tel.get("memory_free_pct") or "100").rstrip("%"))
        if time.time() - t_all > proto["stop_rules"]["wall_clock_s"]:
            stopped = "wall_clock"
        elif swap_frac is not None and swap_frac > proto["stop_rules"]["swap_used_fraction_max"]:
            stopped = f"swap_used_fraction={swap_frac:.3f}"
        elif free_pct < proto["stop_rules"]["memory_free_pct_min"]:
            stopped = f"memory_free_pct={free_pct}"
        if stopped:
            note("stopped", reason=stopped)
            break
        t = tasks[iid]
        gsize = conds[c]["group_size"]
        msgs = build_messages(t, t["_buggy"])
        rewards, secs = [], []
        for k in range(gsize):
            seed = int(hashlib.sha256(f"study-v1:{c}:{iid}:{k}".encode()).hexdigest()[:8], 16)
            t0 = time.time()
            g = gvs.generate(model, tok, msgs, seed=seed, temperature=proto["decoding"]["temperature"],
                             top_p=proto["decoding"]["top_p"], max_tokens=base["budgets"]["inference"]["max_tokens"])
            rec = scoring.score_candidate(env(t), t, t["_buggy"], g["text"], timeout=base["budgets"]["grading"]["timeout_s"])
            dt = time.time() - t0
            rewards.append(float(rec["reward"]))
            secs.append(dt)
            gvs.append(samples_log, {"condition": c, "instance_id": iid, "split": t["split"], "task_group": t["upstream_repo"],
                                     "sample_index": k, "seed": seed, "prompt_version": PROMPT_VERSION,
                                     "prompt_tokens": len(g["prompt_ids"]), "completion_tokens": len(g["completion_ids"]),
                                     "finish": g["finish"], "output_sha256": rec["output_sha256"], "reward": rec["reward"],
                                     "outcome": rec["outcome"], "gen_seconds": g["seconds"], "rollout_seconds": round(dt, 3)})
        env(t).reset()
        row = {"condition": c, "instance_id": iid, "group_size": gsize, "rewards": rewards,
               "mixed": len(set(rewards)) > 1, "sec_per_rollout": round(sum(secs) / len(secs), 3)}
        groups.append(row)
        note("group", **row, mlx_peak=round(mx.get_peak_memory() / GiB, 3),
             swap_fraction=None if swap_frac is None else round(swap_frac, 4))
    summary = summarize(proto, groups, stopped, time.time() - t_all)
    gvs.jdump(run_dir / "summary.json", summary)
    note("done", wall_clock_s=summary["wall_clock_s"])


def summarize(proto, groups, stopped, wall) -> dict[str, Any]:
    out: dict[str, Any] = {"schema": "vare.grpo_group_informativeness.summary/v1", "label": proto["label"],
                           "stopped": stopped, "wall_clock_s": round(wall, 1), "conditions": {}}
    for c, spec in proto["conditions"].items():
        rows = [g for g in groups if g["condition"] == c]
        n = len(rows)
        mixed = sum(r["mixed"] for r in rows)
        zero = sum(set(r["rewards"]) == {0.0} for r in rows)
        one = sum(set(r["rewards"]) == {1.0} for r in rows)
        gens = sum(len(r["rewards"]) for r in rows)
        wasted = sum(len(r["rewards"]) for r in rows if not r["mixed"])
        secs = [r["sec_per_rollout"] * len(r["rewards"]) for r in rows]
        out["conditions"][c] = {
            "group_size": spec["group_size"], "groups_planned": len(spec["tasks"]), "groups_done": n,
            "complete": n == len(spec["tasks"]), "generations": gens, "mixed_groups": mixed,
            "mixed_group_fraction": round(mixed / n, 4) if n else None, "mixed_group_fraction_wilson95": wilson(mixed, n),
            "all_zero_groups": zero, "all_one_groups": one, "wasted_generations": wasted,
            "informative_generation_fraction": round(1 - wasted / gens, 4) if gens else None,
            "mean_reward": round(sum(sum(r["rewards"]) for r in rows) / gens, 4) if gens else None,
            "sec_per_rollout": round(sum(secs) / gens, 2) if gens else None,
        }
    return out


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--run001-graded", required=True)
    f.add_argument("--budget", type=int, default=96)
    f.add_argument("--seed", type=int, default=20261101)
    f.add_argument("--wall-s", type=int, default=5100)
    r = sub.add_parser("run")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--cache", required=True)
    r.add_argument("--expect-sha", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse()
    freeze(args) if args.cmd == "freeze" else run(args)
