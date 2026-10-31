"""Exploratory / diagnostic: grow the Train pool of *mixed* tasks (base-policy
pass rate strictly between 0 and 1 under G = 4) before a GRPO confirmation study.

Candidates are Train-only: (A) admitted Train tasks of the v1 curriculum that
have no base estimate yet, then (B) further SWE-smith-py rows from the same five
Train repositories that are in neither v1 manifest. Every candidate passes the
v1 curriculum's static filters, patch / (file, issue) de-duplication against
both v1 manifests (Train and held-out), and an instance-id scan of the VARE
tree; the Train repositories are re-checked against the VARE exclusion scan and
the held-out repository list. Held-out tasks are never loaded.

``freeze`` writes the protocol (ordered candidate list, gates, classification
and stop rules) before any inference; ``run`` verifies its hash, then for each
candidate in order: file check, calibration (baseline fails / reference passes),
prompt-length gate, then G = 4 base rollouts at T = 0.7.

  python scripts/grpo_train_pool_probe.py freeze --cache ~/.cache/vare-learning-pipeline-v1
  python scripts/grpo_train_pool_probe.py run --run-dir ~/grpo-runs/pool-probe-v1 \
      --cache ~/.cache/vare-learning-pipeline-v1 --expect-sha <sha256>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

PROTOCOL = ROOT / "protocols/grpo_train_pool_probe_v1.json"
BASE_PROTOCOL = ROOT / "protocols/coding_policy_learning_mlx_4b_v1.json"
CURRICULUM = ROOT / "configs/learning_pipeline_v1/curriculum.json"
REPOS_LOCK = ROOT / "configs/learning_pipeline_v1/repos.lock.json"
TRAIN_MANIFEST = ROOT / "configs/learning_pipeline_v1/train.manifest.jsonl"
HELDOUT_MANIFEST = ROOT / "configs/learning_pipeline_v1/heldout.manifest.jsonl"
ADMITTED = ROOT / "results/learning-pipeline-v1/run-001/calibration/admitted.json"
PRIOR_PROTOCOL = ROOT / "protocols/grpo_group_informativeness_v1.json"
SCHEMA = "vare.grpo_train_pool_probe.protocol/v1"
G = 4
# Files allowed to mention a tier-A instance id (the v1 manifest and the run-001
# admission list). Any other mention (e.g. run-001 SFT provenance, i.e. the task
# was used in run-001) excludes the candidate.
TIER_A_ALLOWED = ("configs/learning_pipeline_v1/train.manifest.jsonl",
                  "results/learning-pipeline-v1/run-001/calibration/admitted.json")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(k: int, n: int) -> str:
    """Frozen rule: with G = 4, mixed iff 1 <= k <= 3, i.e. p_hat in
    {0.25, 0.5, 0.75}, inside [0.1, 0.9]."""
    if k == 0:
        return "all_zero"
    if k == n:
        return "all_one"
    return "mixed"


def swap_fraction() -> float | None:
    from grpo_group_informativeness_study import swap_fraction as sf
    return sf()


def free_disk_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 2**30


def tracked_text() -> dict[str, str]:
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    out = {}
    for rel in files:
        p = ROOT / rel
        if rel == PROTOCOL.relative_to(ROOT).as_posix() or not p.is_file() or p.stat().st_size > 8_000_000:
            continue
        if p.suffix not in {".md", ".json", ".jsonl", ".py", ".txt", ".yml", ".yaml", ".toml", ".csv", ".log"}:
            continue
        out[rel] = p.read_text(encoding="utf-8", errors="ignore")
    return out


# ------------------------------------------------------------------ freeze
def freeze(a) -> None:
    from vare.learning_pipeline import curriculum, exclusions, sources

    cfg = json.loads(CURRICULUM.read_text())
    lock = json.loads(REPOS_LOCK.read_text())
    train_repos, heldout_repos = list(cfg["train_repositories"]), list(cfg["heldout_repositories"])
    assert not set(train_repos) & set(heldout_repos)
    scan = exclusions.scan_vare_tree(ROOT, skip=cfg.get("scan_skip", ()))
    assert not exclusions.check_required(scan)
    excluded = exclusions.excluded_repositories(scan)
    repo_check = {r: exclusions.is_excluded(r, excluded) for r in train_repos}
    if any(repo_check.values()):
        raise SystemExit(f"Train repository excluded by the VARE scan: {repo_check}")

    train_m = curriculum.load_manifest(TRAIN_MANIFEST)
    heldout_m = curriculum.load_manifest(HELDOUT_MANIFEST)
    admitted = set(json.loads(ADMITTED.read_text())["train"])
    prior = json.loads(PRIOR_PROTOCOL.read_text())["prior"]["per_task"]
    used = set(prior)  # run-001 rollouts + smoke-001 survey: already have a base G=4 estimate
    manifest_ids = {t["instance_id"] for t in train_m} | {t["instance_id"] for t in heldout_m}
    mirrors = {lock[r]["mirror"]: r for r in train_repos}

    cache = Path(a.cache).expanduser()
    shards = curriculum.fetch_shards(sources.SWE_SMITH_PY, cache)
    rows = curriculum.load_rows(shards, set(mirrors))
    heldout_rows = {r["instance_id"]: r for r in curriculum.load_rows(shards, {t["mirror"] for t in heldout_m})}
    train_rows = {r["instance_id"]: r for r in rows}
    seen_patch, seen_key = set(), set()
    for t in train_m + heldout_m:
        r = train_rows.get(t["instance_id"]) or heldout_rows.get(t["instance_id"])
        seen_patch.add(exclusions.normalized_patch_sha256(r["patch"]))
        seen_key.add((r["repo"], t["target_file"], curriculum.sha256_text(r["problem_statement"].strip())))

    texts = tracked_text()
    rejected: dict[str, int] = {}

    def mentioned_outside(iid: str, allowed: tuple[str, ...]) -> list[str]:
        return [rel for rel, txt in texts.items() if iid in txt and rel not in allowed]

    def entry(t: dict[str, Any], tier: str) -> dict[str, Any]:
        keep = ("instance_id", "upstream_repo", "upstream_license", "source", "source_revision", "dataset_license",
                "mirror", "mirror_commit", "target_file", "changed_lines", "patch_sha256", "problem_statement_sha256",
                "fail_to_pass", "pass_to_pass_count", "pass_to_pass_sha256")
        return {**{k: t[k] for k in keep}, "split": "train", "tier": tier,
                "order_key": curriculum.order_key(f"pool-probe-v1:{a.seed}", t["instance_id"])}

    tier_a = []
    for t in train_m:
        iid = t["instance_id"]
        if iid in used or iid not in admitted:
            continue
        hits = mentioned_outside(iid, TIER_A_ALLOWED)
        if hits:
            why = "tier_a_used_in_run001_sft" if any("run-001" in h for h in hits) else "tier_a_mentioned_elsewhere"
            rejected[why] = rejected.get(why, 0) + 1
            continue
        tier_a.append(entry(t, "A_admitted_unprobed"))
    tier_a.sort(key=lambda e: e["order_key"])

    per_repo: dict[str, list[dict[str, Any]]] = {r: [] for r in train_repos}
    for row in sorted(rows, key=lambda r: curriculum.order_key(f"pool-probe-v1:{a.seed}", r["instance_id"])):
        iid = row["instance_id"]
        if iid in manifest_ids:
            continue
        reason, info = curriculum.prefilter(row, cfg)
        if reason is None:
            ph = exclusions.normalized_patch_sha256(row["patch"])
            key = (row["repo"], info["target_file"], curriculum.sha256_text(row["problem_statement"].strip()))
            if ph in seen_patch or key in seen_key:
                reason = "duplicate"
            elif mentioned_outside(iid, ()):
                reason = "mentioned_in_vare_tree"
            else:
                seen_patch.add(ph)
                seen_key.add(key)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        up = mirrors[row["repo"]]
        per_repo[up].append(entry({
            "instance_id": iid, "upstream_repo": up, "upstream_license": sources.REPO_LICENSES[up],
            "source": sources.SWE_SMITH_PY.source_id, "source_revision": sources.SWE_SMITH_PY.revision,
            "dataset_license": sources.SWE_SMITH_PY.dataset_license, "mirror": row["repo"],
            "mirror_commit": lock[up]["commit"], "target_file": info["target_file"],
            "changed_lines": info["changed_lines"], "patch_sha256": curriculum.sha256_text(row["patch"]),
            "problem_statement_sha256": curriculum.sha256_text(row["problem_statement"]),
            "fail_to_pass": list(row["FAIL_TO_PASS"]),
            "pass_to_pass_count": len(row["PASS_TO_PASS"] or []),
            "pass_to_pass_sha256": curriculum.sha256_text(json.dumps(list(row["PASS_TO_PASS"] or []))),
        }, "B_new_train_repo_row"))
    available_b = {r: len(v) for r, v in per_repo.items()}
    tier_b: list[dict[str, Any]] = []
    while len(tier_b) < a.max_tier_b and any(per_repo.values()):
        for r in train_repos:
            if per_repo[r] and len(tier_b) < a.max_tier_b:
                tier_b.append(per_repo[r].pop(0))
    candidates = tier_a + tier_b
    ids = [c["instance_id"] for c in candidates]
    assert len(ids) == len(set(ids)) and not set(ids) & {t["instance_id"] for t in heldout_m} and not set(ids) & used
    assert all(c["upstream_repo"] in train_repos for c in candidates)

    existing = {"mixed": 0, "all_zero": 0, "all_one": 0}
    for e in prior.values():
        existing[classify(e["k"], e["n"])] += 1
    protocol = {
        "schema": SCHEMA,
        "label": "exploratory / diagnostic; Train tasks only; no held-out data; no capability claim",
        "question": "How many Train tasks have a mixed base-policy outcome (1-3 of G=4 trusted passes at T=0.7)?",
        "goal": {"target_mixed_tasks_in_train_pool": 24,
                 "counting": "existing base G=4 first estimates (run-001 rollouts, smoke-001 survey) plus this probe"},
        "existing_pool": {"source": {"path": str(PRIOR_PROTOCOL.relative_to(ROOT)), "sha256": sha256_file(PRIOR_PROTOCOL)},
                          "tasks": len(prior), "by_class": existing},
        "base_protocol": {"path": str(BASE_PROTOCOL.relative_to(ROOT)), "sha256": sha256_file(BASE_PROTOCOL)},
        "policy": "base model, no adapter",
        "decoding": {"group_size": G, "temperature": 0.7, "top_p": 0.95, "max_tokens": "base protocol budgets.inference.max_tokens"},
        "verifier": "trusted_full_v1 via scoring.score_candidate; reward = 1 iff trusted pass",
        "classification": {"rule": "k trusted passes of G=4: k=0 all_zero, k=4 all_one, 1<=k<=3 mixed (p_hat in [0.25, 0.75], within [0.1, 0.9])",
                           "secondary": "pooled with group-informativeness v1 base samples where available, mixed iff 0.1 <= p_hat <= 0.9; reported, not used for the goal"},
        "seed": a.seed,
        "rollout_seed": "int(sha256(f'pool-probe-v1:{instance_id}:{k}')[:8], 16)",
        "candidate_rules": {
            "tier_A": "admitted v1 Train tasks with no base estimate yet, ordered by order_key",
            "tier_B": "SWE-smith-py rows from the five v1 Train repositories, not in either v1 manifest; v1 static filters; "
                      "dedup by normalized patch hash and (repo, file, issue-text hash) against both v1 manifests and earlier candidates; "
                      "instance id not mentioned anywhere in the tracked VARE tree; round-robin over repositories in order_key order",
            "order_key": f"sha256(f'pool-probe-v1:{a.seed}:{{instance_id}}')",
            "run_time_gates": "file check (bug applies, file <= max_file_chars, reference blocks restore the file); calibration "
                              "(baseline fails F2P and passes P2P; reference passes); prompt <= base max_prompt_tokens. "
                              "Gated-out candidates get no rollouts and are reported by reason.",
        },
        "repository_checks": {"train_repositories": train_repos, "heldout_repositories": heldout_repos,
                              "train_heldout_disjoint": True, "vare_exclusion_scan": repo_check,
                              "scan_identifiers": len(scan["identifiers"])},
        "candidate_counts": {"tier_A": len(tier_a), "tier_B": len(tier_b), "tier_B_available": available_b,
                             "rejected": dict(sorted(rejected.items()))},
        "candidates": candidates,
        "stop_rules": {"wall_clock_s": a.wall_s, "swap_used_fraction_max": 0.97, "free_disk_gib_min": 8.0,
                       "memory_free_pct_min": 15, "target_reached_total_mixed": a.stop_total_mixed,
                       "action": "checked before each candidate; stop and record the reason"},
        "analysis": "descriptive only; one G=4 group per task is a noisy estimate (mixed tasks replicated 15/21 in group-informativeness v1)",
    }
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(PROTOCOL, sha256_file(PROTOCOL), {"tier_A": len(tier_a), "tier_B": len(tier_b), "available_B": available_b,
                                            "rejected": rejected, "existing": existing})


# --------------------------------------------------------------------- run
def run(a) -> None:
    import mlx.core as mx
    import grpo_vertical_slice as gvs
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    from vare.learning_pipeline import curriculum, scoring
    from vare.learning_pipeline.prompting import PROMPT_VERSION, build_messages
    from vare.learning_pipeline.repo_env import env_for
    from vare.learning_pipeline.resources import GiB, memory_snapshot

    proto_sha = sha256_file(PROTOCOL)
    if a.expect_sha and proto_sha != a.expect_sha:
        raise SystemExit("protocol hash mismatch")
    proto = json.loads(PROTOCOL.read_text())
    base = json.loads(BASE_PROTOCOL.read_text())
    if sha256_file(BASE_PROTOCOL) != proto["base_protocol"]["sha256"]:
        raise SystemExit("base protocol changed since freeze")
    cfg = json.loads(CURRICULUM.read_text())
    stops = proto["stop_rules"]
    run_dir = Path(a.run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    events, samples_log, tasks_log = run_dir / "events.jsonl", run_dir / "samples.jsonl", run_dir / "tasks.jsonl"
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    gvs.jdump(run_dir / "run_meta.json", {"schema": "vare.grpo_train_pool_probe.run/v1", "protocol_sha256": proto_sha,
                                          "git_head": head, "label": proto["label"], "heldout_manifest_loaded": False,
                                          "mlx": mx.__version__, "model": base["model"]})

    def note(event, **info):
        row = {"event": event, **info, **gvs.telemetry(), "swap_fraction": swap_fraction(),
               "free_disk_gib": round(free_disk_gib(Path.home()), 2)}
        gvs.append(events, row)
        print(json.dumps(row, default=str)[:400], flush=True)
        return row

    cache = Path(a.cache).expanduser()
    cands = proto["candidates"]
    assert all(c["split"] == "train" for c in cands)
    payload = {t["instance_id"]: t for t in curriculum.attach_payloads([dict(c) for c in cands], cache)}
    note("start", memory=memory_snapshot(), protocol_sha256=proto_sha, candidates=len(cands))
    m = base["model"]
    path = snapshot_download(m["id"], revision=m["revision"], allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt",
                                                                             "tokenizer.model", "*.tiktoken", "*.jinja"])
    mx.reset_peak_memory()
    model, tok = load(path)
    note("loaded")
    envs: dict[str, Any] = {}

    def env(t):
        e = envs.get(t["mirror"])
        if e is None:
            e = env_for(t, cache)
            if not e.python.exists() or not (e.path / ".git").exists():
                e.materialize()
            envs[t["mirror"]] = e
        return e

    existing_mixed = proto["existing_pool"]["by_class"]["mixed"]
    counts = {"mixed": 0, "all_zero": 0, "all_one": 0}
    gated: dict[str, int] = {}
    t_all = time.time()
    stopped = None
    probed = 0
    for c in cands:
        swap = swap_fraction()
        tel = gvs.telemetry()
        free_pct = int(str(tel.get("memory_free_pct") or "100").rstrip("%"))
        if time.time() - t_all > stops["wall_clock_s"]:
            stopped = "wall_clock"
        elif swap is not None and swap >= stops["swap_used_fraction_max"]:
            stopped = f"swap_used_fraction={swap:.3f}"
        elif free_disk_gib(Path.home()) < stops["free_disk_gib_min"]:
            stopped = f"free_disk_gib={free_disk_gib(Path.home()):.2f}"
        elif free_pct < stops["memory_free_pct_min"]:
            stopped = f"memory_free_pct={free_pct}"
        elif existing_mixed + counts["mixed"] >= stops["target_reached_total_mixed"]:
            stopped = "target_reached"
        if stopped:
            note("stopped", reason=stopped)
            break
        iid = c["instance_id"]
        t = payload[iid]
        e = env(t)
        rec: dict[str, Any] = {"instance_id": iid, "tier": c["tier"], "task_group": c["upstream_repo"], "split": "train",
                               "patch_sha256": c["patch_sha256"], "problem_statement_sha256": c["problem_statement_sha256"]}
        t0 = time.time()
        why = curriculum._file_check(e, t, cfg)
        if why is None:
            cal = scoring.calibrate(e, t, timeout=base["budgets"]["grading"]["timeout_s"])
            rec["calibration"] = {k: cal[k] for k in ("qualified", "baseline_fails_f2p", "baseline_passes_p2p", "reference_passes")}
            if not cal["baseline_fails_f2p"]:
                why = "baseline_does_not_fail"
            elif not cal["baseline_passes_p2p"]:
                why = "baseline_p2p_not_green"
            elif not cal["reference_passes"]:
                why = "reference_fails"
        buggy = None
        if why is None:
            buggy = e.buggy_file_text(t["_patch"], t["target_file"])
            msgs = build_messages(t, buggy)
            n_prompt = len(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True))
            rec["prompt_tokens"] = n_prompt
            if n_prompt > base["budgets"]["inference"]["max_prompt_tokens"]:
                why = "prompt_too_long"
        rec["gate_seconds"] = round(time.time() - t0, 3)
        if why:
            e.reset()
            gated[why] = gated.get(why, 0) + 1
            rec.update(gate=why)
            gvs.append(tasks_log, rec)
            note("gated", instance_id=iid, reason=why)
            continue
        rewards, secs = [], []
        for k in range(G):
            seed = int(hashlib.sha256(f"pool-probe-v1:{iid}:{k}".encode()).hexdigest()[:8], 16)
            t1 = time.time()
            g = gvs.generate(model, tok, msgs, seed=seed, temperature=proto["decoding"]["temperature"],
                             top_p=proto["decoding"]["top_p"], max_tokens=base["budgets"]["inference"]["max_tokens"])
            sc = scoring.score_candidate(e, t, buggy, g["text"], timeout=base["budgets"]["grading"]["timeout_s"])
            dt = time.time() - t1
            rewards.append(float(sc["reward"]))
            secs.append(dt)
            gvs.append(samples_log, {"instance_id": iid, "tier": c["tier"], "split": "train", "task_group": c["upstream_repo"],
                                     "sample_index": k, "seed": seed, "prompt_version": PROMPT_VERSION,
                                     "prompt_tokens": len(g["prompt_ids"]), "completion_tokens": len(g["completion_ids"]),
                                     "finish": g["finish"], "output_sha256": sc["output_sha256"], "reward": sc["reward"],
                                     "outcome": sc["outcome"], "gen_seconds": g["seconds"], "rollout_seconds": round(dt, 3)})
        e.reset()
        n_pass = sum(1 for r in rewards if r > 0.5)
        cls = classify(n_pass, G)
        counts[cls] += 1
        probed += 1
        rec.update(gate=None, rewards=rewards, k=n_pass, n=G, p_hat=n_pass / G, cls=cls,
                   sec_per_rollout=round(sum(secs) / G, 3))
        gvs.append(tasks_log, rec)
        note("probed", instance_id=iid, tier=c["tier"], rewards=rewards, cls=cls, sec_per_rollout=rec["sec_per_rollout"],
             mlx_peak=round(mx.get_peak_memory() / GiB, 3), total_mixed=existing_mixed + counts["mixed"])
    summary = {"schema": "vare.grpo_train_pool_probe.summary/v1", "label": proto["label"], "protocol_sha256": proto_sha,
               "stopped": stopped, "wall_clock_s": round(time.time() - t_all, 1), "candidates_listed": len(cands),
               "candidates_examined": probed + sum(gated.values()), "probed": probed, "gated": gated,
               "probe_by_class": counts, "existing_by_class": proto["existing_pool"]["by_class"],
               "total_mixed": existing_mixed + counts["mixed"], "target": proto["goal"]["target_mixed_tasks_in_train_pool"],
               "mlx_peak_gib": round(mx.get_peak_memory() / GiB, 3)}
    gvs.jdump(run_dir / "summary.json", summary)
    note("done", **{k: summary[k] for k in ("wall_clock_s", "probed", "total_mixed", "stopped")})


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--cache", required=True)
    f.add_argument("--seed", type=int, default=20261102)
    f.add_argument("--max-tier-b", type=int, default=140)
    f.add_argument("--wall-s", type=int, default=5100)
    f.add_argument("--stop-total-mixed", type=int, default=34)
    r = sub.add_parser("run")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--cache", required=True)
    r.add_argument("--expect-sha", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse()
    freeze(args) if args.cmd == "freeze" else run(args)
