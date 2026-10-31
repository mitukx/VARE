"""GRPO vertical slice on Qwen3-4B (MLX 4-bit, LoRA) using Train tasks only.

Evidence level: real RL parameter updates on a handful of Train groups. This
script never loads the held-out manifest and makes no capability claim.

Phases
  survey  base policy (LoRA B = 0, identical to the base model) samples
          ``group_size`` completions on Train tasks not used by run-001's
          rollouts; every completion is graded with the v1 trusted verifier.
          Reports reward variance, mixed/all-zero/all-one group counts.
  update  step 0 trains on the survey groups (they are on-policy for step 0).
          Steps 1.. resample fresh on-policy groups from the current policy on
          the survey's informative tasks (a transparent Train-only difficulty
          rule), grade them and update again. Every step checkpoints LoRA
          params + optimizer state + RNG schedule.
  reload  a fresh model loads the last checkpoint, the parameter digest is
          checked, and post-update completions on the same Train tasks are
          generated and graded.

Example:
  python scripts/grpo_vertical_slice.py --run-dir ~/grpo-runs/smoke-001 \
      --cache ~/.cache/vare-learning-pipeline-v1
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vare.learning_pipeline import curriculum, grpo_mlx, scoring  # noqa: E402
from vare.learning_pipeline.prompting import PROMPT_VERSION, build_messages  # noqa: E402
from vare.learning_pipeline.repo_env import env_for  # noqa: E402
from vare.learning_pipeline.resources import GiB, memory_snapshot  # noqa: E402

PROTOCOL = ROOT / "protocols/coding_policy_learning_mlx_4b_v1.json"
ADMITTED = ROOT / "results/learning-pipeline-v1/run-001/calibration/admitted.json"
VERIFIER = "trusted_full_v1"


def jdump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def append(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


class ResourceStop(RuntimeError):
    """A frozen resource stop rule fired; the run stops before the next unit of work."""


def swap_fraction() -> float | None:
    """Used / total swap (macOS ``vm.swapusage``; Linux /proc/meminfo), or None if unknown."""
    if platform.system() == "Darwin":
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        m = re.search(r"total = ([\d.]+)M\s+used = ([\d.]+)M", out)
        return float(m.group(2)) / float(m.group(1)) if m and float(m.group(1)) > 0 else None
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    total = int(info["SwapTotal"].split()[0])
    return (total - int(info["SwapFree"].split()[0])) / total if total else None


def peak_rss_gib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
    return round(rss / GiB if platform.system() == "Darwin" else rss * 1024 / GiB, 3)


def telemetry() -> dict[str, Any]:
    import mlx.core as mx

    snap = memory_snapshot()
    out = {"time": round(time.time(), 3), "mlx_active_gib": round(mx.get_active_memory() / GiB, 3),
           "mlx_peak_gib": round(mx.get_peak_memory() / GiB, 3),
           "available_gib": round(snap["available"] / GiB, 2) if snap.get("available") else None,
           "swap_used_gib": round(snap["swap_used"] / GiB, 3) if snap.get("swap_used") is not None else None,
           "swap_fraction": swap_fraction(), "peak_rss_gib": peak_rss_gib()}
    if platform.system() == "Darwin":
        mp = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
        out["memory_free_pct"] = next((ln.split(":")[-1].strip() for ln in mp.splitlines() if "percentage" in ln), None)
        th = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
        out["thermal"] = " ".join(ln.strip() for ln in th.splitlines() if "CPU_" in ln or "warning" in ln.lower())
    return out


@contextmanager
def grad_checkpointing(layer):
    """mlx-lm's grad_checkpoint patches the layer class; undo it afterwards so
    KV-cached generation uses the original ``__call__``."""
    from mlx_lm.tuner.trainer import grad_checkpoint

    cls = type(layer)
    original = cls.__call__
    grad_checkpoint(layer)
    try:
        yield
    finally:
        cls.__call__ = original


def generate(model, tokenizer, messages, *, seed, temperature, top_p, max_tokens):
    # Exact-ratio objective: sample from the untruncated softmax(logits / T) only.
    grpo_mlx.check_exact_ratio_sampling(temperature=temperature, top_p=top_p)
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    mx.random.seed(seed)
    sampler = make_sampler(temp=temperature, top_p=top_p, min_p=0.0, top_k=0)
    toks, sampler_lp, text, finish = [], [], "", "length"
    t0 = time.time()
    for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
        text += r.text
        toks.append(int(r.token))
        sampler_lp.append(float(r.logprobs[r.token]))
        if r.finish_reason:
            finish = r.finish_reason
    return {"prompt_ids": list(prompt), "completion_ids": toks, "text": text, "finish": finish,
            "sampler_logprobs_t1": sampler_lp, "seconds": round(time.time() - t0, 3)}


class Slice:
    def __init__(self, a):
        self.a = a
        grpo_mlx.check_exact_ratio_sampling(temperature=a.temperature, top_p=a.top_p)
        self.run = Path(a.run_dir).expanduser()
        self.run.mkdir(parents=True, exist_ok=True)
        self.cache = Path(a.cache).expanduser()
        self.protocol = json.loads(PROTOCOL.read_text())
        self.b = self.protocol["budgets"]
        self.events = self.run / "events.jsonl"
        self.samples_log = self.run / "samples.jsonl"
        self.steps_log = self.run / "steps.jsonl"
        self.envs: dict[str, Any] = {}
        self.cfg = grpo_mlx.GRPOConfig(clip_eps_low=a.clip, clip_eps_high=a.clip, normalize_std=True,
                                       temperature=a.temperature, max_policy_lag=0, max_grad_norm=1.0,
                                       group_size=a.group_size)

    def guard(self):
        """Between units of work (one rollout, one microbatch): optionally release
        MLX's buffer cache, then apply the frozen swap stop rule."""
        if self.a.clear_cache:
            import mlx.core as mx
            mx.clear_cache()
        frac = swap_fraction()
        if self.a.max_swap_fraction is not None and frac is not None and frac >= self.a.max_swap_fraction:
            raise ResourceStop(f"swap_fraction {frac:.4f} >= {self.a.max_swap_fraction}")

    def note(self, event, **info):
        row = {"event": event, **info, **telemetry()}
        append(self.events, row)
        print(json.dumps(row, default=str)[:600], flush=True)

    # ---------------------------------------------------------------- tasks
    def train_tasks(self):
        manifest = curriculum.load_manifest(ROOT / self.protocol["curriculum"]["train_manifest"])
        admitted = set(json.loads(ADMITTED.read_text())["train"])
        rollout_n = self.b["rollouts"]["tasks"]
        qualified = [t for t in manifest if t["instance_id"] in admitted]
        assert all(t["split"] == "train" for t in qualified)
        if self.a.task_list:
            # A frozen protocol names the tasks; each must be an admitted Train task.
            by_id = {t["instance_id"]: t for t in qualified}
            ids = json.loads(Path(self.a.task_list).expanduser().read_text())["tasks"]
            missing = [i for i in ids if i not in by_id]
            if missing or len(set(ids)) != len(ids):
                raise ValueError(f"task list must name distinct admitted Train tasks; not admitted: {missing}")
            chosen = [by_id[i] for i in ids]
        else:
            chosen = qualified[rollout_n + self.a.task_offset: rollout_n + self.a.task_offset + self.a.tasks]
        tasks = curriculum.attach_payloads(chosen, self.cache)
        for t in tasks:
            env = self.env(t)
            t["_buggy"] = env.buggy_file_text(t["_patch"], t["target_file"])
        return tasks

    def env(self, t):
        e = self.envs.get(t["mirror"])
        if e is None:
            e = env_for(t, self.cache)
            if not e.python.exists() or not (e.path / ".git").exists():
                e.materialize()
            self.envs[t["mirror"]] = e
        return e

    # ---------------------------------------------------------------- model
    def load(self):
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_lm import load

        m = self.protocol["model"]
        path = snapshot_download(m["id"], revision=m["revision"],
                                 allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "tokenizer.model",
                                                 "*.tiktoken", "*.jinja"])
        model, tok = load(path)
        hp = self.b["training"]
        grpo_mlx.make_lora_policy(model, num_layers=hp["num_layers"], rank=hp["lora_rank"],
                                  scale=hp["lora_scale"], dropout=0.0, seed=self.a.seed)
        mx.eval(model.parameters())
        return model, tok

    def ckpt_id(self, model, step):
        return f"grpo:{grpo_mlx.trainable_digest(model)[:16]}:step{step}"

    # ------------------------------------------------------------- sampling
    def sample_groups(self, model, tok, tasks, *, step, stage):
        ck = self.ckpt_id(model, step)
        out = []
        for t in tasks:
            msgs = build_messages(t, t["_buggy"])
            for k in range(self.a.group_size):
                seed = int(hashlib.sha256(f"{self.a.seed}:{stage}:{step}:{t['instance_id']}:{k}".encode())
                           .hexdigest()[:8], 16)
                g = generate(model, tok, msgs, seed=seed, temperature=self.a.temperature, top_p=self.a.top_p,
                             max_tokens=self.b["inference"]["max_tokens"])
                env = self.env(t)
                t0 = time.time()
                rec = scoring.score_candidate(env, t, t["_buggy"], g["text"], timeout=self.b["grading"]["timeout_s"])
                grade_s = round(time.time() - t0, 2)
                s = grpo_mlx.Sample(instance_id=t["instance_id"], behavior_checkpoint_id=ck, behavior_step=step,
                                    prompt_ids=g["prompt_ids"], completion_ids=g["completion_ids"],
                                    reward=float(rec["reward"]), verifier=VERIFIER, sample_index=k)
                out.append((s, g))
                self.guard()
                append(self.samples_log, {
                    "stage": stage, "step": step, "group_id": f"{ck}|{t['instance_id']}",
                    "behavior_checkpoint_id": ck, "instance_id": t["instance_id"], "task_group": t["upstream_repo"],
                    "split": t["split"], "sample_index": k, "seed": seed, "temperature": self.a.temperature,
                    "top_p": self.a.top_p, "prompt_version": PROMPT_VERSION, "prompt_tokens": len(g["prompt_ids"]),
                    "completion_tokens": len(g["completion_ids"]), "finish": g["finish"],
                    "output_sha256": rec["output_sha256"], "reward": rec["reward"], "outcome": rec["outcome"],
                    "verifier": VERIFIER, "cheap_passed": rec["verifiers"]["cheap_f2p_v1"]["passed"],
                    "trusted_passed": rec["verifiers"][VERIFIER]["passed"], "gen_seconds": g["seconds"],
                    "grade_seconds": grade_s, "output_text": g["text"]})
        for e in self.envs.values():
            e.reset()
        return ck, out

    @staticmethod
    def group_stats(samples):
        groups: dict[str, list[float]] = {}
        for s in samples:
            groups.setdefault(s.instance_id, []).append(s.reward)
        mixed = [k for k, v in groups.items() if len(set(v)) > 1]
        zero = [k for k, v in groups.items() if set(v) == {0.0}]
        one = [k for k, v in groups.items() if set(v) == {1.0}]
        n = len(groups)
        return {"groups": n, "mixed": len(mixed), "all_zero": len(zero), "all_one": len(one),
                "mixed_fraction": len(mixed) / n if n else None,
                "wasted_generations": sum(len(groups[k]) for k in zero + one),
                "mean_reward": sum(sum(v) for v in groups.values()) / max(1, sum(len(v) for v in groups.values())),
                "mean_within_group_variance": (sum(_var(v) for v in groups.values()) / n) if n else None,
                "mixed_tasks": mixed, "per_task_pass": {k: sum(v) for k, v in groups.items()}}

    # ---------------------------------------------------------------- run
    def main(self):
        import mlx.core as mx

        a = self.a
        meta = {"schema": "vare.grpo_vertical_slice.run/v1", "evidence_level": "real RL parameter updates (Train only)",
                "model": self.protocol["model"], "args": {k: v for k, v in vars(a).items()
                                                          if k not in ("run_dir", "cache", "task_list")},
                "grpo_config": self.cfg.to_dict(), "lora": {k: self.b["training"][k] for k in
                                                            ("num_layers", "lora_rank", "lora_scale")},
                "optimizer": {"name": "Adam", "learning_rate": a.lr}, "prompt_version": PROMPT_VERSION,
                "sampling": {"temperature": a.temperature, "top_p": a.top_p, "top_k": 0, "min_p": 0.0,
                             "distribution": "untruncated softmax(logits / temperature)",
                             "ratio": "exact (behavior and current log-probs under the sampled distribution)"},
                "platform": platform.platform(), "machine": platform.machine(), "mlx": mx.__version__,
                "heldout_manifest_loaded": False,
                "task_list": Path(a.task_list).name if a.task_list else None,
                "task_list_sha256": (hashlib.sha256(Path(a.task_list).expanduser().read_bytes()).hexdigest()
                                     if a.task_list else None)}
        jdump(self.run / "run_meta.json", meta)
        t_all = time.time()
        start_swap = swap_fraction()
        self.note("start", memory=memory_snapshot(), start_swap_fraction=start_swap)
        if (a.start_max_swap_fraction is not None and start_swap is not None
                and start_swap >= a.start_max_swap_fraction):
            jdump(self.run / "STOPPED.json", {"reason": "start_swap_at_or_above_limit", "swap_fraction": start_swap,
                                              "limit": a.start_max_swap_fraction, "started": False})
            self.note("not_started", swap_fraction=start_swap)
            return
        if a.mlx_cache_limit_gib is not None:
            mx.set_cache_limit(int(a.mlx_cache_limit_gib * GiB))
        try:
            self._run(t_all)
        except ResourceStop as stop:
            jdump(self.run / "STOPPED.json", {"reason": "resource_stop_rule", "detail": str(stop), "started": True})
            self.note("stopped", detail=str(stop))
            raise SystemExit(3)

    def _run(self, t_all):
        import mlx.core as mx
        import mlx.optimizers as optim

        a = self.a
        tasks = self.train_tasks()
        jdump(self.run / "train_tasks.json", [{k: t[k] for k in ("instance_id", "upstream_repo", "split",
                                                                    "target_file", "patch_sha256")} for t in tasks])
        mx.reset_peak_memory()
        model, tok = self.load()
        self.note("loaded")
        opt = optim.Adam(learning_rate=a.lr)
        t0 = time.time()
        grpo_mlx.reseed(a.seed, 0)
        ck0, pairs = self.sample_groups(model, tok, tasks, step=0, stage="survey")
        survey = self.group_stats([s for s, _ in pairs])
        survey["seconds"] = round(time.time() - t0, 1)
        survey["generations"] = len(pairs)
        gen_tokens, gen_s_total = sum(len(g["completion_ids"]) for _, g in pairs), sum(g["seconds"] for _, g in pairs)
        survey["generation_tokens_per_s"] = round(gen_tokens / gen_s_total, 2) if gen_s_total else None
        jdump(self.run / "survey.json", survey)
        self.note("survey", **{k: v for k, v in survey.items() if k != "per_task_pass"})
        informative = [t for t in tasks if t["instance_id"] in survey["mixed_tasks"]]
        step_tasks = informative[: a.groups_per_step] or tasks[: a.groups_per_step]
        drift_probe = [(s, g) for s, g in pairs if s.instance_id in survey["mixed_tasks"]]

        step_records = []
        for step in range(a.steps):
            grpo_mlx.reseed(a.seed, step)
            t_step = time.time()
            if step == 0:
                ck, cur = ck0, pairs
                gen_s = 0.0
            else:
                ck, cur = self.sample_groups(model, tok, step_tasks, step=step, stage="update")
                gen_s = time.time() - t_step
            samples = [s for s, _ in cur]
            t_lp = time.time()
            if a.clear_cache:
                mx.clear_cache()
            mx.reset_peak_memory()
            swap_before = telemetry()["swap_used_gib"]
            with grad_checkpointing(model.layers[0]):  # behavior pass on the exact update path
                grpo_mlx.compute_behavior_logprobs(model, grpo_mlx.informative_samples(samples, self.cfg), self.cfg,
                                                   after_sample=self.guard)
                lp_s = time.time() - t_lp
                mismatch = sampler_forward_mismatch(model, [p for p in cur if p[0].behavior_logprobs][:2])
                t_upd = time.time()
                rec = grpo_mlx.train_step(model, opt, samples, self.cfg, policy_checkpoint_id=ck, policy_step=step,
                                          after_sample=self.guard)
                upd_s = time.time() - t_upd
            rec["update_peak_gib"] = round(mx.get_peak_memory() / GiB, 3)
            tel = telemetry()
            rec["memory"] = {"peak_rss_gib": tel["peak_rss_gib"], "swap_used_gib_before_update": swap_before,
                             "swap_used_gib_after_update": tel["swap_used_gib"], "swap_fraction": tel["swap_fraction"]}
            if a.clear_cache:
                mx.clear_cache()
            rec.update(step_stats=self.group_stats(samples), seconds={
                "generate_and_grade": round(gen_s, 1), "behavior_logprobs": round(lp_s, 1),
                "update": round(upd_s, 1), "total": round(time.time() - t_step, 1)},
                tokens_trained=rec["n_tokens"], sampler_vs_forward_t1=mismatch,
                update_tokens_per_s=round(rec["n_tokens"] / upd_s, 1) if rec.get("update_applied") else None)
            path = self.run / "checkpoints" / f"step-{step + 1:04d}"
            path.parent.mkdir(exist_ok=True)
            man = grpo_mlx.save_checkpoint(path, model, opt, step=step + 1, base_seed=a.seed,
                                           extra={"data_cursor": step + 1, "behavior_checkpoint_id": ck})
            rec["checkpoint"] = {"dir": f"checkpoints/{path.name}", "trainable_digest": man["trainable_digest"]}
            append(self.steps_log, rec)
            step_records.append(rec)
            self.note("step", step=step, **{k: rec.get(k) for k in (
                "update_applied", "n_informative_groups", "n_tokens", "loss", "grad_norm_preclip", "clip_fraction",
                "ratio_mean", "approx_kl", "max_abs_log_ratio", "params_changed", "update_peak_gib", "seconds")})

        # context-dependent change on the step-0 informative samples
        drift = []
        for s, _ in drift_probe:
            tokens, mask, start = grpo_mlx.sample_tensors(s, 0)
            lp_new = grpo_mlx.token_logprobs(model, tokens, temperature=self.cfg.temperature, start=start)
            mx.eval(lp_new)
            new = float(lp_new[0, : len(s.completion_ids)].sum())
            old = float(sum(s.behavior_logprobs or [])) if s.behavior_logprobs else None
            drift.append({"instance_id": s.instance_id, "sample_index": s.sample_index, "reward": s.reward,
                          "seq_logprob_before": old, "seq_logprob_after": new,
                          "delta": None if old is None else new - old})
        final_digest = grpo_mlx.trainable_digest(model)
        last = self.run / "checkpoints" / f"step-{a.steps:04d}"
        del model, opt
        mx.clear_cache()

        # reload into a fresh model and generate post-update
        t_r = time.time()
        model2, tok2 = self.load()
        opt2 = optim.Adam(learning_rate=a.lr)
        man = grpo_mlx.load_checkpoint(last, model2, opt2)
        reload_ok = grpo_mlx.trainable_digest(model2) == final_digest == man["trainable_digest"]
        post_ck, post = self.sample_groups(model2, tok2, step_tasks, step=a.steps, stage="post_update")
        post_stats = self.group_stats([s for s, _ in post])
        self.note("post_update", reload_ok=reload_ok, **{k: v for k, v in post_stats.items() if k != "per_task_pass"})
        pre_on_same = {t["instance_id"]: survey["per_task_pass"][t["instance_id"]] for t in step_tasks}
        summary = {
            "schema": "vare.grpo_vertical_slice.summary/v1",
            "evidence_level": "real RL parameter updates on Train tasks; not a capability measurement",
            "train_tasks": len(tasks), "group_size": a.group_size, "steps": a.steps,
            "survey": {k: v for k, v in survey.items()},
            "steps_applied": sum(bool(r.get("update_applied")) for r in step_records),
            "params_changed_every_applied_step": all(r.get("params_changed") for r in step_records
                                                     if r.get("update_applied")),
            "step_metrics": [{k: r.get(k) for k in (
                "policy_step", "update_applied", "n_groups", "n_informative_groups", "n_trained_samples",
                "n_tokens", "loss", "grad_norm_preclip", "clip_fraction", "ratio_mean", "approx_kl", "max_abs_log_ratio",
                "params_changed", "update_peak_gib", "seconds", "update_tokens_per_s")} for r in step_records],
            "context_dependent_drift": drift,
            "reload": {"ok": reload_ok, "seconds_incl_post_generation": round(time.time() - t_r, 1),
                       "checkpoint_step": man["step"]},
            "post_update_same_train_tasks": {"pass_per_task": post_stats["per_task_pass"],
                                             "pre_update_pass_per_task": pre_on_same,
                                             "note": "same Train tasks as the update; descriptive only"},
            "wall_clock_s": round(time.time() - t_all, 1),
            "peak_mlx_gib_overall": round(max([r.get("update_peak_gib") or 0 for r in step_records]), 3),
            "final_telemetry": telemetry(),
        }
        jdump(self.run / "summary.json", summary)
        self.note("done", wall_clock_s=summary["wall_clock_s"])


def sampler_forward_mismatch(model, pairs):
    """Max |sampler log-prob - full-forward log-prob| at T=1 for sampled tokens
    (KV-cached incremental decode vs one full pass of the same parameters)."""
    import mlx.core as mx

    worst = 0.0
    for s, g in pairs:
        tokens, _, start = grpo_mlx.sample_tensors(s, 0)
        lp = grpo_mlx.token_logprobs(model, tokens, temperature=1.0, start=start)
        mx.eval(lp)
        fwd = lp[0, : len(s.completion_ids)].tolist()
        worst = max([worst] + [abs(x - y) for x, y in zip(fwd, g["sampler_logprobs_t1"])])
    return worst


def _var(v):
    m = sum(v) / len(v)
    return sum((x - m) ** 2 for x in v) / len(v)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--tasks", type=int, default=12, help="Train tasks in the survey")
    p.add_argument("--task-offset", type=int, default=0, help="offset after run-001's rollout tasks")
    p.add_argument("--task-list", default=None,
                   help="frozen protocol JSON whose 'tasks' lists admitted Train instance ids (overrides --tasks)")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--groups-per-step", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=grpo_mlx.EXACT_RATIO_TOP_P,
                   help="must be 1.0: the exact-ratio objective models the untruncated softmax(logits / T)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mlx-cache-limit-gib", type=float, default=None,
                   help="mx.set_cache_limit for the run (default: MLX's own limit)")
    p.add_argument("--clear-cache", action="store_true",
                   help="mx.clear_cache() after every rollout and microbatch and between phases")
    p.add_argument("--max-swap-fraction", type=float, default=None,
                   help="stop rule: stop before the next rollout/microbatch once swap used/total reaches this")
    p.add_argument("--start-max-swap-fraction", type=float, default=None,
                   help="start gate: do not load the model if swap used/total is already at or above this")
    return p.parse_args(argv)


if __name__ == "__main__":
    Slice(parse()).main()
