"""Orchestration of coding-policy-learning v1 (MLX, single Apple-silicon Mac).

Phases (each resumable; outputs under ``run_dir``):

  preflight  -> disk/memory probe, one-task setup+grade+generate (+ optional
                short training smoke) measurements, projections, stop rules
  calibrate  -> materialize repos, baseline-fails / reference-passes check,
                prompt-length admission; held-out minimum stop rule
  rollout    -> base-policy samples on Train tasks (RFT data), persisted
                before grading; graded with both verifiers afterwards
  train      -> sft_reference and rft_self LoRA adapters, matched budgets
  evaluate   -> each arm, greedy, identical budgets on the qualified held-out
                set; outputs persisted before grading
  compare    -> paired McNemar + Holm + bootstrap, frozen decision rule

Synthetic modes (``mock``: MockPolicy and no training; ``synthetic_model``: a
tiny randomly initialized MLX model) write ``"synthetic": true`` into every
artifact and their numbers are not evidence about any real policy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Iterable, Mapping

from . import comparison, curriculum, resources, scoring, training
from .policy import MLXPolicy, MockPolicy
from .prompting import PROMPT_VERSION, build_messages
from .repo_env import RepoEnv, dump_json, env_for
from .resources import GiB, StopRule

ARMS = ("no_update", "sft_reference", "rft_self")
# One JSON object per (checkpoint, task, sample). Keys sufficient for offline
# replay without the model: checkpoint_id/step, group_id (= checkpoint x task),
# task_group (repository cluster), sample_index, output_text, final_diff, and
# per-test statuses + wall times for both verifiers.
ROLLOUT_SCHEMA = "vare.learning_pipeline.rollout/v1"


# --------------------------------------------------------------------- utils
def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # torn last line after a crash
                pass
    return rows


def code_hashes(repo_root: Path, files: Iterable[str]) -> dict[str, str]:
    return {f: curriculum.sha256_file(repo_root / f) for f in files}


def verify_protocol(protocol: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """Recompute every frozen hash; raise StopRule on any mismatch."""
    frozen = protocol["frozen"]
    bad = {}
    for rel, want in {**frozen["inputs"], **frozen["code"]}.items():
        got = curriculum.sha256_file(repo_root / rel) if (repo_root / rel).exists() else None
        if got != want:
            bad[rel] = {"want": want, "got": got}
    if protocol["budgets"]["prompt_version"] != PROMPT_VERSION:
        bad["prompt_version"] = {"want": protocol["budgets"]["prompt_version"], "got": PROMPT_VERSION}
    if bad:
        raise StopRule("protocol_hash_mismatch", bad)
    return {"verified_files": len(frozen["inputs"]) + len(frozen["code"])}


class Run:
    def __init__(self, protocol_path: Path, repo_root: Path, run_dir: Path, cache: Path, *,
                 mode: str = "real", model_override: str | None = None, log: Callable[[str], None] = print):
        self.protocol_path = Path(protocol_path)
        self.protocol = load_json(protocol_path)
        self.repo_root = Path(repo_root)
        self.run_dir = Path(run_dir)
        self.cache = Path(cache)
        self.mode = mode  # real | synthetic_model | mock
        self.synthetic = mode != "real"
        self.model_override = model_override
        self.log = log
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.b = self.protocol["budgets"]
        self.stops = self.protocol["stopping_rules"]
        self._tasks: dict[str, list[dict[str, Any]]] | None = None
        self._model_path: str | None = None
        self.started = time.time()

    # ------------------------------------------------------------ helpers
    def note(self, event: str, **info: Any) -> None:
        row = {"event": event, "time": round(time.time(), 3), "synthetic": self.synthetic, **info}
        append_jsonl(self.run_dir / "events.jsonl", row)
        self.log(f"[{event}] " + json.dumps(info, default=str)[:400])

    def guard(self, stage: str) -> None:
        resources.check_disk(self.cache, self.stops["min_free_disk_gib"], stage=stage)
        resources.check_disk(self.run_dir, self.stops["min_free_disk_gib"], stage=stage)
        elapsed = time.time() - self.started
        cap = self.stops["max_total_wall_clock_s"]
        if self.mode == "real" and elapsed > cap:
            raise StopRule("wall_clock", {"stage": stage, "elapsed_s": round(elapsed), "cap_s": cap})

    def tasks(self) -> dict[str, list[dict[str, Any]]]:
        if self._tasks is None:
            cur = self.protocol["curriculum"]
            out = {}
            for split in ("train", "heldout"):
                manifest = curriculum.load_manifest(self.repo_root / cur[f"{split}_manifest"])
                out[split] = curriculum.attach_payloads(manifest, self.cache)
            self._tasks = out
        return self._tasks

    def model_path(self) -> str:
        if self.model_override:
            return self.model_override
        if self._model_path is None:
            from huggingface_hub import snapshot_download

            m = self.protocol["model"]
            self._model_path = snapshot_download(m["id"], revision=m["revision"],
                                                 allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt",
                                                                 "tokenizer.model", "*.tiktoken", "*.jinja"])
        return self._model_path

    def policy(self, arm: str):
        if self.mode == "mock":
            return MockPolicy(self.protocol["synthetic_mock"]["p_correct"][arm], policy_id=arm)
        adapter = None if arm == "no_update" else str(self.run_dir / "adapters" / arm)
        return MLXPolicy(self.model_path(), adapter_path=adapter,
                         max_prompt_tokens=self.b["inference"]["max_prompt_tokens"], synthetic=self.synthetic)

    def checkpoint(self, arm: str) -> dict[str, Any]:
        """Stable id + training step of the policy that produced a sample."""
        if self.mode == "mock":
            model = "MOCK-no-model"
        else:
            model = self.model_override or f"{self.protocol['model']['id']}@{self.protocol['model']['revision']}"
        if arm == "no_update":
            return {"checkpoint_id": f"base:{model}", "checkpoint_step": 0}
        adapter = self.run_dir / "adapters" / arm
        done = adapter / "TRAINING_COMPLETE.json"
        path = adapter / "adapters.safetensors"
        digest = curriculum.sha256_file(path)[:16] if path.exists() else "mock-untrained"
        step = self.b["training"]["iters"] if done.exists() or self.mode == "mock" else None
        return {"checkpoint_id": f"{arm}:{digest}:{model}", "checkpoint_step": step}

    def buggy_texts(self) -> dict[str, str]:
        return load_json(self.run_dir / "calibration" / "buggy_files.json")

    def qualified(self, split: str) -> list[dict[str, Any]]:
        q = set(load_json(self.run_dir / "calibration" / "admitted.json")[split])
        return [t for t in self.tasks()[split] if t["instance_id"] in q]

    # ---------------------------------------------------------- preflight
    def preflight(self, *, train_smoke: bool = True) -> dict[str, Any]:
        out_path = self.run_dir / "preflight.json"
        pf = self.protocol["preflight"]
        report: dict[str, Any] = {"synthetic": self.synthetic, "mode": self.mode,
                                  "memory_start": resources.memory_snapshot()}
        free0 = resources.check_disk(self.cache, self.stops["min_free_disk_gib"], stage="preflight_start")
        report["free_disk_start_gib"] = round(free0 / GiB, 2)
        resources.check_memory(self.stops["min_available_memory_gib_start"], stage="preflight_start") \
            if self.mode == "real" else None

        t0 = time.time()
        cache_before = resources.dir_bytes(self.cache)
        tasks = self.tasks()
        report["dataset_seconds"] = round(time.time() - t0, 1)
        report["dataset_bytes"] = resources.dir_bytes(self.cache / "datasets")

        # The probe is a Train task: preflight never touches held-out tasks.
        task = next(t for t in tasks["train"] if t["instance_id"] == pf["probe_instance_id"]) \
            if pf.get("probe_instance_id") else tasks["train"][0]
        env = env_for(task, self.cache)
        before = resources.dir_bytes(self.cache / "repos") + resources.dir_bytes(self.cache / "venvs")
        t1 = time.time()
        env.materialize()
        report["probe_task"] = task["instance_id"]
        report["repo_setup_seconds"] = round(time.time() - t1, 1)
        report["repo_setup_bytes"] = (resources.dir_bytes(env.path) + resources.dir_bytes(env.venv))
        cal = scoring.calibrate(env, task, timeout=self.b["grading"]["timeout_s"])
        report["probe_calibration"] = {k: cal[k] for k in ("qualified", "baseline_fails_f2p", "reference_passes")}
        report["grade_seconds"] = round(cal["baseline"]["seconds"] + cal["reference"]["seconds"], 2) / 2

        model_bytes = 0
        if self.mode != "mock":
            t2 = time.time()
            path = self.model_path()
            report["model_download_or_resolve_seconds"] = round(time.time() - t2, 1)
            model_bytes = resources.dir_bytes(Path(path))
            report["model_bytes"] = model_bytes
            pol = self.policy("no_update")
            buggy = env.buggy_file_text(task["_patch"], task["target_file"])
            inf = self.b["inference"]
            gen = pol.generate(build_messages(task, buggy), max_tokens=inf["max_tokens"], temperature=0.0,
                               top_p=1.0, seed=0, task=task)
            report["probe_generation"] = {"prompt_tokens": gen.prompt_tokens,
                                          "completion_tokens": gen.completion_tokens,
                                          "seconds": gen.seconds, "finish": gen.finish,
                                          "peak_memory_gib": round((pol.peak_memory_bytes() or 0) / GiB, 2)}
            pol.close()
            del pol
            if train_smoke:
                report["train_smoke"] = self._train_smoke(task, buggy)
        report["cache_growth_bytes"] = resources.dir_bytes(self.cache) - cache_before
        report["memory_end"] = resources.memory_snapshot()
        report["projection"] = self._project(report)
        dump_json(out_path, report)
        self.note("preflight", projection=report["projection"])
        proj = report["projection"]
        if proj["free_disk_after_run_gib"] < self.stops["min_free_disk_gib"]:
            raise StopRule("disk_budget_projection", proj)
        if self.mode == "real" and proj["total_seconds"] > self.stops["max_projected_seconds"]:
            raise StopRule("wall_clock_projection", proj)
        smoke = report.get("train_smoke") or {}
        if self.mode == "real" and (smoke.get("peak_memory_gb") or 0) * 1e9 > self.stops["max_train_peak_memory_gib"] * GiB:
            raise StopRule("train_memory", smoke)
        return report

    def _train_smoke(self, task, buggy) -> dict[str, Any]:
        """A few LoRA iterations at the longest admissible sequence, to measure
        peak memory and seconds/iteration before committing to the full run.
        The probe Train task's prompt is padded towards ``max_seq_length`` so
        the measured peak is close to the worst admissible case. The adapter
        is deleted and never evaluated."""
        smoke_dir = self.run_dir / "preflight_train_smoke"
        if smoke_dir.exists():
            shutil.rmtree(smoke_dir)
        ex = training.sft_examples([task], {task["instance_id"]: buggy},
                                   {task["instance_id"]: {"qualified": True}})
        if not ex:
            return {"skipped": "no_example"}
        target_chars = int(0.85 * self.b["training"]["max_seq_length"] * 3.0)
        user = ex[0]["messages"][1]["content"]
        filler = "# memory-probe filler line\n" * max(0, (target_chars - len(user)) // 27)
        ex[0]["messages"][1]["content"] = filler + user
        rows = ex * 4
        training.write_dataset(rows, smoke_dir / "data", valid_count=1, seed=0)
        hp = dict(self.b["training"], iters=self.protocol["preflight"]["train_smoke_iters"],
                  steps_per_report=1, steps_per_eval=10_000, save_every=10_000, val_batches=1)
        cfg = training.lora_config(model=self.model_path(), data_dir=smoke_dir / "data",
                                   adapter_dir=smoke_dir / "adapter", hp=hp)
        res = training.run_lora(cfg, smoke_dir / "train.log", timeout_s=1800, resume=False)
        metrics = training.read_metrics(smoke_dir / "adapter" / "metrics.jsonl")
        tr = [m for m in metrics if m["kind"] == "train"]
        # mlx-lm reports throughput in *unmasked target* tokens; the projection
        # uses seconds per iteration of the padded worst-case sequence instead.
        ips = tr[-1].get("iterations_per_second") if tr else None
        out = {"iters": hp["iters"], "seconds": res["seconds"], "peak_memory_gb": res["peak_memory_gb"],
               "seconds_per_iteration": round(1.0 / ips, 3) if ips else None,
               "target_tokens_per_second": tr[-1]["tokens_per_second"] if tr else None,
               "trained_tokens": res["trained_tokens"]}
        shutil.rmtree(smoke_dir / "adapter", ignore_errors=True)
        return out

    def _project(self, r: Mapping[str, Any]) -> dict[str, Any]:
        b = self.b
        n_repos = len({t["upstream_repo"] for s in self.tasks().values() for t in s})
        n_tasks = sum(len(s) for s in self.tasks().values())
        n_heldout = len(self.tasks()["heldout"])
        roll = b["rollouts"]
        grade_s = float(r["grade_seconds"]) or 1.0
        gen = r.get("probe_generation") or {}
        gen_s = float(gen.get("seconds") or 0.0)
        # sampled generations rarely stop early; assume max_tokens at measured speed
        tok_s = (gen.get("completion_tokens") or 0) / gen_s if gen_s else None
        sample_s = (roll["max_tokens"] / tok_s + 3.0) if tok_s else 0.0
        eval_s = (b["inference"]["max_tokens"] / tok_s + 3.0) if tok_s else 0.0
        smoke = r.get("train_smoke") or {}
        spi = smoke.get("seconds_per_iteration")
        # upper-bound: every iteration costs as much as the padded ~0.85 x max_seq probe
        train_s = (b["training"]["iters"] * spi + 120) if spi else 0.0
        seconds = {
            "setup_repos": r["repo_setup_seconds"] * n_repos,
            "calibrate": 2 * grade_s * n_tasks,
            "rollouts": roll["tasks"] * roll["samples_per_task"] * (sample_s + 2 * grade_s),
            "train_two_arms": 2 * train_s,
            "evaluate_three_arms": 3 * n_heldout * (eval_s + 2 * grade_s) + 3 * 60,
        }
        need = (r["repo_setup_bytes"] * (n_repos - 1)) + 0.3 * GiB  # adapters, logs, outputs
        free_now = resources.free_disk_bytes(self.cache)
        return {
            "seconds": {k: round(v) for k, v in seconds.items()},
            "total_seconds": round(sum(seconds.values())),
            "projected_additional_disk_gib": round(need / GiB, 2),
            "free_disk_now_gib": round(free_now / GiB, 2),
            "free_disk_after_run_gib": round((free_now - need) / GiB, 2),
            "assumptions": "sampled outputs run to max_tokens; every training iteration as slow as the padded preflight probe; +120 s load per arm",
            "measured_training_throughput": spi is not None,
        }

    # ---------------------------------------------------------- calibrate
    def calibrate(self) -> dict[str, Any]:
        cal_dir = self.run_dir / "calibration"
        done_path = cal_dir / "admitted.json"
        if done_path.exists():
            return load_json(done_path)
        tasks = self.tasks()
        results_path = cal_dir / "results.jsonl"
        done = {r["instance_id"]: r for r in read_jsonl(results_path)}
        buggy_path = cal_dir / "buggy_files.json"
        buggy: dict[str, str] = load_json(buggy_path) if buggy_path.exists() else {}
        envs: dict[str, RepoEnv] = {}
        for split in ("train", "heldout"):
            for t in tasks[split]:
                if t["instance_id"] in done:
                    continue
                self.guard("calibrate")
                env = envs.get(t["mirror"])
                if env is None:
                    env = env_for(t, self.cache)
                    env.materialize()
                    envs[t["mirror"]] = env
                cal = scoring.calibrate(env, t, timeout=self.b["grading"]["timeout_s"])
                buggy[t["instance_id"]] = env.buggy_file_text(t["_patch"], t["target_file"])
                cal["split"] = split
                append_jsonl(results_path, cal)
                done[t["instance_id"]] = cal
            dump_json(buggy_path, buggy)
        env_last = None
        for env_last in envs.values():
            env_last.reset()
        prompt_tokens = self._prompt_tokens(tasks, buggy)
        max_prompt = self.b["inference"]["max_prompt_tokens"]
        admitted: dict[str, Any] = {"synthetic": self.synthetic}
        drops: dict[str, dict[str, int]] = {}
        for split in ("train", "heldout"):
            keep = []
            for t in tasks[split]:
                c = done[t["instance_id"]]
                why = None
                if not c["baseline_fails_f2p"]:
                    why = "baseline_does_not_fail"
                elif not c["baseline_passes_p2p"]:
                    why = "baseline_p2p_not_green"
                elif not c["reference_passes"]:
                    why = "reference_fails"
                elif prompt_tokens.get(t["instance_id"], 0) > max_prompt:
                    why = "prompt_too_long"
                if why:
                    drops.setdefault(split, {}).setdefault(why, 0)
                    drops[split][why] += 1
                else:
                    keep.append(t["instance_id"])
            admitted[split] = keep
        admitted["drops"] = drops
        admitted["prompt_tokens"] = prompt_tokens
        need = self.stops["min_heldout_tasks"]
        self.note("calibrated", train=len(admitted["train"]), heldout=len(admitted["heldout"]), drops=drops)
        dump_json(done_path, admitted)
        if len(admitted["heldout"]) < need:
            raise StopRule("too_few_heldout_tasks", {"qualified": len(admitted["heldout"]), "min": need})
        return admitted

    def _prompt_tokens(self, tasks, buggy) -> dict[str, int]:
        if self.mode == "mock":
            return {t["instance_id"]: len(json.dumps(build_messages(t, buggy[t["instance_id"]]))) // 4
                    for s in tasks.values() for t in s}
        from mlx_lm.utils import load_tokenizer

        tok = load_tokenizer(Path(self.model_path()))
        out = {}
        for s in tasks.values():
            for t in s:
                ids = tok.apply_chat_template(build_messages(t, buggy[t["instance_id"]]),
                                              add_generation_prompt=True, tokenize=True)
                out[t["instance_id"]] = len(ids)
        return out

    # -------------------------------------------------- generation + grading
    def _generate(self, arm: str, tasks: list[dict[str, Any]], out_path: Path, *, samples: int,
                  temperature: float, top_p: float, max_tokens: int, seed_base: int, stage: str) -> None:
        done = {(r["instance_id"], r["sample_index"]) for r in read_jsonl(out_path)}
        todo = [(t, k) for t in tasks for k in range(samples) if (t["instance_id"], k) not in done]
        if not todo:
            return
        buggy = self.buggy_texts()
        ckpt = self.checkpoint(arm)
        pol = self.policy(arm)
        try:
            for t, k in todo:
                self.guard(stage)
                seed = int(hashlib.sha256(f"{seed_base}:{t['instance_id']}:{k}".encode()).hexdigest()[:8], 16)
                g = pol.generate(build_messages(t, buggy[t["instance_id"]]), max_tokens=max_tokens,
                                 temperature=temperature, top_p=top_p, seed=seed, task=t)
                append_jsonl(out_path, {
                    "schema": ROLLOUT_SCHEMA, "stage": stage, **ckpt,
                    "group_id": f"{ckpt['checkpoint_id']}|{t['instance_id']}",
                    "task_group": t["upstream_repo"], "split": t["split"],
                    "arm": arm, "policy_id": pol.policy_id, "instance_id": t["instance_id"], "sample_index": k,
                    "seed": seed, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
                    "output_text": g.text, "output_sha256": hashlib.sha256(g.text.encode()).hexdigest(),
                    "prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens,
                    "seconds": g.seconds, "finish": g.finish, "synthetic": self.synthetic or pol.synthetic,
                })
            peak = pol.peak_memory_bytes()
            self.note("generated", arm=arm, stage=stage, n=len(todo),
                      peak_memory_gib=round(peak / GiB, 2) if peak else None)
        finally:
            pol.close()
            del pol

    def _grade(self, tasks: list[dict[str, Any]], gen_path: Path, graded_path: Path, stage: str) -> list[dict[str, Any]]:
        by_id = {t["instance_id"]: t for t in tasks}
        graded = read_jsonl(graded_path)
        done = {(r["instance_id"], r["sample_index"]) for r in graded}
        buggy = self.buggy_texts()
        envs: dict[str, RepoEnv] = {}
        for g in read_jsonl(gen_path):
            key = (g["instance_id"], g["sample_index"])
            if key in done or g["instance_id"] not in by_id:
                continue
            self.guard(stage)
            t = by_id[g["instance_id"]]
            env = envs.setdefault(t["mirror"], env_for(t, self.cache))
            rec = scoring.score_candidate(env, t, buggy[t["instance_id"]], g["output_text"],
                                          timeout=self.b["grading"]["timeout_s"])
            rec.update({k: g[k] for k in ("schema", "stage", "checkpoint_id", "checkpoint_step", "group_id",
                                          "task_group", "split", "arm", "policy_id", "sample_index", "seed",
                                          "temperature", "top_p", "max_tokens", "output_text", "finish",
                                          "prompt_tokens", "completion_tokens", "seconds", "synthetic")})
            rec["generation_seconds"] = rec.pop("seconds")
            append_jsonl(graded_path, rec)
            graded.append(rec)
            done.add(key)
        for env in envs.values():
            env.reset()
        return graded

    # ------------------------------------------------------------ rollout
    def rollout(self) -> dict[str, Any]:
        roll = self.b["rollouts"]
        tasks = self.qualified("train")[: roll["tasks"]]
        d = self.run_dir / "rollouts"
        self._generate("no_update", tasks, d / "generations.jsonl", samples=roll["samples_per_task"],
                       temperature=roll["temperature"], top_p=roll["top_p"], max_tokens=roll["max_tokens"],
                       seed_base=roll["seed"], stage="rollout")
        graded = self._grade(tasks, d / "generations.jsonl", d / "graded.jsonl", "rollout")
        from .grpo import informative_groups

        summary = {
            "synthetic": self.synthetic,
            "tasks": len(tasks), "samples": len(graded),
            "trusted_pass": sum(r["reward"] for r in graded),
            "tasks_with_pass": len({r["instance_id"] for r in graded if r["reward"]}),
            "cheap_false_positives": sum(bool(r.get("cheap_false_positive")) for r in graded),
            "outcomes": _count(r["outcome"] for r in graded),
            "grpo_informative_groups": len(informative_groups(graded)),
        }
        dump_json(d / "summary.json", summary)
        self.note("rollout", **summary)
        return summary

    # -------------------------------------------------------------- train
    def train(self) -> dict[str, Any]:
        hp = self.b["training"]
        buggy = self.buggy_texts()
        train_tasks = self.qualified("train")
        cal = {r["instance_id"]: {"qualified": True} for r in train_tasks}
        rollouts = read_jsonl(self.run_dir / "rollouts" / "graded.jsonl")
        datasets = {
            "sft_reference": training.sft_examples(train_tasks, buggy, cal),
            "rft_self": training.rft_examples(train_tasks, buggy, rollouts,
                                              max_per_task=self.b["rft"]["max_examples_per_task"]),
        }
        report: dict[str, Any] = {"synthetic": self.synthetic}
        for arm, rows in datasets.items():
            self.guard(f"train_{arm}")
            n_tasks = len({r["instance_id"] for r in rows})
            info: dict[str, Any] = {"examples": len(rows), "tasks": n_tasks}
            if arm == "rft_self" and (len(rows) < self.stops["rft_min_examples"] or n_tasks < self.stops["rft_min_tasks"]):
                info["status"] = "NO_GO_insufficient_self_successes"
                report[arm] = info
                self.note("train_skipped", arm=arm, **info)
                continue
            data_dir = self.run_dir / "data" / arm
            info["dataset"] = training.write_dataset(rows, data_dir, valid_count=hp["valid_rows"], seed=hp["seed"])
            info["dataset_sha256"] = {n: curriculum.sha256_file(data_dir / f"{n}.jsonl") for n in ("train", "valid")}
            adapter_dir = self.run_dir / "adapters" / arm
            if self.mode == "mock":
                adapter_dir.mkdir(parents=True, exist_ok=True)
                (adapter_dir / "MOCK_NO_TRAINING.txt").write_text("synthetic mock arm; no weights were trained\n")
                info["status"] = "mock_not_trained"
            else:
                cfg = training.lora_config(model=self.model_path(), data_dir=data_dir, adapter_dir=adapter_dir, hp=hp)
                info["run"] = training.run_lora(cfg, self.run_dir / f"train_{arm}.log",
                                                timeout_s=self.stops["max_train_seconds_per_arm"])
                info["adapter"] = training.adapter_fingerprint(adapter_dir)
                if not info["adapter"].get("lora_b_nonzero_tensors"):
                    raise StopRule("adapter_not_updated", {"arm": arm, **info["adapter"]})
                info["status"] = "trained"
            report[arm] = info
            self.note("trained", arm=arm, **{k: v for k, v in info.items() if k != "dataset_sha256"})
        dump_json(self.run_dir / "train_report.json", report)
        return report

    # ----------------------------------------------------------- evaluate
    def evaluate(self) -> dict[str, Any]:
        inf = self.b["inference"]
        tasks = self.qualified("heldout")
        train_report = load_json(self.run_dir / "train_report.json")
        arms = [a for a in ARMS if a == "no_update" or train_report.get(a, {}).get("status") in
                ("trained", "mock_not_trained")]
        results: dict[str, Any] = {"synthetic": self.synthetic, "arms": {}}
        for arm in arms:
            d = self.run_dir / "eval" / arm
            self._generate(arm, tasks, d / "generations.jsonl", samples=inf["eval_samples_per_task"],
                           temperature=inf["eval_temperature"], top_p=1.0, max_tokens=inf["max_tokens"],
                           seed_base=inf["seed"], stage=f"eval_{arm}")
            graded = self._grade(tasks, d / "generations.jsonl", d / "graded.jsonl", f"eval_{arm}")
            results["arms"][arm] = {
                "n": len(graded), "trusted_pass": sum(r["reward"] for r in graded),
                "cheap_pass": sum(bool(r["verifiers"][scoring.CHEAP]["passed"]) for r in graded),
                "outcomes": _count(r["outcome"] for r in graded),
                "finish": _count(r["finish"] for r in graded),
            }
        dump_json(self.run_dir / "eval_summary.json", results)
        self.note("evaluated", **{a: v["trusted_pass"] for a, v in results["arms"].items()})
        return results

    # ------------------------------------------------------------ compare
    def compare(self) -> dict[str, Any]:
        tasks = [t["instance_id"] for t in self.qualified("heldout")]
        outcomes = {}
        for arm in ARMS:
            p = self.run_dir / "eval" / arm / "graded.jsonl"
            rows = [r for r in read_jsonl(p) if r["sample_index"] == 0]
            if rows:
                by = {r["instance_id"]: r["reward"] for r in rows}
                missing = [t for t in tasks if t not in by]
                if missing:
                    raise StopRule("incomplete_eval", {"arm": arm, "missing": len(missing)})
                outcomes[arm] = by
        decision = comparison.decide(outcomes, tasks, self.protocol["success_criteria"])
        decision["synthetic"] = self.synthetic
        decision["label"] = ("SYNTHETIC - not evidence about any real policy" if self.synthetic
                             else "measured")
        dump_json(self.run_dir / "comparison.json", decision)
        self.note("compared", **{a: c["verdict"] for a, c in decision["comparisons"].items()})
        return decision

    def run_all(self, *, skip_preflight: bool = False, train_smoke: bool = True) -> dict[str, Any]:
        verify_protocol(self.protocol, self.repo_root)
        dump_json(self.run_dir / "run_meta.json", {
            "protocol": str(self.protocol_path), "protocol_sha256": curriculum.sha256_file(self.protocol_path),
            "mode": self.mode, "synthetic": self.synthetic, "model_override": self.model_override,
            "started": self.started})
        if not skip_preflight and not (self.run_dir / "preflight.json").exists():
            self.preflight(train_smoke=train_smoke)
        self.calibrate()
        self.rollout()
        self.train()
        self.evaluate()
        out = self.compare()
        self.note("done", elapsed_s=round(time.time() - self.started))
        return out


def _count(items: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[str(i)] = out.get(str(i), 0) + 1
    return dict(sorted(out.items()))
