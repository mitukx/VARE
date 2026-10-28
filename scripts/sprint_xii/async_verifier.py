#!/usr/bin/env python3
"""Bounded synthetic verifier-transition experiment; not a capability benchmark.

Reuses VARE replay/lag admission and the pinned RVL GRPO trainer. The frozen
Qwen model carries the same vocabulary-bias adapter used in Sprint XI. Only
A/B/C/D are enabled, consistently in actor sampling and trainer likelihoods.
"""

from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":")).encode()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def dump(path, obj):
    write(path, json.dumps(obj, sort_keys=True, indent=2).encode() + b"\n")


def load(path):
    return json.loads(Path(path).read_text())


def journal(path, event):
    rows = (
        [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    )
    row = {"seq": len(rows), "previous": rows[-1]["sha"] if rows else "0" * 64, **event}
    row["sha"] = sha(canonical(row))
    with path.open("ab") as f:
        f.write(canonical(row) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def reward(token, version):
    # Explicit objective transition, not an erroneous real-world oracle.
    return float(token in ("AB" if version == 0 else "CD"))


def backend(p):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    tok = AutoTokenizer.from_pretrained(
        p["model"]["local_snapshot"], local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        p["model"]["local_snapshot"],
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    for par in model.parameters():
        par.requires_grad_(False)
    model.register_parameter(
        "vare_action_bias", torch.nn.Parameter(torch.zeros(model.config.vocab_size))
    )
    ids = [tok.encode(x, add_special_tokens=False) for x in "ABCD"]
    assert all(len(x) == 1 for x in ids)
    ids = [x[0] for x in ids]
    mask = torch.full((model.config.vocab_size,), -1e9)
    mask[ids] = 0
    model.lm_head.register_forward_hook(
        lambda _m, _x, y: y + model.vare_action_bias + mask
    )
    model.eval()
    return torch, tok, model, ids


def prompt(tok, index, evaluation=False):
    kind = "evaluation" if evaluation else "routing"
    return tok.apply_chat_template(
        [
            {"role": "system", "content": "Follow the output format."},
            {
                "role": "user",
                "content": f"Synthetic {kind} case {index}. Select one route label from A, B, C, D. Reply with one label.",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def wait_for(path, deadline):
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(0.02)


def actor(p, root, seed, arm):
    torch, tok, model, ids = backend(p)
    deadline = time.monotonic() + p["limits"]["wall_seconds"]
    for i in range(p["groups"]):
        if arm == "A":
            wait_for(root / f"request-{i}", deadline)
        path = root / f"group-{i}.json"
        if not path.exists():
            created = time.time()
            text = prompt(tok, i)
            inputs = tok.encode(text, add_special_tokens=False)
            with torch.inference_mode():
                logits = model(torch.tensor([inputs])).logits[0, -1, ids].float()
            probs = torch.softmax(logits / p["temperature"], dim=-1)
            rng = torch.Generator().manual_seed(seed * 100 + i)
            choices = torch.multinomial(
                probs, p["group_size"], replacement=True, generator=rng
            )
            version = 0 if i == 0 or (arm != "A" and i in (1, 2)) else 1
            rows = [
                {
                    "response": "ABCD"[int(x)],
                    "token": ids[int(x)],
                    "behavior_logprob": float(torch.log(probs[x])),
                }
                for x in choices
            ]
            group = {
                "index": i,
                "seed": seed,
                "group_id": f"xii-{seed}-{i}",
                "prompt": text,
                "prompt_ids": inputs,
                "action_ids": ids,
                "base_logits": logits.tolist(),
                "generations": rows,
                "rewards": [reward(x["response"], version) for x in rows],
                "verifier_version": version,
                "verifier_sha": sha(canonical(p["verifiers"][str(version)])),
                "policy_version": 0,
                "policy_identity": p["initial_policy_identity"],
                "created_unix": created,
                "worker_pid": os.getpid(),
            }
            dump(path, group)
            dump(root / f"group-{i}.sha.json", {"sha256": sha(path.read_bytes())})
            if arm == "E" and i == 1 and not (root / "worker-fault").exists():
                dump(
                    root / "worker-fault",
                    {"pid": os.getpid(), "group_sha": sha(path.read_bytes())},
                )
                os.kill(os.getpid(), signal.SIGKILL)
        if i in (1, 2) and arm != "A":
            wait_for(root / "verifier-v1", deadline)
            time.sleep(p["stale_delay_seconds"])
        dump(
            root / f"arrival-{i}",
            {
                "unix": time.time(),
                "pid": os.getpid(),
                "artifact_sha": sha(path.read_bytes()),
            },
        )
    dump(root / "actor-done.json", {"pid": os.getpid()})


def checkpoint(path, model, trainer, torch, metadata):
    started = time.monotonic()
    state = {
        "bias": model.vare_action_bias.detach().clone(),
        "optimizer": trainer.optimizer.state_dict(),
        "rng": torch.get_rng_state(),
    }
    buf = io.BytesIO()
    torch.save(state, buf)
    payload = gzip.compress(buf.getvalue(), mtime=0)
    write(path, payload)
    manifest = {
        "sha256": sha(payload),
        "metadata": metadata,
        "bytes": len(payload),
        "io_seconds": time.monotonic() - started,
    }
    dump(Path(str(path) + ".json"), manifest)
    return manifest


def restore(path, model, trainer, torch):
    meta = load(str(path) + ".json")
    raw = path.read_bytes()
    assert sha(raw) == meta["sha256"]
    state = torch.load(
        io.BytesIO(gzip.decompress(raw)), weights_only=True, map_location="cpu"
    )
    with torch.no_grad():
        model.vare_action_bias.copy_(state["bias"])
    trainer.optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"])
    return meta


def samples(p, g, rewards, version):
    from vare.types import Task, Attempt, Verification, Experience
    from rvl_systems.types import Generation, VerifiedGeneration

    exps = []
    verified = []
    for j, row in enumerate(g["generations"]):
        metadata = {
            "vare_rollout_group": g["group_id"],
            "vare_rollout_group_size": len(rewards),
            "sample_index": j,
            "policy_checkpoint_sha256": g["policy_identity"],
        }
        task = Task(
            id=g["group_id"],
            prompt=g["prompt"],
            family="synthetic-routing",
            metadata=metadata,
        )
        attempt = Attempt(
            task=task,
            output=row["response"],
            policy_id="initial",
            policy_version=0,
            created_step=0,
            logprob=row["behavior_logprob"],
            metadata=metadata,
        )
        verification = Verification(
            score=rewards[j],
            passed=bool(rewards[j]),
            confidence=1.0,
            verifier_name="routing-transition",
            verifier_version=version,
            trusted=True,
            metadata={
                "rvl_verification_config_sha256": sha(
                    canonical(p["verifiers"][str(version)])
                )
            },
        )
        exps.append(Experience(attempt, verification, 0, 0, 0.0))
        generation = Generation(
            g["group_id"],
            g["prompt"],
            row["response"],
            row["behavior_logprob"],
            1,
            0.0,
            {
                "prompt_token_ids": g["prompt_ids"],
                "response_token_ids": [row["token"]],
                "response_token_logprobs": [row["behavior_logprob"]],
                "sampling_temperature": p["temperature"],
            },
        )
        metadata["rvl_generation"] = {
            "prompt_id": g["group_id"],
            "prompt": g["prompt"],
            "metadata": generation.metadata,
        }
        verified.append(VerifiedGeneration(generation, rewards[j], 0.0, version))
    return exps, verified


def learn(p, root, seed, arm):
    import numpy as np
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from vare.replay import PrioritizedReplay
    from vare.config import ReplayConfig, LagConfig
    from vare.lag import LagController, make_experience_freshness_assessor
    from vare.integrations.rvl_grpo import RVLGRPOHooks

    torch, tok, model, ids = backend(p)
    trainer = HFCausalLMGRPOTrainer(
        model, config=HFTTrainerConfig(learning_rate=p["learning_rate"])
    )
    torch.manual_seed(seed)
    jpath = root / "learner.jsonl"
    journal(jpath, {"kind": "start", "pid": os.getpid(), "unix": time.time()})
    version = 0
    if (root / "active.json").exists():
        active = load(root / "active.json")
        restore(root / active["path"], model, trainer, torch)
        version = active["policy_version"]
    deadline = time.monotonic() + p["limits"]["wall_seconds"]
    for i in range(p["groups"]):
        if arm == "A":
            dump(root / f"request-{i}", {"unix": time.time()})
        if (root / f"done-{i}.json").exists():
            if i == 0:
                dump(root / "verifier-v1", {"unix": time.time()})
            continue
        wait_for(root / f"arrival-{i}", deadline)
        path = root / f"group-{i}.json"
        g = load(path)
        assert sha(path.read_bytes()) == load(root / f"group-{i}.sha.json")["sha256"]
        assert (
            g["policy_identity"] == p["initial_policy_identity"]
            and g["policy_version"] == 0
        )
        active_verifier = 0 if i == 0 else 1
        stale = g["verifier_version"] != active_verifier
        regrade = stale and arm in ("D", "E")
        rewards = (
            [reward(x["response"], active_verifier) for x in g["generations"]]
            if regrade
            else g["rewards"]
        )
        scored_version = active_verifier if regrade else g["verifier_version"]
        exps, verified = samples(p, g, rewards, scored_version)
        replay = PrioritizedReplay(ReplayConfig(capacity=64), seed=seed)
        assert replay.add_group(exps, [1.0] * len(exps))
        assessor = make_experience_freshness_assessor(
            LagController(
                LagConfig(max_policy_lag=5, max_verifier_lag=0, max_shift=0.0)
            ),
            active_policy_version=version,
            active_verifier_version=active_verifier,
        )
        eligible = replay.sample_current(
            len(exps),
            grouped=True,
            freshness_fn=(lambda x: 1.0)
            if arm == "B"
            else (lambda x: 1.0 if assessor(x).admitted else None),
        )
        cp = root / f"checkpoint-{i}.pt.gz"
        if eligible:
            assert len(eligible) == p["group_size"]
            if len(set(rewards)) < 2:
                raise RuntimeError("NO_GO_constant_group")
            if not cp.exists():
                before = model.vare_action_bias.detach().clone()
                advantages = RVLGRPOHooks._group_relative_advantages(
                    exps, verified, eps=1e-6, clip=5.0
                )
                journal(
                    jpath,
                    {
                        "kind": "prepare",
                        "group": i,
                        "pid": os.getpid(),
                        "rewards": rewards,
                        "stale": stale,
                        "regraded": regrade,
                    },
                )
                metrics = trainer.train_step(verified, advantages=advantages)
                grad = model.vare_action_bias.grad.detach().clone()
                journal(
                    jpath,
                    {
                        "kind": "optimizer_applied",
                        "group": i,
                        "pid": os.getpid(),
                        "metrics": metrics,
                        "gradient4": grad[ids].tolist(),
                        "bias_before4": before[ids].tolist(),
                        "bias_after4": model.vare_action_bias.detach()[ids].tolist(),
                    },
                )
                if arm == "E" and i == 2 and not (root / "fault-optimizer").exists():
                    dump(root / "fault-optimizer", {"pid": os.getpid()})
                    os.kill(os.getpid(), signal.SIGKILL)
                version += 1
                checkpoint(
                    cp,
                    model,
                    trainer,
                    torch,
                    {
                        "group": i,
                        "policy_version": version,
                        "verifier_version": active_verifier,
                        "behavior_identity": g["policy_identity"],
                    },
                )
            else:
                meta = restore(cp, model, trainer, torch)
                version = meta["metadata"]["policy_version"]
            if (
                arm == "E"
                and i == 3
                and not (root / "fault-before-publication").exists()
            ):
                dump(root / "fault-before-publication", {"pid": os.getpid()})
                os.kill(os.getpid(), signal.SIGKILL)
            dump(
                root / "active.json",
                {
                    "path": cp.name,
                    "sha256": load(str(cp) + ".json")["sha256"],
                    "policy_version": version,
                },
            )
            rows = [json.loads(x) for x in jpath.read_text().splitlines()]
            if not any(x["kind"] == "commit" and x["group"] == i for x in rows):
                journal(
                    jpath,
                    {
                        "kind": "commit",
                        "group": i,
                        "pid": os.getpid(),
                        "checkpoint_sha": load(str(cp) + ".json")["sha256"],
                    },
                )
            if (
                arm == "E"
                and i == 4
                and not (root / "fault-after-publication").exists()
            ):
                dump(root / "fault-after-publication", {"pid": os.getpid()})
                os.kill(os.getpid(), signal.SIGKILL)
        receipt = {
            "group": i,
            "accepted": bool(eligible),
            "stale": stale,
            "regraded": regrade,
            "rewards": rewards,
            "current_rewards": [
                reward(x["response"], active_verifier) for x in g["generations"]
            ],
            "verifier_version": scored_version,
            "policy_version": version,
            "queue_inclusive_seconds": time.time() - g["created_unix"],
            "verifier_calls": len(rewards) * (1 + int(regrade)),
        }
        dump(root / f"done-{i}.json", receipt)
        if i == 0:
            dump(
                root / "verifier-v1",
                {
                    "unix": time.time(),
                    "verifier_sha": sha(canonical(p["verifiers"]["1"])),
                },
            )
    model.eval()
    evaluations = []
    for i in range(3):
        inp = tok.encode(prompt(tok, 100 + i, True), add_special_tokens=False)
        with torch.inference_mode():
            probs = torch.softmax(model(torch.tensor([inp])).logits[0, -1, ids], dim=-1)
        evaluations.append(
            {
                "prompt_ids": inp,
                "probabilities": probs.tolist(),
                "v1_expected_reward": float(probs[2:].sum()),
            }
        )
    dump(
        root / "evaluation.json",
        {
            "tasks": evaluations,
            "mean_v1_reward": float(
                np.mean([x["v1_expected_reward"] for x in evaluations])
            ),
            "pid": os.getpid(),
        },
    )
    dump(root / "learner-done.json", {"pid": os.getpid(), "policy_version": version})


def campaign(p, output):
    import psutil

    if output.exists():
        raise FileExistsError("append-only output exists")
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    disk = psutil.disk_usage(str(output.parent))
    assert (
        vm.available >= p["limits"]["min_available"]
        and sw.free >= p["limits"]["min_free_swap"]
        and disk.free >= p["limits"]["min_disk"]
    )
    for path, digest in p["hashes"].items():
        assert sha(Path(path).read_bytes()) == digest, path
    output.mkdir()
    dump(
        output / "preflight.json",
        {
            "ram_available": vm.available,
            "swap_used": sw.used,
            "swap_free": sw.free,
            "disk_free": disk.free,
            "pid": os.getpid(),
        },
    )
    started = time.monotonic()
    summary = []
    for seed in p["seeds"]:
        for arm in p["arm_order"]:
            root = output / f"seed-{seed}" / arm
            root.mkdir(parents=True)
            procs = {}
            handles = []
            records = []
            attempts = {"actor": 0, "learner": 0}
            case_start = time.monotonic()
            peak = 0
            minram = vm.available
            maxswap = 0

            def launch(mode):
                attempts[mode] += 1
                stream = (root / f"{mode}-attempt-{attempts[mode]}.log").open("w")
                handles.append(stream)
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        mode,
                        "--protocol",
                        str(P_PATH),
                        "--output",
                        str(root),
                        "--seed",
                        str(seed),
                        "--arm",
                        arm,
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                procs[mode] = proc
                records.append(
                    {
                        "mode": mode,
                        "attempt": attempts[mode],
                        "pid": proc.pid,
                        "launched_unix": time.time(),
                    }
                )

            launch("actor")
            launch("learner")
            try:
                while procs:
                    rss = 0
                    for mode, proc in list(procs.items()):
                        if proc.poll() is not None:
                            next(x for x in records if x["pid"] == proc.pid)[
                                "returncode"
                            ] = proc.returncode
                            del procs[mode]
                            if (
                                proc.returncode == -signal.SIGKILL
                                and arm == "E"
                                and attempts[mode] < 5
                            ):
                                launch(mode)
                            elif proc.returncode != 0:
                                raise RuntimeError(
                                    f"{arm}/{mode} exited {proc.returncode}"
                                )
                        else:
                            try:
                                rss += psutil.Process(proc.pid).memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                    peak = max(peak, rss)
                    minram = min(minram, psutil.virtual_memory().available)
                    maxswap = max(maxswap, psutil.swap_memory().used - sw.used)
                    if (
                        rss > p["limits"]["max_rss"]
                        or minram < p["limits"]["hard_min_available"]
                        or maxswap > p["limits"]["max_swap_growth"]
                        or psutil.disk_usage(str(output)).free < p["limits"]["min_disk"]
                        or time.monotonic() - started > p["limits"]["wall_seconds"]
                    ):
                        raise RuntimeError("RESOURCE_ABORT")
                    time.sleep(0.1)
            finally:
                for proc in procs.values():
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                for f in handles:
                    f.close()
                dump(root / "processes.json", records)
            receipts = [load(root / f"done-{i}.json") for i in range(p["groups"])]
            elapsed = time.monotonic() - case_start
            row = {
                "seed": seed,
                "arm": arm,
                "wall_seconds": elapsed,
                "peak_child_rss": peak,
                "minimum_host_available": minram,
                "swap_growth": maxswap,
                "effective_commits": sum(x["accepted"] for x in receipts),
                "contaminated_updates": sum(
                    x["accepted"]
                    and x["verifier_version"] != (0 if x["group"] == 0 else 1)
                    for x in receipts
                ),
                "generated_trajectories": p["groups"] * p["group_size"],
                "trajectories_per_second": p["groups"] * p["group_size"] / elapsed,
                "useful_trajectories_per_second": sum(x["accepted"] for x in receipts)
                * p["group_size"]
                / elapsed,
                "v1_reward": load(root / "evaluation.json")["mean_v1_reward"],
            }
            dump(root / "case-summary.json", row)
            summary.append(row)
            dump(output / "partial-summary.json", summary)
    dump(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "actor", "learner"])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm")
    a = parser.parse_args()
    P_PATH = a.protocol.resolve()
    p = load(P_PATH)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
    )
    sys.path[:0] = [str(Path(p["vare_root"]) / "src"), str(Path(p["rvl_root"]) / "src")]
    try:
        if a.mode == "run":
            campaign(p, a.output)
        elif a.mode == "actor":
            actor(p, a.output, a.seed, a.arm)
        else:
            learn(p, a.output, a.seed, a.arm)
    except BaseException as exc:
        if a.mode == "run" and not (a.output / "summary.json").exists():
            dump(
                a.output / "failure.json",
                {"type": type(exc).__name__, "message": str(exc), "unix": time.time()},
            )
        raise
