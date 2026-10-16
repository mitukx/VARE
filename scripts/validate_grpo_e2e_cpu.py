#!/usr/bin/env python3
"""Bounded real-model RVL integration check; synthetic rewards, no ability claim."""

from __future__ import annotations

import argparse
import asyncio
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RVL_REVISION = "e381c4b3b8c2ad8e5dc4c4b1a33b7922531c3e33"
MODEL_REVISION = "77f1b168e219585646439073245fe87e56b3023e"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def equal(a, b):
    import torch

    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def training_reward(text):
    return (sum(text.encode("utf-8")) % 13) / 12


def evaluation_score(text):
    # Separate deterministic diagnostic criterion, never an ability metric.
    return (int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 7) / 6


def reference_update(model, before, batch, config):
    """Independent full-graph equation, not a second usable training backend."""
    import torch

    model.load_state_dict(before["model"])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(copy.deepcopy(before["optimizer"]))
    optimizer.zero_grad(set_to_none=True)
    grouped = {}
    for exp in batch:
        grouped.setdefault(exp.attempt.metadata["vare_rollout_group"], []).append(exp)
    advantages = {}
    for group in grouped.values():
        values = [x.verification.score for x in group]
        mean = sum(values) / len(values)
        scale = math.sqrt(
            sum((x - mean) ** 2 for x in values) / len(values) + config.advantage_eps
        )
        for exp in group:
            advantages[exp.attempt.created_step] = max(
                -config.clip_advantage,
                min(config.clip_advantage, (exp.verification.score - mean) / scale),
            )
    losses = []
    alignment = []
    for exp in batch:
        meta = exp.attempt.metadata["rvl_generation"]["metadata"]
        prompt, response = meta["prompt_token_ids"], meta["response_token_ids"]
        logits = model(torch.tensor([prompt + response])).logits[
            0, len(prompt) - 1 : len(prompt) + len(response) - 1
        ]
        logps = torch.log_softmax(logits.float() / meta["sampling_temperature"], -1)
        selected = logps[torch.arange(len(response)), torch.tensor(response)]
        old = torch.tensor(meta["response_token_logprobs"])
        alignment.append(float((selected.detach() - old).abs().max()))
        ratio = torch.exp(torch.clamp(selected - old, -20, 20))
        adv = advantages[exp.attempt.created_step]
        losses.append(
            -torch.minimum(
                ratio * adv, ratio.clamp(1 - config.clip_eps, 1 + config.clip_eps) * adv
            ).mean()
        )
    loss = torch.stack(losses).mean()
    loss.backward()  # Contrast RVL's per-response accumulation with one backward.
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.max_grad_norm, error_if_nonfinite=True
    )
    grads = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "grads": grads,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "advantages": [advantages[x.attempt.created_step] for x in batch],
        "max_behavior_alignment_error": max(alignment),
    }


async def run(args):
    require(not args.output.exists(), "output already exists")
    # Reserve ownership atomically before preflight, so rejected model/source
    # inputs leave evidence without overwriting another run's output.
    args.output.mkdir(parents=True)
    args.output_created = True
    protocol_path = ROOT / "benchmarks/grpo_e2e_cpu_v4/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    require(
        sha(__file__) == protocol["runner_sha256"],
        "runner source differs from protocol",
    )
    for name, digest in protocol["model_files"].items():
        require(
            sha(args.model_dir / name) == digest, f"pinned model file mismatch: {name}"
        )
    require(
        subprocess.check_output(
            ["git", "-C", str(args.rvl_source), "rev-parse", "HEAD"], text=True
        ).strip()
        == RVL_REVISION,
        "RVL revision mismatch",
    )
    require(
        subprocess.check_output(
            ["git", "-C", str(args.rvl_source), "status", "--porcelain"], text=True
        ).strip()
        == "M src/rvl_systems/hf_backend.py"
        and sha(args.rvl_source / "src/rvl_systems/hf_backend.py")
        == protocol["rvl_patched_backend_sha256"]
        and sha(ROOT / "patches/rvl-contiguous-transition-logprobs-v1.patch")
        == protocol["rvl_patch_sha256"],
        "RVL checkout must contain exactly the pinned transition-score patch",
    )
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(args.rvl_source / "src"))
    import torch
    import transformers
    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from rvl_systems.types import Generation, VerifiedGeneration
    from vare.config import EngineConfig, LagConfig, PromotionConfig, ReplayConfig
    from vare.engine import CapabilityLoop
    from vare.integrations.rvl_grpo import RVLGRPOConfig, RVLGRPOHooks
    from vare.promotion import PromotionGate
    from vare.types import Task
    from vare.telemetry import EventLog
    from vare.verifiers import (
        FunctionalAttemptVerifier,
        VerifierEnsemble,
        VerifierMember,
    )

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    require(not torch.cuda.is_available(), "CPU run required")
    (args.output / "protocol.json").write_bytes(protocol_path.read_bytes())
    for path in (
        Path(__file__),
        ROOT / "src/vare/integrations/rvl_grpo.py",
        args.rvl_source / "src/rvl_systems/hf_trainer.py",
        args.rvl_source / "src/rvl_systems/hf_backend.py",
    ):
        (args.output / (path.stem + ".source.py")).write_bytes(path.read_bytes())
    backend = HFLocalBackend(
        str(args.model_dir), max_new_tokens=3, device="cpu", precision="fp32"
    )
    model = backend.model
    # Freeze the backbone to bound optimizer and numerical-reference memory.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("transformer.wpe.") or name.startswith("transformer.ln_f.")
        )
    config = HFTTrainerConfig(learning_rate=1e-4)
    trainer = HFCausalLMGRPOTrainer(model, config=config)
    optimizer_calls = 0
    raw_optimizer_step = trainer.optimizer.step

    def counted_optimizer_step(*a, **kw):
        nonlocal optimizer_calls
        optimizer_calls += 1
        return raw_optimizer_step(*a, **kw)

    trainer.optimizer.step = counted_optimizer_step
    reference_model = copy.deepcopy(model)
    eval_tasks = [
        Task(
            f"grpo-independent-{args.seed}-{i}",
            f"Independent GRPO diagnostic {args.seed}, row {i}: Continue the sentence: A quiet river",
            "independent",
        )
        for i in range(4)
    ]
    hooks = RVLGRPOHooks(
        backend=backend,
        trainer=trainer,
        eval_tasks=eval_tasks,
        score_fn=lambda task, text: evaluation_score(text),
        config=RVLGRPOConfig(seed=args.seed, train_temperature=0.8),
        current_verifier_version=0,
        freshness_lag_config=LagConfig(max_policy_lag=0, max_verifier_lag=0),
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )
    stage = {
        key: []
        for key in (
            "rollout",
            "verification",
            "validation",
            "training",
            "evaluation",
            "round",
            "freshness_prune",
        )
    }
    records, evaluations, rollout_records, fault_records = [], [], [], []
    replay_records, negative_controls, prune_records = [], [], []
    original_rollout = hooks.rollout
    original_train = hooks.train_candidate_with_freshness
    original_evaluate = hooks.evaluate
    original_step = trainer.train_step
    expected_active_hash = tensor_hash(model.state_dict())
    worker_acknowledgments = []

    def save_policy(policy_id):
        snapshot = hooks._states[policy_id].training_state
        path = args.output / (policy_id + ".pt")
        torch.save(snapshot, path)
        require(
            equal(snapshot, torch.load(path, weights_only=True)),
            "checkpoint round trip mismatch",
        )
        return snapshot

    save_policy("policy-0")

    async def rollout(*values):
        started = time.perf_counter()
        row = await original_rollout(*values)
        digest = tensor_hash(model.state_dict())
        require(
            digest == expected_active_hash, "rollout loaded wrong policy tensor state"
        )
        worker_acknowledgments.append(
            {
                "policy_id": row.policy_id,
                "policy_version": row.policy_version,
                "step": row.created_step,
                "model_sha256": digest,
            }
        )
        stage["rollout"].append(time.perf_counter() - started)
        rollout_records.append(asdict(row))
        return row

    hooks.rollout = rollout

    def verify(attempt):
        started = time.perf_counter()
        # Two complete groups of the same prompt: one has a null signal. The
        # last round is all-null, including with warm Adam moments.
        gid = attempt.metadata["vare_rollout_group"]
        value = (
            0.5
            if gid.endswith("g1") or current_round == args.rounds - 1
            else training_reward(attempt.output)
        )
        stage["verification"].append(time.perf_counter() - started)
        return value

    verifier = VerifierEnsemble(
        [VerifierMember(FunctionalAttemptVerifier(verify, version=0, trusted=True))]
    )
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=verifier,
        seed=args.seed,
        event_log=EventLog(args.output / "events.jsonl"),
        config=EngineConfig(
            samples_per_task=4,
            rollout_concurrency=4,
            replay_batch_size=8,
            lag=LagConfig(max_policy_lag=0, max_verifier_lag=0),
            replay=ReplayConfig(capacity=16),
            promotion=PromotionConfig(
                min_primary_gain=0, min_eval_examples=4, max_relative_cost_increase=1e6
            ),
        ),
    )
    original_sample = loop.replay.sample_current
    original_prune = loop.replay.prune_current

    def prune(*a, **kw):
        started = time.perf_counter()
        stored = len(loop.replay)
        removed = original_prune(*a, **kw)
        stage["freshness_prune"].append(time.perf_counter() - started)
        prune_records.append(
            {"round": current_round, "stored_items": stored, "removed_items": removed}
        )
        return removed

    loop.replay.prune_current = prune

    def sample(*a, **kw):
        items = [item.experience for item in loop.replay._heap]
        active_version = hooks.active_policy()[1]
        stale = sum(x.attempt.policy_version != active_version for x in items)
        selected = original_sample(*a, **kw)
        require(
            all(x.attempt.policy_version == active_version for x in selected),
            "stale replay selected",
        )
        replay_records.append(
            {
                "round": current_round,
                "stored_items": len(items),
                "stale_items": stale,
                "selected_items": len(selected),
                "selected_policy_versions": [
                    x.attempt.policy_version for x in selected
                ],
            }
        )
        return selected

    loop.replay.sample_current = sample

    async def train(incumbent, batch, assessor):
        started = time.perf_counter()
        verified = [hooks._to_verified_generation(x) for x in batch]
        actual_adv = hooks._group_relative_advantages(
            batch, verified, eps=config.advantage_eps, clip=config.clip_advantage
        )
        stage["validation"].append(time.perf_counter() - started)
        before = trainer.snapshot_training_state()
        ref = reference_update(reference_model, before, batch, config)
        if current_round == 0:
            merged = copy.deepcopy(batch)
            for exp in merged:
                exp.attempt.metadata["vare_rollout_group"] = "wrong-prompt-only-group"
            wrong = reference_update(reference_model, before, merged, config)
            distance = max(
                float((wrong["grads"][n] - v).abs().max())
                for n, v in ref["grads"].items()
            )
            require(distance > 2e-6, "group negative control is uninformative")
            shifted = copy.deepcopy(batch)
            for exp in shifted:
                meta = exp.attempt.metadata["rvl_generation"]["metadata"]
                meta["response_token_logprobs"] = [
                    v + 0.4 for v in meta["response_token_logprobs"]
                ]
            wrong = reference_update(reference_model, before, shifted, config)
            require(
                wrong["max_behavior_alignment_error"] > 0.3,
                "logprob negative control is uninformative",
            )
            negative_controls.extend(
                [
                    {
                        "control": "merge_same_prompt_groups",
                        "gradient_distance": distance,
                    },
                    {
                        "control": "misaligned_behavior_scores",
                        "alignment_error": wrong["max_behavior_alignment_error"],
                    },
                ]
            )
        actual = {}
        optimizer_step = trainer.optimizer.step

        def capture(*a, **kw):
            actual["grads"] = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
            return optimizer_step(*a, **kw)

        def step(*a, **kw):
            actual["metrics"] = original_step(*a, **kw)
            return actual["metrics"]

        trainer.optimizer.step = capture
        trainer.train_step = step
        started = time.perf_counter()
        try:
            candidate = await original_train(incumbent, batch, assessor)
        finally:
            trainer.optimizer.step = optimizer_step
            trainer.train_step = original_step
        elapsed = time.perf_counter() - started
        stage["training"].append(elapsed)
        require(
            equal(before, trainer.snapshot_training_state()),
            "incumbent model/optimizer/RNG rollback mismatch",
        )
        candidate_state = save_policy(candidate)
        torch.save(
            {
                "actual_clipped_grads": actual["grads"],
                "reference_clipped_grads": ref["grads"],
                "actual_trainable_parameters": {
                    n: candidate_state["model"][n] for n in ref["grads"]
                },
                "reference_trainable_parameters": {
                    n: ref["model"][n] for n in ref["grads"]
                },
            },
            args.output / (candidate + "-numerics.pt"),
        )
        require(
            torch.equal(before["cpu_rng"], candidate_state["cpu_rng"]),
            "deterministic GRPO changed CPU RNG",
        )
        require(
            candidate_state["optimizer"]["param_groups"]
            == ref["optimizer"]["param_groups"],
            "optimizer configuration mismatch",
        )
        require(
            candidate_state["optimizer"]["state"].keys()
            == ref["optimizer"]["state"].keys(),
            "optimizer state cardinality mismatch",
        )
        gradient_error = max(
            float((actual["grads"][n] - v).abs().max()) for n, v in ref["grads"].items()
        )
        parameter_error = max(
            float((candidate_state["model"][n] - v).abs().max())
            for n, v in ref["model"].items()
        )
        loss_error = abs(actual["metrics"]["loss"] - ref["loss"])
        require(
            loss_error < 2e-6 and gradient_error < 2e-6 and parameter_error < 2e-6,
            "reference loss/gradient/parameter mismatch",
        )
        # Optimizer values can differ by fp32 accumulation rounding; measure all
        # tensors as well as exact step/counter/config correspondence.
        optimizer_error = 0.0
        for key, state in candidate_state["optimizer"]["state"].items():
            for name, value in state.items():
                expected = ref["optimizer"]["state"][key][name]
                if isinstance(value, torch.Tensor):
                    optimizer_error = max(
                        optimizer_error, float((value - expected).abs().max())
                    )
                else:
                    require(value == expected, "optimizer scalar mismatch")
        require(optimizer_error < 2e-6, "optimizer reference mismatch")
        require(
            max(abs(a - b) for a, b in zip(actual_adv, ref["advantages"])) < 1e-12,
            "group advantage mismatch",
        )
        require(
            ref["max_behavior_alignment_error"]
            < protocol["numeric_limits"]["behavior_alignment_abs"],
            "token/logprob behavior alignment failed",
        )
        changed = sum(
            not torch.equal(v, before["model"][n])
            for n, v in candidate_state["model"].items()
        )
        records.append(
            {
                "round": current_round,
                "candidate": candidate,
                "batch": [asdict(x) for x in batch],
                "advantages": actual_adv,
                "metrics": actual["metrics"],
                "loss_error": loss_error,
                "gradient_error": gradient_error,
                "parameter_error": parameter_error,
                "optimizer_error": optimizer_error,
                "behavior_alignment_error": ref["max_behavior_alignment_error"],
                "changed_state_tensors": changed,
                "candidate_model_sha256": tensor_hash(candidate_state["model"]),
                "training_s": elapsed,
            }
        )
        return candidate

    hooks.train_candidate_with_freshness = train

    async def evaluate(policy_id):
        started = time.perf_counter()
        report = await original_evaluate(policy_id)
        corrupt = copy.deepcopy(report)
        corrupt.metadata["per_task_scores"].pop(
            next(iter(corrupt.metadata["per_task_scores"]))
        )
        require(
            not PromotionGate(loop.config.promotion).decide(report, corrupt).accepted,
            "missing evaluation row accepted",
        )
        stage["evaluation"].append(time.perf_counter() - started)
        # Independent checkpoint consumer: no adapter state, no verifier, no
        # optimizer, and no RVL generation implementation.
        saved = torch.load(args.output / (policy_id + ".pt"), weights_only=True)
        reference_model.load_state_dict(saved["model"])
        reference_model.eval()
        scores = {}
        with torch.no_grad():
            for task in eval_tasks:
                ids = backend.tokenizer.encode(task.prompt)
                output = []
                for _ in range(3):
                    token = int(
                        reference_model(torch.tensor([ids + output]))
                        .logits[0, -1]
                        .argmax()
                    )
                    output.append(token)
                    if token == backend.tokenizer.eos_token_id:
                        break
                scores[task.id] = evaluation_score(
                    backend.tokenizer.decode(output, skip_special_tokens=True)
                )
        require(
            scores == report.metadata["per_task_scores"],
            "independent checkpoint evaluation mismatch",
        )
        evaluations.append(asdict(report))
        return report

    hooks.evaluate = evaluate
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    current_round = 0
    try:
        for current_round in range(args.rounds):
            # Deliberately exercise rejection every fourth round. Other rounds
            # allow a diagnostic tie; this is not an improvement acceptance rule.
            loop.promotion = PromotionGate(
                replace(
                    loop.config.promotion,
                    min_primary_gain=1.1 if current_round % 4 == 3 else 0.0,
                )
            )
            task = Task(
                f"grpo-new-{args.seed}-{current_round}",
                f"Fresh GRPO systems row {args.seed}-{current_round}: A small observatory records",
                "systems",
            )
            started = time.perf_counter()
            previous_updates = len(records)
            result = await loop.run_round(
                [task], round_index=current_round, rollout_count=8
            )
            stage["round"].append(time.perf_counter() - started)
            require(
                len(records) == previous_updates + 1
                and records[-1]["round"] == current_round,
                "round completed without a GRPO optimizer update",
            )
            records[-1]["decision"] = asdict(result.decision)
            records[-1]["active_policy"] = hooks.active_policy()
            require(time.perf_counter() - wall_start < 600, "600-second campaign limit")
            require(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024)
                < 8 * 1024**3,
                "8-GiB peak RSS limit",
            )
            expected_active_hash = tensor_hash(model.state_dict())
            require(
                expected_active_hash
                == tensor_hash(
                    hooks._states[hooks.active_policy()[0]].training_state["model"]
                ),
                "active tensor state differs from checkpoint",
            )
            if current_round % 4 == 3:
                require(
                    not result.decision.accepted, "negative promotion control accepted"
                )
            write(args.output / "progress.json", {"completed_rounds": len(records)})
        measured_wall, measured_cpu = (
            time.perf_counter() - wall_start,
            time.process_time() - cpu_start,
        )
        # Faults use a newly generated complete group from the current policy,
        # never one of the historical frozen/confirmation cohorts.
        active, version = hooks.active_policy()
        task = Task(
            f"fault-{args.seed}",
            f"GRPO integrity canary {args.seed}: The copper clock",
            metadata={"vare_rollout_group": "fault", "vare_rollout_group_size": 4},
        )
        from vare.types import Experience, Verification

        before = trainer.snapshot_training_state()
        try:
            await hooks.rollout(task, active, version + 1, 9999)
        except ValueError as exc:
            fault_records.append(
                {
                    "fault": "rollout_policy_version",
                    "rejected": True,
                    "error": str(exc),
                    "state_unchanged": equal(before, trainer.snapshot_training_state()),
                }
            )
        else:
            raise RuntimeError("wrong rollout policy version accepted")

        fresh = []
        for i in range(4):
            attempt = await hooks.rollout(task, active, version, 10000 + i)
            attempt.metadata.update(task.metadata)
            fresh.append(
                Experience(
                    attempt,
                    Verification(
                        training_reward(attempt.output),
                        True,
                        1.0,
                        0,
                        "canary",
                        trusted=True,
                    ),
                    0,
                    0,
                    0.0,
                )
            )
        write(args.output / "fault-inputs.json", [asdict(x) for x in fresh])
        for fault in (
            "response_binding",
            "partial_group",
            "mixed_policy",
            "stale",
            "logprob_length",
            "nonfinite_logprob",
            "after_optimizer",
        ):
            bad = copy.deepcopy(fresh)
            if fault == "response_binding":
                bad[0].attempt.output += " DIFFERENT VERIFIED RESPONSE"
            if fault == "partial_group":
                bad.pop()
            if fault == "mixed_policy":
                bad[0].attempt.policy_version += 1
            if fault == "stale":
                for item in bad:
                    item.attempt.policy_version = max(0, version - 1)
            if fault == "logprob_length":
                bad[0].attempt.metadata["rvl_generation"]["metadata"][
                    "response_token_logprobs"
                ].pop()
            if fault == "nonfinite_logprob":
                bad[0].attempt.metadata["rvl_generation"]["metadata"][
                    "response_token_logprobs"
                ][0] = float("nan")
            before = trainer.snapshot_training_state()
            if fault == "after_optimizer":

                def fail_after(*a, **kw):
                    original_step(*a, **kw)
                    raise RuntimeError("injected after actual optimizer")

                trainer.train_step = fail_after
            error = None
            try:
                accepted_candidate = await original_train(
                    active, bad, hooks._training_freshness_assessor
                )
                torch.save(
                    hooks._states[accepted_candidate].training_state,
                    args.output / ("fault-accepted-" + fault + ".pt"),
                )
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                error = str(exc)
            finally:
                trainer.train_step = original_step
            fault_records.append(
                {
                    "fault": fault,
                    "rejected": error is not None,
                    "error": error,
                    "state_unchanged": equal(before, trainer.snapshot_training_state()),
                }
            )
            write(args.output / "faults.json", fault_records)
            require(
                error is not None and fault_records[-1]["state_unchanged"],
                f"fault accepted or mutated incumbent: {fault}",
            )
        # Persisted GRPO optimizer/RNG resume and the same next update. This is
        # trainer state consistency, not a reimplementation of #69 SIGKILL.
        saved = args.output / "resume.pt"
        torch.save(trainer.snapshot_training_state(), saved)
        baseline_candidate = await original_train(
            active, fresh, hooks._training_freshness_assessor
        )
        expected = copy.deepcopy(hooks._states[baseline_candidate].training_state)
        await hooks.discard(baseline_candidate)
        trainer.restore_training_state(torch.load(saved, weights_only=True))
        resumed_candidate = await original_train(
            active, fresh, hooks._training_freshness_assessor
        )
        require(
            equal(expected, hooks._states[resumed_candidate].training_state),
            "resumed next GRPO update differs",
        )
        await hooks.discard(resumed_candidate)
        status = "PASS"
    except BaseException:
        write(args.output / "rounds.json", records)
        write(args.output / "rollouts.json", rollout_records)
        write(args.output / "evaluations.json", evaluations)
        raise
    write(args.output / "rounds.json", records)
    write(args.output / "rollouts.json", rollout_records)
    write(args.output / "evaluations.json", evaluations)
    write(args.output / "worker-acknowledgments.json", worker_acknowledgments)
    write(args.output / "replay.json", replay_records)
    write(args.output / "replay-pruning.json", prune_records)
    stats = {
        key: {
            "n": len(values),
            "total_s": sum(values),
            "p50_s": sorted(values)[len(values) // 2],
            "p95_s": sorted(values)[math.ceil(len(values) * 0.95) - 1],
        }
        for key, values in stage.items()
    }
    # Full correctness wall includes reference/audit/checkpoint work. An isolated
    # operational cost is the sum of measured VARE stages; not a speed baseline.
    summary = {
        "status": status,
        "protocol_sha256": sha(protocol_path),
        "protocol_conformant": args.rounds == protocol["rounds"]
        and args.seed in protocol["seeds"],
        "seed": args.seed,
        "rounds": args.rounds,
        "candidate_updates": len(records),
        "actual_optimizer_calls_including_fault_and_resume": optimizer_calls,
        "model": "roneneldan/TinyStories-1M",
        "model_revision": MODEL_REVISION,
        "rvl_revision": RVL_REVISION,
        "vare_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {p.name: sha(p) for p in args.output.glob("*.source.py")},
        "model_files_sha256": {
            p.name: sha(p) for p in args.model_dir.iterdir() if p.is_file()
        },
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "wall_s_with_reference_checkpoint_audits": measured_wall,
        "cpu_s": measured_cpu,
        "cpu_core_percent": 100 * measured_cpu / measured_wall,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "throughput_tasks_s_including_audits": 8 * args.rounds / measured_wall,
        "throughput_tokens_s_including_audits": sum(
            x["metadata"]["token_count"] for x in rollout_records[: 8 * args.rounds]
        )
        / measured_wall,
        "stages": stats,
        "promotions": sum(x["decision"]["accepted"] for x in records),
        "rejections": sum(not x["decision"]["accepted"] for x in records),
        "faults": fault_records,
        "resume_next_update_exact": True,
        "negative_controls": negative_controls,
        "replay_pruning_rejection_rate": sum(x["removed_items"] for x in prune_records)
        / sum(x["stored_items"] for x in prune_records),
        "dependencies": sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in __import__(
                "importlib.metadata", fromlist=["distributions"]
            ).distributions()
        ),
        "stale_replay_events": [
            json.loads(line)
            for line in (args.output / "events.jsonl").read_text().splitlines()
            if json.loads(line)["event"] == "replay_experiences_screened"
        ],
        "limitations": [
            "Pretrained toy model, fresh synthetic diagnostic tasks; no capability gain or RSI.",
            "Single shared CPU model under adapter lock; no separate rollout worker pool.",
            "Reference/audit/save overhead included in headline throughput; timings are not production serving rates.",
            "No process crash injection; #69 remains separate unmerged recovery evidence.",
        ],
    }
    write(args.output / "summary.json", summary)
    manifest = {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    write(args.output / "SHA256.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rvl-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=64031)
    parser.add_argument("--rounds", type=int, choices=range(4, 21), default=12)
    args = parser.parse_args()
    args.output_created = False

    def timeout(*_):
        raise TimeoutError("1200-second hard wall cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1200)
    try:
        print(json.dumps(asyncio.run(run(args)), indent=2))
    except BaseException as exc:
        if args.output_created:
            write(
                args.output / "failure.json",
                {"type": type(exc).__name__, "error": str(exc)},
            )
        raise


if __name__ == "__main__":
    main()
