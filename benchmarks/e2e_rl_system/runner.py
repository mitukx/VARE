from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vare.config import EngineConfig, LagConfig, PromotionConfig, ReplayConfig  # noqa: E402
from vare.curriculum import AdaptiveCurriculum  # noqa: E402
from vare.demo import ToyHooks, make_tasks  # noqa: E402
from vare.engine import CapabilityLoop  # noqa: E402
from vare.failure import FailureMiner  # noqa: E402
from vare.lag import DistributionShiftMonitor  # noqa: E402
from vare.replay import PrioritizedReplay  # noqa: E402
from vare.types import Attempt, Experience, Task, Verification  # noqa: E402
from vare.verifiers import VerifierEnsemble, VerifierMember  # noqa: E402


PROTOCOL = "e2e-rl-system-benchmark-v1"
DEFAULT_SEEDS = tuple(range(20261001, 20261013))
DEFAULT_ROUNDS = 6
DEFAULT_ROLLOUTS = 64
DEFAULT_EVAL_N = 256
DEFAULT_REPLAY_BATCH = 32
DEFAULT_CONCURRENCY = 8


class TimedVerifier:
    name = "benchmark-oracle"
    trusted = True

    def __init__(self, version: int = 1) -> None:
        self.version = version
        self.seconds: list[float] = []
        self.max_verifier_lag = 0

    async def verify(self, attempt: Attempt) -> Verification:
        started = time.perf_counter()
        # Same deterministic oracle as ToyTrustedVerifier, expressed here so
        # benchmark fault modes can report a deliberately stale version.
        from vare.demo import _truth

        correct = int(attempt.output) == _truth(attempt.task)
        self.seconds.append(time.perf_counter() - started)
        version = 0 if getattr(self, "stale", False) else self.version
        self.max_verifier_lag = max(self.max_verifier_lag, self.version - version)
        return Verification(
            float(correct), correct, 1.0, version, self.name, trusted=True
        )


class MeasuredHooks(ToyHooks):
    """The repository's executable CPU policy with per-stage timing hooks."""

    def __init__(self, *, seed: int, eval_n: int) -> None:
        super().__init__(seed=seed, eval_n=eval_n)
        self.seconds: dict[str, list[float]] = defaultdict(list)
        self.max_policy_lag = 0
        self.max_consumed_policy_lag = 0
        self.max_consumed_verifier_lag = 0
        self.invalid_training_rows = 0
        self.invalid_update_acceptances = 0
        self.rollout_trace: list[tuple[str, str, int, int]] = []
        self.training_batches: list[list[tuple[str, int, int]]] = []

    async def rollout(
        self, task: Task, policy_id: str, policy_version: int, step: int
    ) -> Attempt:
        attempt = await self._timed(
            "rollout", super().rollout(task, policy_id, policy_version, step)
        )
        self.rollout_trace.append((task.id, policy_id, policy_version, step))
        self.max_policy_lag = max(
            self.max_policy_lag,
            max(0, self.active_policy()[1] - attempt.policy_version),
        )
        return attempt

    async def train_candidate(self, incumbent_id, experiences) -> str:
        self.training_batches.append([
            (
                exp.attempt.task.id,
                exp.attempt.policy_version,
                exp.verification.verifier_version,
            )
            for exp in experiences
        ])
        return await self._timed(
            "training", super().train_candidate(incumbent_id, experiences)
        )

    async def train_candidate_with_freshness(self, incumbent_id, experiences, assessor):
        for experience in experiences:
            decision = assessor(experience)
            if not decision.admitted:
                raise RuntimeError("engine passed stale replay data to training hook")
            self.max_consumed_policy_lag = max(
                self.max_consumed_policy_lag, decision.policy_lag
            )
            self.max_consumed_verifier_lag = max(
                self.max_consumed_verifier_lag, decision.verifier_lag
            )
        return await self.train_candidate(incumbent_id, experiences)

    async def evaluate(self, *args: Any, **kwargs: Any):
        return await self._timed("evaluation", super().evaluate(*args, **kwargs))

    async def promote(self, *args: Any, **kwargs: Any) -> None:
        await self._timed("promotion", super().promote(*args, **kwargs))

    async def discard(self, *args: Any, **kwargs: Any) -> None:
        await self._timed("rollback", super().discard(*args, **kwargs))

    async def _timed(self, name: str, awaitable):
        started = time.perf_counter()
        try:
            return await awaitable
        finally:
            self.seconds[name].append(time.perf_counter() - started)


def _make_config(*, paired_gate: bool = False) -> EngineConfig:
    return EngineConfig(
        rollout_concurrency=DEFAULT_CONCURRENCY,
        replay_batch_size=DEFAULT_REPLAY_BATCH,
        samples_per_task=1,
        preserve_rollout_groups=True,
        replay=ReplayConfig(capacity=4096),
        promotion=PromotionConfig(
            min_primary_gain=0.01,
            max_slice_regression=0.08,
            max_relative_cost_increase=0.20,
            max_verifier_disagreement=0.20,
            min_eval_examples=64,
            paired_confidence_gate=paired_gate,
            min_paired_examples=64,
            paired_bootstrap_samples=500,
        ),
    )


def _tasks(seed: int, count: int) -> list[Task]:
    # A fresh, pre-generated training cohort. Every row has the same family so
    # VARE's curriculum sampling and the reference path draw the same sequence.
    return [
        Task(t.id, t.prompt, "fixed-workload", t.metadata)
        for t in make_tasks(count, seed=seed + 1)
    ]


async def _baseline_round(
    *, hooks: MeasuredHooks, verifier: TimedVerifier,
    verifier_ensemble: VerifierEnsemble, tasks: list[Task], replay: PrioritizedReplay,
    curriculum: AdaptiveCurriculum, failure_miner: FailureMiner, round_index: int,
    config: EngineConfig, eval_tasks: list[Task],
) -> dict[str, Any]:
    """Straight-line reference path with the same dataplane primitives.

    It intentionally omits VARE's use-time verifier/policy checks, candidate
    transaction ownership and PromotionGate. It retains the same oracle,
    prioritized replay, rollout group size, trainer, evaluator and promotion
    point threshold, making this a narrow control-mechanism comparator.
    """
    hooks.eval_tasks = eval_tasks
    started = time.perf_counter()
    incumbent_id, incumbent_version = hooks.active_policy()
    chosen = curriculum.choose(tasks, DEFAULT_ROLLOUTS)
    semaphore = asyncio.Semaphore(config.rollout_concurrency)

    async def one(index: int, task: Task):
        async with semaphore:
            attempt = await hooks.rollout(
                task, incumbent_id, incumbent_version,
                round_index * DEFAULT_ROLLOUTS + index,
            )
            attempt.metadata["vare_rollout_group"] = f"r{round_index}-run{round_index + 1}-g{index}"
            attempt.metadata["vare_rollout_group_size"] = 1
            result = await verifier_ensemble.verify(attempt)
            return attempt, result

    pairs = await asyncio.gather(*(one(i, task) for i, task in enumerate(chosen)))
    family_shift = DistributionShiftMonitor()
    family_shift.fit_reference(t.family for t in tasks)
    experiences: list[Experience] = []
    for attempt, result in pairs:
        family_shift.observe(attempt.task.family)
        exp = Experience(
            attempt,
            result,
            max(0, incumbent_version - attempt.policy_version),
            max(0, verifier.version - result.verifier_version),
            family_shift.score(),
        )
        replay.add_group([exp], [1.0])
        experiences.append(exp)
    failures = failure_miner.mine(experiences)
    curriculum.update(failures)
    # Reference path consumes the same replay structure and batch size but
    # deliberately has no use-time freshness filter.
    batch = replay.sample_current(
        config.replay_batch_size, freshness_fn=lambda _experience: 1.0, grouped=True
    )
    if batch:
        hooks.max_consumed_policy_lag = max(
            hooks.max_consumed_policy_lag,
            max(max(0, incumbent_version - exp.attempt.policy_version) for exp in batch),
        )
        hooks.max_consumed_verifier_lag = max(
            hooks.max_consumed_verifier_lag,
            max(max(0, verifier.version - exp.verification.verifier_version) for exp in batch),
        )
    if hasattr(hooks, "invalid_training_rows"):
        invalid_rows = sum(
            incumbent_version - exp.attempt.policy_version > config.lag.max_policy_lag
            or verifier.version - exp.verification.verifier_version > config.lag.max_verifier_lag
            for exp in batch
        )
        hooks.invalid_training_rows += invalid_rows
        hooks.invalid_update_acceptances += int(bool(batch) and invalid_rows > 0)
    candidate_id = incumbent_id if not batch else await hooks.train_candidate(incumbent_id, batch)
    old_eval, candidate_eval = await asyncio.gather(
        hooks.evaluate(incumbent_id), hooks.evaluate(candidate_id)
    )
    # The threshold is frozen to the configured primary gain. No paired lower
    # bound, report identity validation, slice gate or fail-closed metrics audit.
    accept = candidate_id != incumbent_id and (
        candidate_eval.primary - old_eval.primary >= config.promotion.min_primary_gain
    )
    if accept:
        await hooks.promote(candidate_id)
        promoted_id = candidate_id
    else:
        if candidate_id != incumbent_id:
            await hooks.discard(candidate_id)
        promoted_id = incumbent_id
    elapsed = time.perf_counter() - started
    return {
        "round_index": round_index,
        "elapsed_s": elapsed,
        "incumbent_id": incumbent_id,
        "candidate_id": candidate_id,
        "promoted_id": promoted_id,
        "promoted": accept,
        "admitted": len(experiences),
        "dropped": 0,
        "policy_lag_max": hooks.max_policy_lag,
        "verifier_lag_max": verifier.max_verifier_lag,
        "consumed_policy_lag_max": hooks.max_consumed_policy_lag,
        "consumed_verifier_lag_max": hooks.max_consumed_verifier_lag,
        "invalid_training_rows_consumed": hooks.invalid_training_rows,
        "invalid_update_acceptances": hooks.invalid_update_acceptances,
        "replay_size": len(replay),
        "eval_task_set_sha256": _task_set_digest(hooks.eval_tasks),
    }


async def _run_arm(arm: str, seed: int, rounds: int, rollouts: int, eval_n: int) -> dict[str, Any]:
    if rollouts != DEFAULT_ROLLOUTS:
        raise ValueError("this locked protocol fixes rollout count at 64")
    hooks = MeasuredHooks(seed=seed, eval_n=eval_n)
    verifier = TimedVerifier()
    verifier_ensemble = VerifierEnsemble([VerifierMember(verifier)])
    config = _make_config()
    training_tasks = _tasks(seed, rollouts)
    evaluation_cohorts = [
        make_tasks(eval_n, seed=seed + 1_000_000 + round_index)
        for round_index in range(rounds)
    ]
    stage_trace: list[dict[str, Any]] = []
    results = []
    started_all = time.perf_counter()
    if arm == "vare":
        loop = CapabilityLoop(
            hooks=hooks, verifier=verifier_ensemble, config=config, seed=seed
        )
        for round_index in range(rounds):
            hooks.eval_tasks = evaluation_cohorts[round_index]
            started = time.perf_counter()
            result = await loop.run_round(
                training_tasks, round_index=round_index, rollout_count=rollouts
            )
            elapsed = time.perf_counter() - started
            row = {
                "round_index": round_index,
                "elapsed_s": elapsed,
                "incumbent_id": result.incumbent_id,
                "candidate_id": result.candidate_id,
                "promoted_id": result.promoted_id,
                "promoted": result.decision.accepted,
                "admitted": result.admitted_experiences,
                "dropped": result.dropped_experiences,
                "policy_lag_max": hooks.max_policy_lag,
                "verifier_lag_max": verifier.max_verifier_lag,
                "consumed_policy_lag_max": hooks.max_consumed_policy_lag,
                "consumed_verifier_lag_max": hooks.max_consumed_verifier_lag,
                "replay_size": len(loop.replay),
                "eval_task_set_sha256": _task_set_digest(hooks.eval_tasks),
            }
            results.append(row)
            stage_trace.append({"round_index": round_index, "decision": result.decision.accepted})
        replay_size = len(loop.replay)
    elif arm == "baseline":
        replay = PrioritizedReplay(config.replay, seed=seed)
        curriculum = AdaptiveCurriculum(seed=seed)
        failure_miner = FailureMiner()
        for round_index in range(rounds):
            row = await _baseline_round(
                hooks=hooks, verifier=verifier, verifier_ensemble=verifier_ensemble,
                tasks=training_tasks, replay=replay,
                curriculum=curriculum, failure_miner=failure_miner,
                round_index=round_index, config=config,
                eval_tasks=evaluation_cohorts[round_index],
            )
            results.append(row)
            stage_trace.append({"round_index": round_index, "decision": row["promoted"]})
        replay_size = len(replay)
    else:
        raise ValueError(f"unknown arm: {arm}")
    elapsed_all = time.perf_counter() - started_all
    hook_s = sum(sum(values) for values in hooks.seconds.values())
    verifier_s = sum(verifier.seconds)
    round_latencies = [r["elapsed_s"] for r in results]
    total_rollouts = rounds * rollouts
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(rss_raw if sys.platform == "darwin" else rss_raw * 1024)
    _, peak_python_bytes = tracemalloc.get_traced_memory()
    return {
        "protocol": PROTOCOL,
        "arm": arm,
        "seed": seed,
        "rounds": rounds,
        "rollout_count_per_round": rollouts,
        "eval_examples": eval_n,
        "round_results": results,
        "stage_seconds": {key: sum(values) for key, values in hooks.seconds.items()},
        "verifier_seconds": verifier_s,
        "rollout_seconds": sum(hooks.seconds["rollout"]),
        "training_seconds": sum(hooks.seconds["training"]),
        "evaluation_seconds": sum(hooks.seconds["evaluation"]),
        "promotion_rollback_seconds": sum(hooks.seconds["promotion"] + hooks.seconds["rollback"]),
        "orchestration_replay_residual_seconds": max(0.0, elapsed_all - hook_s - verifier_s),
        "elapsed_s": elapsed_all,
        "round_latency_p50_s": _quantile(round_latencies, 0.50),
        "round_latency_p95_s": _quantile(round_latencies, 0.95),
        "rollouts_per_second": total_rollouts / elapsed_all,
        "accepted_rollouts_per_second": sum(r["admitted"] for r in results) / elapsed_all,
        "policy_lag_max": hooks.max_policy_lag,
        "verifier_lag_max": verifier.max_verifier_lag,
        "consumed_policy_lag_max": hooks.max_consumed_policy_lag,
        "consumed_verifier_lag_max": hooks.max_consumed_verifier_lag,
        "invalid_training_rows_consumed": hooks.invalid_training_rows,
        "invalid_update_acceptances": hooks.invalid_update_acceptances,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_python_bytes": peak_python_bytes,
        "replay_final_size": replay_size,
        "stage_trace_sha256": hashlib.sha256(
            json.dumps(
                {
                    "rounds": stage_trace,
                    "training_batches": hooks.training_batches,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "rollout_trace_sha256": hashlib.sha256(
            json.dumps(hooks.rollout_trace, separators=(",", ":")).encode()
        ).hexdigest(),
        "rollout_task_ids_sha256": hashlib.sha256(
            json.dumps(
                [row[0] for row in hooks.rollout_trace], separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "active_policy": hooks.active_policy(),
    }


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _child_trial(arm: str, seed: int, rounds: int, rollouts: int, eval_n: int) -> dict[str, Any]:
    tracemalloc.start()
    return asyncio.run(_run_arm(arm, seed, rounds, rollouts, eval_n))


def _machine_info() -> dict[str, Any]:
    memory = None
    cpu_brand = None
    if sys.platform == "darwin":
        try:
            memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
        try:
            cpu_brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_brand": cpu_brand,
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": memory,
        "python": platform.python_version(),
    }


def _bootstrap_ci(values: list[float], *, seed: int, draws: int = 10_000) -> list[float]:
    if len(values) < 2:
        return [values[0], values[0]] if values else [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(draws)
    )
    return [means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]]


def _summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {arm: [r for r in trials if r["arm"] == arm] for arm in ("baseline", "vare")}
    metrics: dict[str, Any] = {}
    for arm, rows in by_arm.items():
        latencies = [round_row["elapsed_s"] for row in rows for round_row in row["round_results"]]
        metrics[arm] = {
            "n_runs": len(rows),
            "n_rounds": len(latencies),
            "round_latency_p50_s": _quantile(latencies, 0.50),
            "round_latency_p95_s": _quantile(latencies, 0.95),
            "round_latency_p50_ci95_bootstrap": _bootstrap_ci(
                [r["round_latency_p50_s"] for r in rows], seed=1501
            ),
            "round_latency_p95_ci95_bootstrap": _bootstrap_ci(
                [r["round_latency_p95_s"] for r in rows], seed=1502
            ),
            "rollouts_per_second_mean": statistics.fmean(r["rollouts_per_second"] for r in rows),
            "rollouts_per_second_ci95_bootstrap": _bootstrap_ci(
                [r["rollouts_per_second"] for r in rows], seed=1401
            ),
            "peak_rss_bytes_median": statistics.median(r["peak_rss_bytes"] for r in rows),
            "peak_rss_bytes_ci95_bootstrap": _bootstrap_ci(
                [float(r["peak_rss_bytes"]) for r in rows], seed=1503
            ),
            "peak_python_bytes_median": statistics.median(r["peak_python_bytes"] for r in rows),
            "mean_orchestration_replay_residual_s_per_round": statistics.fmean(
                r["orchestration_replay_residual_seconds"] / r["rounds"] for r in rows
            ),
            "orchestration_replay_residual_ci95_bootstrap_s_per_round": _bootstrap_ci(
                [r["orchestration_replay_residual_seconds"] / r["rounds"] for r in rows],
                seed=1701,
            ),
            "policy_lag_max": max(r["policy_lag_max"] for r in rows),
            "verifier_lag_max": max(r["verifier_lag_max"] for r in rows),
            "consumed_policy_lag_max": max(r["consumed_policy_lag_max"] for r in rows),
            "consumed_verifier_lag_max": max(r["consumed_verifier_lag_max"] for r in rows),
            "mean_consumed_policy_lag": statistics.fmean(
                rr["consumed_policy_lag_max"] for r in rows for rr in r["round_results"]
            ),
            "mean_consumed_verifier_lag": statistics.fmean(
                rr["consumed_verifier_lag_max"] for r in rows for rr in r["round_results"]
            ),
            "invalid_training_rows_consumed": sum(
                r["invalid_training_rows_consumed"] for r in rows
            ),
            "invalid_update_acceptances": sum(
                r["invalid_update_acceptances"] for r in rows
            ),
        }
    paired_deltas = []
    for seed in sorted({r["seed"] for r in trials}):
        baseline = next(r for r in by_arm["baseline"] if r["seed"] == seed)
        vare = next(r for r in by_arm["vare"] if r["seed"] == seed)
        paired_deltas.append(vare["rollouts_per_second"] - baseline["rollouts_per_second"])
    metrics["paired_vare_minus_baseline_rollouts_per_second"] = {
        "mean": statistics.fmean(paired_deltas),
        "ci95_bootstrap": _bootstrap_ci(paired_deltas, seed=1402),
        "per_seed": paired_deltas,
    }
    paired_by_metric = {}
    for metric in (
        "round_latency_p50_s", "round_latency_p95_s", "peak_rss_bytes",
        "orchestration_replay_residual_seconds",
    ):
        deltas = []
        for seed in sorted({row["seed"] for row in trials}):
            baseline = next(row for row in by_arm["baseline"] if row["seed"] == seed)
            vare = next(row for row in by_arm["vare"] if row["seed"] == seed)
            if metric.startswith("round_latency"):
                left = baseline[metric]
                right = vare[metric]
            elif metric == "orchestration_replay_residual_seconds":
                left = baseline[metric] / baseline["rounds"]
                right = vare[metric] / vare["rounds"]
            else:
                left = baseline[metric]
                right = vare[metric]
            deltas.append(right - left)
        paired_by_metric[metric + "_vare_minus_baseline"] = {
            "mean": statistics.fmean(deltas),
            "ci95_bootstrap": _bootstrap_ci(deltas, seed=1600 + len(paired_by_metric)),
            "per_seed": deltas,
        }
    metrics["paired_differences"] = paired_by_metric
    return metrics


def _svg_plot(summary: dict[str, Any]) -> str:
    baseline = summary["baseline"]["rollouts_per_second_mean"]
    vare = summary["vare"]["rollouts_per_second_mean"]
    max_value = max(baseline, vare, 1e-9)
    width, height = 720, 360
    bars = []
    for index, (name, value, color) in enumerate(
        [("Baseline", baseline, "#73859b"), ("VARE", vare, "#286a8e")]
    ):
        x = 240 + index * 240
        bar_width = 110
        bar_height = 210 * value / max_value
        y = 275 - bar_height
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}" />'
            f'<text x="{x + bar_width / 2}" y="{y - 10:.1f}" text-anchor="middle" font-size="15">{value:.1f}</text>'
            f'<text x="{x + bar_width / 2}" y="305" text-anchor="middle" font-size="16">{name}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white" />'
        '<text x="360" y="34" text-anchor="middle" font-size="19">Paired CPU pipeline throughput</text>'
        '<text x="360" y="55" text-anchor="middle" font-size="12">rollouts/s, mean across independent seeds; lower is slower</text>'
        '<line x1="160" y1="275" x2="620" y2="275" stroke="#333" />'
        + "".join(bars)
        + '<text x="360" y="340" text-anchor="middle" font-size="12">Fixed local CPU workload; simulated four-parameter policy</text></svg>'
    )


def _svg_latency(summary: dict[str, Any]) -> str:
    width, height = 720, 360
    values = [
        ("Baseline p50", summary["baseline"]["round_latency_p50_s"], "#73859b"),
        ("VARE p50", summary["vare"]["round_latency_p50_s"], "#286a8e"),
        ("Baseline p95", summary["baseline"]["round_latency_p95_s"], "#a9b4c0"),
        ("VARE p95", summary["vare"]["round_latency_p95_s"], "#72a6bf"),
    ]
    maximum = max((value for _, value, _ in values), default=1.0) or 1.0
    bars = []
    for index, (name, value, color) in enumerate(values):
        x = 110 + index * 145
        bar_height = 210 * value / maximum
        y = 275 - bar_height
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="80" height="{bar_height:.1f}" fill="{color}" />'
            f'<text x="{x + 40}" y="{y - 9:.1f}" text-anchor="middle" font-size="13">{value:.3f}</text>'
            f'<text x="{x + 40}" y="305" text-anchor="middle" font-size="12">{name}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white" />'
        '<text x="360" y="34" text-anchor="middle" font-size="19">End-to-end round latency</text>'
        '<text x="360" y="55" text-anchor="middle" font-size="12">seconds; quantiles over round samples</text>'
        '<line x1="70" y1="275" x2="670" y2="275" stroke="#333" />'
        + "".join(bars)
        + "</svg>"
    )


def _svg_memory(summary: dict[str, Any]) -> str:
    values = [
        ("Baseline RSS", summary["baseline"]["peak_rss_bytes_median"] / (1024**2), "#73859b"),
        ("VARE RSS", summary["vare"]["peak_rss_bytes_median"] / (1024**2), "#286a8e"),
        ("Baseline Python", summary["baseline"]["peak_python_bytes_median"] / (1024**2), "#a9b4c0"),
        ("VARE Python", summary["vare"]["peak_python_bytes_median"] / (1024**2), "#72a6bf"),
    ]
    maximum = max((value for _, value, _ in values), default=1.0) or 1.0
    bars = []
    for index, (name, value, color) in enumerate(values):
        x = 110 + index * 145
        bar_height = 210 * value / maximum
        y = 275 - bar_height
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="80" height="{bar_height:.1f}" fill="{color}" />'
            f'<text x="{x + 40}" y="{y - 9:.1f}" text-anchor="middle" font-size="13">{value:.1f}</text>'
            f'<text x="{x + 40}" y="305" text-anchor="middle" font-size="11">{name}</text>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360" viewBox="0 0 720 360">'
        '<rect width="100%" height="100%" fill="white" />'
        '<text x="360" y="34" text-anchor="middle" font-size="19">Peak memory per isolated trial</text>'
        '<text x="360" y="55" text-anchor="middle" font-size="12">MiB; median of independent runs</text>'
        '<line x1="70" y1="275" x2="670" y2="275" stroke="#333" />'
        + "".join(bars)
        + "</svg>"
    )


def run_experiment(output: Path, seeds: tuple[int, ...], rounds: int) -> dict[str, Any]:
    if seeds != DEFAULT_SEEDS or rounds != DEFAULT_ROUNDS:
        raise ValueError("locked v1 requires all 12 seeds and six rounds; reduced runs are not protocol evidence")
    output.mkdir(parents=True, exist_ok=False)
    trials: list[dict[str, Any]] = []
    script = Path(__file__).resolve()
    for seed in seeds:
        order = ("vare", "baseline") if seed % 2 else ("baseline", "vare")
        for arm in order:
            completed = subprocess.run(
                [
                    sys.executable, str(script), "--child-trial", arm, str(seed),
                    str(rounds), str(DEFAULT_ROLLOUTS), str(DEFAULT_EVAL_N),
                ],
                check=True, capture_output=True, text=True,
            )
            row = json.loads(completed.stdout)
            trials.append(row)
            with (output / "trials.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = _summary(trials)
    raw_digest = hashlib.sha256((output / "trials.jsonl").read_bytes()).hexdigest()
    payload = {
        "protocol": PROTOCOL,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tracked_source_changes_present": bool(
            subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
            or subprocess.call(["git", "diff", "--cached", "--quiet"], cwd=ROOT) != 0
        ),
        "machine": _machine_info(),
        "seeds": list(seeds),
        "rounds_per_trial": rounds,
        "rollouts_per_round": DEFAULT_ROLLOUTS,
        "eval_examples": DEFAULT_EVAL_N,
        "training_is_simulated": True,
        "model_updates_are_real_lm_updates": False,
        "protocol_sha256": hashlib.sha256(
            (ROOT / "protocols/e2e_rl_system_benchmark_v1.lock.json").read_bytes()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "raw_trials_sha256": raw_digest,
        "summary": summary,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output / "throughput.svg").write_text(_svg_plot(summary), encoding="utf-8")
    (output / "latency.svg").write_text(_svg_latency(summary), encoding="utf-8")
    (output / "memory.svg").write_text(_svg_memory(summary), encoding="utf-8")
    _write_csv(output / "summary.csv", summary)
    return payload


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = [
        "arm,n_runs,round_latency_p50_s,round_latency_p95_s,"
        "round_latency_p50_ci95_low_s,round_latency_p50_ci95_high_s,"
        "round_latency_p95_ci95_low_s,round_latency_p95_ci95_high_s,"
        "rollouts_per_second_mean,throughput_ci95_low,throughput_ci95_high,"
        "peak_rss_bytes_median,mean_orchestration_replay_residual_s_per_round"
    ]
    for arm in ("baseline", "vare"):
        item = summary[arm]
        values = [
            item["n_runs"], item["round_latency_p50_s"], item["round_latency_p95_s"],
            *item["round_latency_p50_ci95_bootstrap"],
            *item["round_latency_p95_ci95_bootstrap"],
            item["rollouts_per_second_mean"],
            *item["rollouts_per_second_ci95_bootstrap"],
            item["peak_rss_bytes_median"],
            item["mean_orchestration_replay_residual_s_per_round"],
        ]
        rows.append(arm + "," + ",".join(str(value) for value in values))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_manifest(output: Path) -> None:
    paths = [
        path for path in output.iterdir()
        if path.is_file() and path.name != "manifest.json"
    ]
    entries = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    entries["protocols/e2e_rl_system_benchmark_v1.lock.json"] = hashlib.sha256(
        (ROOT / "protocols/e2e_rl_system_benchmark_v1.lock.json").read_bytes()
    ).hexdigest()
    entries["benchmarks/e2e_rl_system/runner.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps({"protocol": PROTOCOL, "sha256": entries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# Fault harness deliberately uses a known positive synthetic candidate so that
# evaluator and publication faults reach the promotion boundary deterministically.
class FaultHooks(MeasuredHooks):
    def __init__(self, *, seed: int, eval_n: int, fault: str):
        super().__init__(seed=seed, eval_n=eval_n)
        self.fault = fault
        self.fault_used = False
        self.barrier_count = 0
        self.barrier = asyncio.Event()
        self.invalid_training_rows = 0
        self.max_policy_lag = 0
        self.forced_candidate = fault in {
            "evaluator_failure", "interrupted_promotion", "concurrent_execution"
        }

    async def rollout(self, task, policy_id, policy_version, step):
        if self.fault == "worker_crash" and not self.fault_used:
            self.fault_used = True
            proc = subprocess.run(
                [sys.executable, "-c", "import os; os._exit(73)"], check=False
            )
            if proc.returncode != 73:
                raise RuntimeError(f"fault worker exited unexpectedly: {proc.returncode}")
            raise RuntimeError("injected worker process exited with status 73")
        attempt = await super().rollout(task, policy_id, policy_version, step)
        if self.fault == "stale_rollouts":
            attempt.policy_version = max(0, policy_version - 3)
        self.max_policy_lag = max(
            self.max_policy_lag,
            max(0, policy_version - attempt.policy_version),
        )
        return attempt

    async def train_candidate(self, incumbent_id, experiences):
        if self.fault == "concurrent_execution":
            # Both baseline transactions capture the same next id before the
            # rendezvous, modeling an uncoordinated shared checkpoint writer.
            candidate_id = f"policy-{len(self.policies)}"
            candidate = copy.deepcopy(self.policies[incumbent_id])
            candidate.weights = [0.2, 1.2, -0.9, 1.4]
            self.barrier_count += 1
            if self.barrier_count >= 2:
                self.barrier.set()
            try:
                await asyncio.wait_for(self.barrier.wait(), timeout=0.03)
            except TimeoutError:
                pass
            self.policies[candidate_id] = candidate
            return candidate_id
        if self.forced_candidate:
            candidate_id = f"policy-{len(self.policies)}"
            candidate = copy.deepcopy(self.policies[incumbent_id])
            candidate.weights = [0.2, 1.2, -0.9, 1.4]
            self.policies[candidate_id] = candidate
            return candidate_id
        return await super().train_candidate(incumbent_id, experiences)

    async def evaluate(self, policy_id):
        if self.fault == "evaluator_failure" and policy_id != self.active_id:
            raise RuntimeError("injected evaluator failure")
        return await super().evaluate(policy_id)

    async def promote(self, candidate_id):
        if self.fault == "interrupted_promotion" and not self.fault_used:
            self.fault_used = True
            await super().promote(candidate_id)
            raise asyncio.CancelledError("injected interruption after publication mutation")
        return await super().promote(candidate_id)


async def _baseline_fault_round(
    hooks, verifier, tasks, replay, curriculum, round_index, config, eval_n=128
):
    eval_tasks = make_tasks(eval_n, seed=20262000 + round_index)
    verifier_ensemble = VerifierEnsemble([VerifierMember(verifier)])
    return await _baseline_round(
        hooks=hooks, verifier=verifier, verifier_ensemble=verifier_ensemble,
        tasks=tasks, replay=replay,
        curriculum=curriculum, failure_miner=FailureMiner(),
        round_index=round_index, config=config, eval_tasks=eval_tasks,
    )


def _task_set_digest(tasks: list[Task]) -> str:
    content = [(task.id, task.prompt, task.family) for task in tasks]
    return hashlib.sha256(
        json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


async def _run_fault_case(arm: str, fault: str, seed: int) -> dict[str, Any]:
    case_started = time.perf_counter()
    hooks = FaultHooks(seed=seed, eval_n=128, fault=fault)
    verifier = TimedVerifier(version=4 if fault == "stale_rollouts" else 1)
    if fault == "stale_rollouts":
        hooks.version = 4
        verifier.stale = True
    ensemble = VerifierEnsemble([VerifierMember(verifier)])
    tasks = _tasks(seed, DEFAULT_ROLLOUTS)
    config = _make_config()
    config = EngineConfig(
        rollout_concurrency=config.rollout_concurrency,
        replay_batch_size=config.replay_batch_size,
        samples_per_task=config.samples_per_task,
        preserve_rollout_groups=config.preserve_rollout_groups,
        replay=config.replay,
        promotion=config.promotion,
        lag=LagConfig(max_policy_lag=2, max_verifier_lag=2, max_shift=0.35),
    )
    before = (hooks.active_policy(), sorted(hooks.policies))
    recovery_error = None
    recovered_exactly = None
    fault_handling_s = None
    recovery_retry_s = None
    outcome: dict[str, Any] = {}
    if arm == "vare":
        loop = CapabilityLoop(hooks=hooks, verifier=ensemble, config=config, seed=seed)
        if fault == "concurrent_execution":
            results = await asyncio.gather(*[
                loop.run_round(tasks, round_index=i, rollout_count=DEFAULT_ROLLOUTS)
                for i in range(2)
            ])
            candidate_ids = [result.candidate_id for result in results]
            outcome = {
                "concurrent_rounds_completed": len(results),
                "duplicate_candidate_ids": len(candidate_ids) != len(set(candidate_ids)),
                "policy_version_after": hooks.version,
                "recovery_fenced": False,
            }
        elif fault == "worker_crash":
            fault_started = time.perf_counter()
            try:
                await loop.run_round(tasks, round_index=0, rollout_count=DEFAULT_ROLLOUTS)
            except RuntimeError as exc:
                recovery_error = str(exc)
            fault_handling_s = time.perf_counter() - fault_started
            recovered_exactly = before == (hooks.active_policy(), sorted(hooks.policies))
            retry_started = time.perf_counter()
            retry = await loop.run_round(tasks, round_index=1, rollout_count=DEFAULT_ROLLOUTS)
            recovery_retry_s = time.perf_counter() - retry_started
            outcome = {
                "retry_completed": True, "retry_promoted": retry.decision.accepted,
                "fault_handling_s": fault_handling_s, "recovery_retry_s": recovery_retry_s,
            }
        elif fault == "stale_rollouts":
            result = await loop.run_round(tasks, round_index=0, rollout_count=DEFAULT_ROLLOUTS)
            outcome = {
                "dropped_stale_rollouts": result.dropped_experiences,
                "trained_on_stale_rows": hooks.invalid_training_rows,
                "max_policy_lag": hooks.max_policy_lag,
                "max_verifier_lag": verifier.max_verifier_lag,
                "candidate_id": result.candidate_id,
                "promoted": result.decision.accepted,
            }
        elif fault == "evaluator_failure":
            fault_started = time.perf_counter()
            try:
                await loop.run_round(tasks, round_index=0, rollout_count=DEFAULT_ROLLOUTS)
            except RuntimeError as exc:
                recovery_error = str(exc)
            fault_handling_s = time.perf_counter() - fault_started
            recovered_exactly = before == (hooks.active_policy(), sorted(hooks.policies))
            candidate_discarded = sorted(hooks.policies) == ["policy-0"]
            hooks.fault = "none"
            retry_started = time.perf_counter()
            await loop.run_round(tasks, round_index=1, rollout_count=DEFAULT_ROLLOUTS)
            recovery_retry_s = time.perf_counter() - retry_started
            outcome = {
                "candidate_discarded": candidate_discarded,
                "retry_completed": True,
                "fault_handling_s": fault_handling_s,
                "recovery_retry_s": recovery_retry_s,
            }
        elif fault == "interrupted_promotion":
            try:
                await loop.run_round(tasks, round_index=0, rollout_count=DEFAULT_ROLLOUTS)
            except BaseException as exc:
                recovery_error = type(exc).__name__ + ": " + str(exc)
            try:
                await loop.run_round(tasks, round_index=1, rollout_count=DEFAULT_ROLLOUTS)
            except BaseException as exc:
                outcome["reentry_error"] = type(exc).__name__
            outcome["recovery_fenced"] = isinstance(
                getattr(loop, "_promotion_failure", None), BaseException
            )
            outcome["active_policy_after_interruption"] = hooks.active_policy()
            outcome["exact_rollback"] = before == (hooks.active_policy(), sorted(hooks.policies))
        else:
            raise ValueError(f"unknown fault: {fault}")
    elif arm == "baseline":
        replay = PrioritizedReplay(config.replay, seed=seed)
        curriculum = AdaptiveCurriculum(seed=seed)
        if fault == "concurrent_execution":
            rows = await asyncio.gather(*[
                _baseline_fault_round(hooks, verifier, tasks, replay, curriculum, i, config)
                for i in range(2)
            ], return_exceptions=True)
            candidate_ids = [r.get("candidate_id") for r in rows if isinstance(r, dict)]
            outcome = {
                "concurrent_rounds_completed": sum(isinstance(r, dict) for r in rows),
                "duplicate_candidate_ids": len(candidate_ids) != len(set(candidate_ids)),
                "policy_version_after": hooks.version,
                "recovery_fenced": False,
            }
        elif fault == "worker_crash":
            fault_started = time.perf_counter()
            try:
                await _baseline_fault_round(hooks, verifier, tasks, replay, curriculum, 0, config)
            except RuntimeError as exc:
                recovery_error = str(exc)
            fault_handling_s = time.perf_counter() - fault_started
            recovered_exactly = before == (hooks.active_policy(), sorted(hooks.policies))
            retry_started = time.perf_counter()
            await _baseline_fault_round(hooks, verifier, tasks, replay, curriculum, 1, config)
            recovery_retry_s = time.perf_counter() - retry_started
            outcome = {
                "retry_completed": True, "retry_promoted": hooks.active_id != "policy-0",
                "fault_handling_s": fault_handling_s, "recovery_retry_s": recovery_retry_s,
            }
        elif fault == "stale_rollouts":
            row = await _baseline_fault_round(hooks, verifier, tasks, replay, curriculum, 0, config)
            outcome = {
                "dropped_stale_rollouts": 0,
                "trained_on_stale_rows": hooks.invalid_training_rows,
                "max_policy_lag": hooks.max_policy_lag,
                "max_verifier_lag": verifier.max_verifier_lag,
                "candidate_id": row["candidate_id"],
                "promoted": row["promoted"],
            }
        elif fault == "evaluator_failure":
            fault_started = time.perf_counter()
            try:
                await _baseline_fault_round(hooks, verifier, tasks, replay, curriculum, 0, config)
            except RuntimeError as exc:
                recovery_error = str(exc)
            fault_handling_s = time.perf_counter() - fault_started
            recovered_exactly = before == (hooks.active_policy(), sorted(hooks.policies))
            candidate_discarded = sorted(hooks.policies) == ["policy-0"]
            hooks.fault = "none"
            retry_started = time.perf_counter()
            await _baseline_fault_round(hooks, verifier, tasks, replay, curriculum, 1, config)
            recovery_retry_s = time.perf_counter() - retry_started
            outcome = {
                "candidate_discarded": candidate_discarded,
                "retry_completed": True,
                "fault_handling_s": fault_handling_s,
                "recovery_retry_s": recovery_retry_s,
            }
        elif fault == "interrupted_promotion":
            try:
                await _baseline_fault_round(hooks, verifier, tasks, replay, curriculum, 0, config)
            except BaseException as exc:
                recovery_error = type(exc).__name__ + ": " + str(exc)
            outcome = {
                "recovery_fenced": False,
                "active_policy_after_interruption": hooks.active_policy(),
                "exact_rollback": before == (hooks.active_policy(), sorted(hooks.policies)),
            }
        else:
            raise ValueError(f"unknown fault: {fault}")
    else:
        raise ValueError(f"unknown arm: {arm}")
    return {
        "protocol": PROTOCOL,
        "arm": arm,
        "fault": fault,
        "seed": seed,
        "recovery_error": recovery_error,
        "recovered_exactly_after_fault": recovered_exactly,
        "fault_handling_s": fault_handling_s,
        "recovery_retry_s": recovery_retry_s,
        "fault_case_elapsed_s": time.perf_counter() - case_started,
        "invalid_update_accepted": bool(outcome.get("trained_on_stale_rows", 0)),
        "outcome": outcome,
    }


async def run_fault_suite(seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = []
    faults = (
        "worker_crash", "stale_rollouts", "evaluator_failure",
        "concurrent_execution", "interrupted_promotion",
    )
    for seed in seeds:
        for fault in faults:
            for arm in ("baseline", "vare"):
                rows.append(await _run_fault_case(arm, fault, seed))
    return rows


def _write_faults(output: Path, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = asyncio.run(run_fault_suite(seeds))
    path = output / "faults.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "faults.sha256").write_text(digest + "  faults.jsonl\n")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--fault-seeds", type=int, default=3)
    parser.add_argument("--skip-faults", action="store_true")
    parser.add_argument("--child-trial", nargs=5, metavar=("ARM", "SEED", "ROUNDS", "ROLLOUTS", "EVAL_N"))
    args = parser.parse_args(argv)
    if args.child_trial:
        arm, seed, rounds, rollouts, eval_n = args.child_trial
        if (arm not in ("vare", "baseline") or int(seed) not in DEFAULT_SEEDS
                or (int(rounds), int(rollouts), int(eval_n)) !=
                (DEFAULT_ROUNDS, DEFAULT_ROLLOUTS, DEFAULT_EVAL_N)):
            parser.error("child trial must match the locked v1 performance workload")
        print(json.dumps(_child_trial(arm, int(seed), int(rounds), int(rollouts), int(eval_n))))
        return 0
    if (args.seeds != len(DEFAULT_SEEDS) or args.rounds != DEFAULT_ROUNDS
            or args.fault_seeds != 3 or args.skip_faults):
        parser.error("locked v1 requires 12 seeds, six rounds, three fault seeds and all faults; reduced runs are not protocol evidence")
    if args.seeds < 2 or args.rounds < 2 or args.fault_seeds < 1:
        parser.error("need >=2 performance seeds/rounds and >=1 fault seed")
    if args.output is None:
        parser.error("--output is required")
    seeds = DEFAULT_SEEDS[:args.seeds]
    if len(seeds) != args.seeds:
        parser.error(f"at most {len(DEFAULT_SEEDS)} locked seeds are available")
    payload = run_experiment(args.output, seeds, args.rounds)
    if not args.skip_faults:
        fault_seeds = tuple(20261991 + i for i in range(args.fault_seeds))
        faults = _write_faults(args.output, fault_seeds)
        payload["fault_runs"] = len(faults)
        payload["faults_sha256"] = hashlib.sha256(
            (args.output / "faults.jsonl").read_bytes()
        ).hexdigest()
        payload["fault_outcomes"] = _fault_summary(faults)
        (args.output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_manifest(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _fault_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for arm in ("baseline", "vare"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        result[arm] = {
            "n_fault_trials": len(arm_rows),
            "invalid_update_acceptances": sum(row["invalid_update_accepted"] for row in arm_rows),
            "exact_recoveries": sum(row["recovered_exactly_after_fault"] is True for row in arm_rows),
            "recovery_fences": sum(bool(row["outcome"].get("recovery_fenced")) for row in arm_rows),
            "candidate_discarded": sum(row["outcome"].get("candidate_discarded") is True for row in arm_rows),
            "mean_fault_handling_s": _mean_present(
                row["fault_handling_s"] for row in arm_rows
            ),
            "mean_recovery_retry_s": _mean_present(
                row["recovery_retry_s"] for row in arm_rows
            ),
        }
    result["by_fault"] = {}
    for fault in sorted({row["fault"] for row in rows}):
        result["by_fault"][fault] = {}
        for arm in ("baseline", "vare"):
            selected = [
                row for row in rows if row["fault"] == fault and row["arm"] == arm
            ]
            result["by_fault"][fault][arm] = {
                "n": len(selected),
                "invalid_update_acceptances": sum(
                    row["invalid_update_accepted"] for row in selected
                ),
                "exact_recoveries": sum(
                    row["recovered_exactly_after_fault"] is True for row in selected
                ),
                "recovery_fences": sum(
                    bool(row["outcome"].get("recovery_fenced")) for row in selected
                ),
                "candidate_discarded": sum(
                    row["outcome"].get("candidate_discarded") is True for row in selected
                ),
                "mean_fault_handling_s": _mean_present(
                    row["fault_handling_s"] for row in selected
                ),
                "mean_recovery_retry_s": _mean_present(
                    row["recovery_retry_s"] for row in selected
                ),
            }
    return result


def _mean_present(values) -> float | None:
    observed = [value for value in values if value is not None]
    return statistics.fmean(observed) if observed else None


if __name__ == "__main__":
    raise SystemExit(main())
