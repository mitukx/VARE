from __future__ import annotations

import asyncio
from collections.abc import Sequence

from .config import EngineConfig
from .curriculum import AdaptiveCurriculum
from .failure import FailureMiner
from .evidence import HashChainLedger
from .interfaces import LoopHooks
from .lag import DistributionShiftMonitor, LagController
from .lifecycle import FailureDrivenTaskGenerator, VerifierRefreshController
from .promotion import PromotionGate
from .replay import PrioritizedReplay
from .telemetry import EventLog
from .types import Experience, RoundResult, Task
from .verifiers import VerifierEnsemble


class CapabilityLoop:
    """Closed-loop capability improvement control plane.

    Model serving and optimizer internals are supplied via LoopHooks. Optional
    lifecycle hooks let failures create new tasks and trigger verifier refresh,
    which makes generator/verifier co-evolution explicit rather than implicit.
    """

    def __init__(
        self,
        *,
        hooks: LoopHooks,
        verifier: VerifierEnsemble,
        config: EngineConfig | None = None,
        event_log: EventLog | None = None,
        task_generator: FailureDrivenTaskGenerator | None = None,
        verifier_refresh: VerifierRefreshController | None = None,
        decision_ledger: HashChainLedger | None = None,
        seed: int = 0,
    ) -> None:
        self.hooks = hooks
        self.verifier = verifier
        self.config = config or EngineConfig()
        self.lag = LagController(self.config.lag)
        self.replay = PrioritizedReplay(self.config.replay, seed=seed)
        self.failures = FailureMiner()
        self.curriculum = AdaptiveCurriculum(seed=seed)
        self.promotion = PromotionGate(self.config.promotion)
        self.shift = DistributionShiftMonitor()
        self.events = event_log or EventLog()
        self.task_generator = task_generator
        self.verifier_refresh = verifier_refresh
        self.decision_ledger = decision_ledger
        self._generated_tasks: list[Task] = []
        self.step = 0
        self._run_serial = 0

    async def _one_rollout(self, sem: asyncio.Semaphore, task: Task, policy_id: str, policy_version: int):
        async with sem:
            # Allocate the rollout sequence before the first await. Coroutine code
            # executes serially until it yields, so concurrent rollouts cannot
            # accidentally reuse the same seed/provenance step.
            step = self.step
            self.step += 1
            attempt = await self.hooks.rollout(task, policy_id, policy_version, step)
            group_id = task.metadata.get("vare_rollout_group")
            if group_id is not None:
                attempt.metadata["vare_rollout_group"] = group_id
                attempt.metadata["vare_rollout_group_size"] = task.metadata[
                    "vare_rollout_group_size"
                ]
            verification = await self.verifier.verify(attempt)
            return attempt, verification

    async def run_round(self, tasks: Sequence[Task], *, round_index: int, rollout_count: int | None = None) -> RoundResult:
        source_tasks = list(tasks) + self._generated_tasks
        if not source_tasks:
            raise ValueError("run_round requires tasks")
        incumbent_id, incumbent_version = self.hooks.active_policy()
        self.shift.fit_reference(t.family for t in source_tasks)
        target_rollouts = len(source_tasks) if rollout_count is None else rollout_count
        if target_rollouts <= 0:
            raise ValueError("rollout_count must be positive")
        if self.config.samples_per_task <= 0:
            raise ValueError("samples_per_task must be positive")
        # A round index can be replayed by callers, so include a deterministic
        # loop-local serial to keep rollout groups unique across run_round calls.
        self._run_serial += 1
        run_serial = self._run_serial
        if self.config.samples_per_task == 1:
            base_tasks = self.curriculum.choose(source_tasks, target_rollouts)
            chosen = [
                Task(
                    id=t.id,
                    prompt=t.prompt,
                    family=t.family,
                    metadata={
                        **t.metadata,
                        "vare_rollout_group": f"r{round_index}-run{run_serial}-g{i}",
                        "vare_rollout_group_size": 1,
                    },
                )
                for i, t in enumerate(base_tasks)
            ]
        else:
            group_count = (target_rollouts + self.config.samples_per_task - 1) // self.config.samples_per_task
            base_tasks = self.curriculum.choose(source_tasks, group_count)
            chosen = []
            for group_index, task in enumerate(base_tasks):
                group_id = f"r{round_index}-run{run_serial}-g{group_index}"
                for _ in range(self.config.samples_per_task):
                    chosen.append(
                        Task(
                            id=task.id,
                            prompt=task.prompt,
                            family=task.family,
                            metadata={
                                **task.metadata,
                                "vare_rollout_group": group_id,
                                "vare_rollout_group_size": self.config.samples_per_task,
                            },
                        )
                    )
        sem = asyncio.Semaphore(self.config.rollout_concurrency)
        pairs = await asyncio.gather(*(self._one_rollout(sem, t, incumbent_id, incumbent_version) for t in chosen))

        admitted: list[Experience] = []
        admitted_freshness: list[float] = []
        dropped = 0
        for attempt, verification in pairs:
            self.shift.observe(attempt.task.family)
            shift_score = self.shift.score()
            lag = self.lag.assess(
                rollout_policy_version=attempt.policy_version,
                active_policy_version=incumbent_version,
                reward_verifier_version=verification.verifier_version,
                active_verifier_version=self.verifier.active_version,
                shift_score=shift_score,
            )
            if not lag.admitted:
                dropped += 1
                self.events.emit("experience_dropped", reasons=lag.reasons, task_id=attempt.task.id)
                continue
            exp = Experience(
                attempt=attempt,
                verification=verification,
                policy_lag=lag.policy_lag,
                verifier_lag=lag.verifier_lag,
                shift_score=lag.shift_score,
            )
            admitted.append(exp)
            admitted_freshness.append(lag.freshness)

        if self.config.preserve_rollout_groups:
            admitted_groups: dict[str, list[tuple[Experience, float]]] = {}
            for exp, freshness in zip(admitted, admitted_freshness, strict=True):
                group_id = exp.attempt.metadata.get("vare_rollout_group")
                if group_id is None:
                    group_id = exp.attempt.task.metadata.get("vare_rollout_group")
                key = str(group_id) if group_id is not None else f"ungrouped:{id(exp)}"
                admitted_groups.setdefault(key, []).append((exp, freshness))

            for group_id, members in admitted_groups.items():
                first = members[0][0]
                expected_size = first.attempt.metadata.get("vare_rollout_group_size")
                if expected_size is None:
                    expected_size = first.attempt.task.metadata.get(
                        "vare_rollout_group_size"
                    )
                complete = (
                    isinstance(expected_size, int)
                    and not isinstance(expected_size, bool)
                    and expected_size == len(members)
                )
                inserted = complete and self.replay.add_group(
                    [exp for exp, _ in members],
                    [freshness for _, freshness in members],
                )
                if not inserted:
                    self.events.emit(
                        "replay_group_not_admitted",
                        group_id=group_id,
                        admitted_count=len(members),
                        expected_size=expected_size,
                        reason=(
                            "incomplete_after_lag"
                            if not complete
                            else "capacity_or_priority"
                        ),
                    )
        else:
            for exp, freshness in zip(admitted, admitted_freshness, strict=True):
                self.replay.add(exp, freshness)

        clusters = self.failures.mine(admitted)
        self.curriculum.update(clusters)

        if self.verifier_refresh is not None:
            refreshed = await self.verifier_refresh.maybe_refresh(clusters, admitted)
            if refreshed:
                self.events.emit("verifier_refreshed", active_version=self.verifier.active_version)

        if self.task_generator is not None and clusters:
            generated = await self.task_generator.generate(
                clusters,
                admitted,
                self.config.generated_tasks_per_round,
            )
            if generated:
                self._generated_tasks.extend(generated)
                self._generated_tasks = self._generated_tasks[-self.config.generated_task_buffer :]
                self.events.emit("tasks_generated", count=len(generated))

        screened_stale = 0

        def current_freshness(exp: Experience) -> float | None:
            nonlocal screened_stale
            decision = self.lag.assess(
                rollout_policy_version=exp.attempt.policy_version,
                active_policy_version=incumbent_version,
                reward_verifier_version=exp.verification.verifier_version,
                active_verifier_version=self.verifier.active_version,
                shift_score=exp.shift_score,
            )
            exp.policy_lag = decision.policy_lag
            exp.verifier_lag = decision.verifier_lag
            if not decision.admitted:
                screened_stale += 1
                return None
            return decision.freshness

        train_batch = self.replay.sample_current(
            self.config.replay_batch_size,
            freshness_fn=current_freshness,
            grouped=self.config.preserve_rollout_groups,
        )
        if screened_stale:
            self.events.emit("replay_experiences_screened", count=screened_stale)
        if not train_batch:
            candidate_id = incumbent_id
        else:
            candidate_id = await self.hooks.train_candidate(incumbent_id, train_batch)

        incumbent_eval, candidate_eval = await asyncio.gather(
            self.hooks.evaluate(incumbent_id), self.hooks.evaluate(candidate_id)
        )
        decision = self.promotion.decide(incumbent_eval, candidate_eval)
        if candidate_id != incumbent_id and decision.accepted:
            await self.hooks.promote(candidate_id)
            promoted_id = candidate_id
            self.events.emit("candidate_promoted", candidate_id=candidate_id, gain=decision.primary_gain)
        else:
            promoted_id = incumbent_id
            if candidate_id != incumbent_id:
                await self.hooks.discard(candidate_id)
            self.events.emit("candidate_rejected", candidate_id=candidate_id, reasons=decision.reasons)

        self.events.emit(
            "round_complete",
            round_index=round_index,
            incumbent_id=incumbent_id,
            candidate_id=candidate_id,
            promoted_id=promoted_id,
            admitted=len(admitted),
            dropped=dropped,
            primary_gain=decision.primary_gain,
            paired_lcb=decision.paired_lcb,
        )
        if self.decision_ledger is not None:
            self.decision_ledger.append(
                "promotion_decision",
                {
                    "round_index": round_index,
                    "incumbent_id": incumbent_id,
                    "candidate_id": candidate_id,
                    "promoted_id": promoted_id,
                    "accepted": decision.accepted,
                    "reasons": list(decision.reasons),
                    "primary_gain": decision.primary_gain,
                    "paired_gain": decision.paired_gain,
                    "paired_lcb": decision.paired_lcb,
                    "paired_n": decision.paired_n,
                    "admitted": len(admitted),
                    "dropped": dropped,
                },
            )
        return RoundResult(
            round_index=round_index,
            incumbent_id=incumbent_id,
            candidate_id=candidate_id,
            promoted_id=promoted_id,
            decision=decision,
            incumbent_eval=incumbent_eval,
            candidate_eval=candidate_eval,
            failures=clusters,
            admitted_experiences=len(admitted),
            dropped_experiences=dropped,
        )
