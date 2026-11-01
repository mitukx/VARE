from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Sequence
from uuid import uuid4

from .attestation import EvidenceTrustRoot
from .safety import SafetyInterlock, SafetyInterlockDenied
from .config import EngineConfig
from .curriculum import AdaptiveCurriculum
from .failure import FailureMiner
from .evidence import HashChainLedger, sha256_json
from .interfaces import LoopHooks
from .lag import (
    DistributionShiftMonitor,
    LagController,
    make_experience_freshness_assessor,
)
from .lifecycle import FailureDrivenTaskGenerator, VerifierRefreshController
from .promotion import PromotionGate
from .promotion_journal import PromotionJournal, PromotionRecoveryRequired
from .replay import PrioritizedReplay
from .reward_evidence import validate_complete_rvl_reward_batch
from .runtime import RuntimeCoordinator, coordinator_for
from .telemetry import EventLog
from .types import Attempt, Experience, PromotionDecision, RoundResult, Task, Verification, snapshot_verification
from .verifiers import VerifierEnsemble


class _NoVerifiedExperienceReady(Exception):
    """An owned ingestion poll found no eligible complete group."""


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
        evidence_trust_root: EvidenceTrustRoot | None = None,
        safety_interlock: SafetyInterlock | None = None,
        promotion_journal: PromotionJournal | None = None,
        checkpoint_hasher: Callable[[str], str] | None = None,
        training_evidence_authorizer: Callable[[Experience], bool] | None = None,
        require_authenticated_training: bool = False,
        require_current_evaluation_verifier: bool = False,
        runtime_coordinator: RuntimeCoordinator | None = None,
        seed: int = 0,
    ) -> None:
        self.hooks = hooks
        self.verifier = verifier
        self.config = config or EngineConfig()
        self.lag = LagController(self.config.lag)
        self.replay = PrioritizedReplay(self.config.replay, seed=seed)
        self.failures = FailureMiner()
        self.curriculum = AdaptiveCurriculum(seed=seed)
        self.promotion = PromotionGate(self.config.promotion, trust_root=evidence_trust_root)
        self.shift = DistributionShiftMonitor()
        self.events = event_log or EventLog()
        self.task_generator = task_generator
        self.verifier_refresh = verifier_refresh
        self.decision_ledger = decision_ledger
        self.safety_interlock = safety_interlock
        self.promotion_journal = promotion_journal
        self.checkpoint_hasher = checkpoint_hasher
        self.evidence_trust_root = evidence_trust_root
        if type(require_authenticated_training) is not bool:
            raise ValueError("require_authenticated_training must be bool")
        if type(require_current_evaluation_verifier) is not bool:
            raise ValueError("require_current_evaluation_verifier must be bool")
        self.require_current_evaluation_verifier = require_current_evaluation_verifier
        if require_authenticated_training and training_evidence_authorizer is None:
            raise ValueError("authenticated training requires a trusted evidence authorizer")
        self.training_evidence_authorizer = training_evidence_authorizer
        if promotion_journal is not None and (
            checkpoint_hasher is None
            or evidence_trust_root is None
            or not evidence_trust_root.strict_checkpoint_binding
        ):
            raise ValueError(
                "durable promotion requires strict attestation and a trusted checkpoint hasher"
            )
        self._generated_tasks: list[Task] = []
        self.step = 0
        self._run_serial = 0
        self.runtime_coordinator = runtime_coordinator or coordinator_for(hooks)
        self._round_lock = self.runtime_coordinator.lock
        self._candidate_unresolved = False
        self._discard_failure: BaseException | None = None
        self._promotion_failure: BaseException | None = None

    async def _one_rollout(self, sem: asyncio.Semaphore, task: Task, policy_id: str, policy_version: int):
        async with sem:
            # Allocate the rollout sequence before the first await. Coroutine code
            # executes serially until it yields, so concurrent rollouts cannot
            # accidentally reuse the same seed/provenance step.
            step = self.step
            self.step += 1
            if self.safety_interlock is not None:
                self.safety_interlock.require("rollout", incumbent_id=policy_id)
            attempt = await self.hooks.rollout(task, policy_id, policy_version, step)
            if not isinstance(attempt, Attempt):
                raise TypeError("rollout hook must return an Attempt")
            if (
                not isinstance(attempt.task, Task)
                or attempt.task.id != task.id
                or attempt.task.prompt != task.prompt
                or attempt.task.family != task.family
            ):
                raise ValueError("rollout task identity mismatch")
            # A hook must not relabel a rollout as a different policy or step.
            # Older policy *versions* remain eligible for the explicit lag gate.
            if not isinstance(attempt.policy_id, str) or attempt.policy_id != policy_id:
                raise ValueError("rollout policy identity mismatch")
            if type(attempt.created_step) is not int or attempt.created_step != step:
                raise ValueError("rollout step identity mismatch")
            group_id = task.metadata.get("vare_rollout_group")
            if group_id is not None:
                attempt.metadata["vare_rollout_group"] = group_id
                attempt.metadata["vare_rollout_group_size"] = task.metadata[
                    "vare_rollout_group_size"
                ]
            verification = snapshot_verification(
                await self.verifier.verify(attempt), context="capability loop verifier result"
            )
            return attempt, verification

    @staticmethod
    async def _cancel_and_drain_rollouts(tasks: Sequence[asyncio.Task]) -> None:
        """Own every launched rollout until its coroutine is terminal."""
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _discard_candidate(
        self, candidate_id: str, *, propagate_cancellation: bool = True
    ) -> None:
        """Finish restoration before unlocking; failed restore poisons this loop.

        An existing transaction error takes precedence over later caller
        cancellation when restoration succeeds. A failed restoration always
        takes precedence because the runtime's state is no longer known.
        """
        discard = asyncio.create_task(self.hooks.discard(candidate_id))
        cancellation: asyncio.CancelledError | None = None
        while not discard.done():
            try:
                await asyncio.shield(discard)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
            except BaseException:
                break
        try:
            discard.result()
        except BaseException as exc:
            self._discard_failure = exc
            raise RuntimeError("candidate discard failed; policy state is unknown") from exc
        self._candidate_unresolved = False
        if cancellation is not None and propagate_cancellation:
            raise cancellation

    async def _publish_candidate(
        self, candidate_id: str, *, defer_caller_cancellation: bool
    ) -> int:
        """Own accepted publication through hook completion despite caller cancellation.

        Publication may mutate the runtime before it acknowledges success, so
        cancelling the hook would leave the caller unable to distinguish a
        rollback from a partial install. Keep the hook in an owned task and
        return only after its terminal result is known. Durable publication is
        shielded through commit; without a journal, cancellation is delivered
        once and drained before propagation.
        """
        publication = asyncio.create_task(self.hooks.promote(candidate_id))
        deferred_cancellations = 0
        caller_cancellation: asyncio.CancelledError | None = None
        cancellation_delivered_to_hook = False
        caller_task = asyncio.current_task()
        last_cancellation_requests = caller_task.cancelling() if caller_task is not None else 0

        def clear_deferred_cancellation_requests() -> None:
            # The durable branch converts cancellation into an explicit result
            # or a stronger recovery error. Clear the consumed request state so
            # a surrounding TaskGroup does not observe a stale cancellation.
            if caller_task is not None:
                while caller_task.cancelling():
                    caller_task.uncancel()

        while not publication.done():
            try:
                await asyncio.shield(publication)
            except asyncio.CancelledError as exc:
                # A self-cancelled hook is a publication failure, not a caller
                # cancellation to defer. publication.result() below preserves
                # that failure and the normal promotion recovery fence.
                if publication.done() and publication.cancelled():
                    break
                if not defer_caller_cancellation:
                    if caller_cancellation is None:
                        caller_cancellation = exc
                    # Preserve the unjournaled API's cancellation behavior,
                    # but retain ownership until the hook has finished its
                    # cancellation cleanup. This prevents detached mutation.
                    if not cancellation_delivered_to_hook:
                        cancellation_delivered_to_hook = True
                        if not publication.cancelling():
                            publication.cancel()
                    continue
                current_requests = (
                    caller_task.cancelling()
                    if caller_task is not None
                    else last_cancellation_requests
                )
                deferred_cancellations += max(
                    1, current_requests - last_cancellation_requests
                )
                last_cancellation_requests = current_requests
            except BaseException:
                # Retrieve the original hook exception below after the owned
                # task is terminal.
                break

        try:
            publication.result()
        except BaseException as exc:
            if defer_caller_cancellation and deferred_cancellations:
                clear_deferred_cancellation_requests()
            elif (caller_cancellation is not None
                  and not isinstance(exc, asyncio.CancelledError)):
                clear_deferred_cancellation_requests()
            raise
        if publication.cancelling():
            # A hook that catches its own cancellation but leaves the task's
            # cancellation request uncleared has not acknowledged a clean
            # publication. Never report it as a successful commit.
            if defer_caller_cancellation and deferred_cancellations:
                clear_deferred_cancellation_requests()
            raise asyncio.CancelledError(
                "promotion hook suppressed cancellation without clearing task state"
            )
        if caller_cancellation is not None:
            raise caller_cancellation
        if defer_caller_cancellation and deferred_cancellations:
            clear_deferred_cancellation_requests()
        return deferred_cancellations

    async def run_round(self, tasks: Sequence[Task], *, round_index: int, rollout_count: int | None = None, _verified_experiences: Sequence[Experience] | None = None, _delivery_complete: Callable[[RoundResult], Awaitable[None]] | None = None, _publication_reserve: Callable[[], Awaitable[None]] | None = None, _delivery_claim: Callable[[str], Awaitable[Sequence[Experience] | None]] | None = None, _delivery_pending: Callable[[str], bool] | None = None, _shift_reference_families: Sequence[str] | None = None) -> RoundResult:
        """Own the shared runtime before any hook; retain crash uncertainty."""
        async with self._round_lock:
            if self._discard_failure is not None:
                # Do not inspect policy identity, emit rollouts or advance
                # provenance against a runtime whose rollback failed. There
                # is no automatic reset: trusted external recovery is required.
                raise RuntimeError(
                    "candidate recovery required; policy state is unknown"
                ) from self._discard_failure
            if self._promotion_failure is not None:
                # An optional journal cannot fence the default in-memory loop.
                # Even a committed journal may have lost its acknowledgment.
                # Read no hook state until independent recovery is complete.
                raise PromotionRecoveryRequired(
                    "unresolved promotion recovery required; verify the runtime "
                    "and reconcile durable evidence before constructing a fresh loop"
                ) from self._promotion_failure
            self.runtime_coordinator.assert_usable()
            owner_id = uuid4().hex
            if self.promotion_journal is not None:
                # Claim failure must not quarantine the *other* live owner.
                self.promotion_journal.acquire_round(owner_id)
            self._candidate_unresolved = False
            try:
                if _delivery_claim is not None:
                    supplied = await _delivery_claim(owner_id)
                    if supplied is None:
                        raise _NoVerifiedExperienceReady()
                    if (not supplied or len(supplied) > min(self.config.replay.capacity, self.config.replay_batch_size)
                            or any(not isinstance(exp, Experience) or not isinstance(exp.attempt, Attempt) for exp in supplied)):
                        raise ValueError("verified batch must fit the bounded replay capacity")
                    _verified_experiences = copy.deepcopy(list(supplied))
                    tasks = [exp.attempt.task for exp in _verified_experiences]
                if _verified_experiences is None:
                    result = await self._run_round(
                        tasks, round_index=round_index, rollout_count=rollout_count
                    )
                else:
                    result = await self._run_round(
                        tasks, round_index=round_index, rollout_count=rollout_count,
                        verified_experiences=_verified_experiences,
                        publication_reserve=_publication_reserve,
                        shift_reference_families=_shift_reference_families,
                    )
                    assert _delivery_complete is not None
                    await _delivery_complete(result)
            except _NoVerifiedExperienceReady:
                self._release_runtime_round(owner_id)
                raise
            except BaseException as exc:
                delivery_unresolved = False
                if _delivery_claim is not None:
                    try:
                        assert _delivery_pending is not None
                        delivery_unresolved = _delivery_pending(owner_id)
                    except BaseException:
                        delivery_unresolved = True  # Cannot prove a clean claim failure.
                if (delivery_unresolved or _verified_experiences is not None or self._candidate_unresolved or self._discard_failure is not None
                        or self._promotion_failure is not None):
                    self.runtime_coordinator.quarantine(exc)
                    # Leave the durable reservation intact, even if a commit
                    # succeeded but its acknowledgment was lost.
                else:
                    self._release_runtime_round(owner_id)
                raise
            self._release_runtime_round(owner_id)
            self._candidate_unresolved = False
            return result

    async def run_verified_round(self, claim_experiences: Callable[[str], Awaitable[Sequence[Experience] | None]], *, round_index: int, on_complete: Callable[[RoundResult], Awaitable[None]], reserve_publication: Callable[[], Awaitable[None]], delivery_pending: Callable[[str], bool], shift_reference_families: Sequence[str]) -> RoundResult | None:
        """Consume bounded authenticated trajectories through the existing transaction.

        Does not regenerate, regrade or rename the source groups. Candidate,
        evaluation, promotion and recovery semantics are identical to run_round.
        Caller owns durable ingestion acknowledgment and restart reconciliation,
        and supplies the independent family distribution used for shift checks.
        """
        if not all(callable(fn) for fn in (claim_experiences, on_complete, reserve_publication, delivery_pending)):
            raise ValueError("streamed delivery requires owned completion and publication boundaries")
        if (not isinstance(shift_reference_families, Sequence)
                or isinstance(shift_reference_families, (str, bytes))
                or not shift_reference_families
                or any(not isinstance(family, str) or not family.strip()
                       for family in shift_reference_families)):
            raise ValueError("streamed GRPO requires an independent task-family reference")
        try:
            return await self.run_round(
                [], round_index=round_index, _delivery_claim=claim_experiences,
                _delivery_complete=on_complete, _publication_reserve=reserve_publication,
                _delivery_pending=delivery_pending,
                _shift_reference_families=tuple(shift_reference_families),
            )
        except _NoVerifiedExperienceReady:
            return None

    def _verified_pairs(self, experiences: Sequence[Experience]):
        if self.training_evidence_authorizer is None or not self.config.preserve_rollout_groups:
            raise ValueError("streamed GRPO requires authentication and intact group replay")
        seen: set[str] = set()
        groups: dict[str, list[tuple[int, str, int, str, str]]] = {}
        pairs: list[tuple[Attempt, Verification]] = []
        for exp in experiences:
            if not isinstance(exp, Experience) or not isinstance(exp.attempt, Attempt):
                raise TypeError("verified ingestion requires Experience/Attempt")
            attempt = exp.attempt
            ident = attempt.metadata.get("trajectory_id")
            group = attempt.metadata.get("vare_rollout_group")
            size = attempt.metadata.get("vare_rollout_group_size")
            if (not isinstance(ident, str) or not ident or ident in seen
                    or not isinstance(group, str) or not group
                    or type(size) is not int or size < 2
                    or attempt.task.metadata.get("vare_rollout_group") != group
                    or attempt.task.metadata.get("vare_rollout_group_size") != size):
                raise ValueError("streamed member/group identity is invalid or duplicated")
            seen.add(ident)
            members = groups.setdefault(group, [])
            members.append((size, attempt.policy_id, attempt.policy_version, attempt.task.id, attempt.task.prompt))
            value = snapshot_verification(exp.verification)
            authorized = self.training_evidence_authorizer(exp)
            if not value.trusted or type(authorized) is not bool or not authorized:
                raise ValueError("unauthenticated streamed trajectory")
            pairs.append((attempt, value))
        if any(len(members) != members[0][0] or len(set(members)) != 1 for members in groups.values()):
            raise ValueError("streamed comparison group is incomplete or heterogeneous")
        self.step += len(pairs)
        self.events.emit("verified_experience_ingested", count=len(pairs), groups=len(groups))
        return pairs

    def _release_runtime_round(self, owner_id: str) -> None:
        if self.promotion_journal is None:
            return
        try:
            assert self.checkpoint_hasher is not None
            policy_id, _ = self.hooks.active_policy()
            self.promotion_journal.assert_clean(
                policy_id=policy_id, checkpoint_sha256=self.checkpoint_hasher(policy_id)
            )
            self.promotion_journal.release_round(owner_id)
        except BaseException as exc:
            self.runtime_coordinator.quarantine(exc)
            raise PromotionRecoveryRequired("runtime round completion requires recovery") from exc

    async def _run_round(self, tasks: Sequence[Task], *, round_index: int, rollout_count: int | None = None, verified_experiences: Sequence[Experience] | None = None, publication_reserve: Callable[[], Awaitable[None]] | None = None, shift_reference_families: Sequence[str] | None = None) -> RoundResult:
        source_tasks = list(tasks) + (self._generated_tasks if verified_experiences is None else [])
        if not source_tasks:
            raise ValueError("run_round requires tasks")
        incumbent_id, incumbent_version = self.hooks.active_policy()
        if self.promotion_journal is not None:
            assert self.checkpoint_hasher is not None
            self.promotion_journal.assert_clean(
                policy_id=incumbent_id,
                checkpoint_sha256=self.checkpoint_hasher(incumbent_id),
            )
        if self.safety_interlock is not None:
            self.safety_interlock.require("rollout", incumbent_id=incumbent_id)
        if verified_experiences is None:
            self.shift.fit_reference(t.family for t in source_tasks)
        else:
            if shift_reference_families is None:
                raise ValueError("streamed GRPO requires an independent task-family reference")
            self.shift.fit_reference(shift_reference_families)
        if verified_experiences is not None:
            pairs = self._verified_pairs(verified_experiences)
        else:
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
            # gather() normally propagates the first failure while other rollouts
            # keep running. Never release the round lock with orphaned verifiers or
            # generation hooks still using the transaction's policy state.
            rollout_tasks = [
                asyncio.create_task(self._one_rollout(sem, t, incumbent_id, incumbent_version))
                for t in chosen
            ]
            group = asyncio.gather(*rollout_tasks)
            try:
                pairs = await asyncio.shield(group)
            except BaseException:
                cleanup = asyncio.create_task(self._cancel_and_drain_rollouts(rollout_tasks))
                # Repeated caller cancellation must not abandon ongoing cleanup.
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                await cleanup
                # Shield detaches the aggregate on caller cancellation. Consume its
                # terminal exception after draining children, so it cannot become
                # an unobserved _GatheringFuture failure.
                try:
                    await group
                except BaseException:
                    pass
                raise

        admitted: list[Experience] = []
        admitted_freshness: list[float] = []
        dropped = 0
        batch_shift_score = self.shift.observe_batch(
            attempt.task.family for attempt, _ in pairs
        )
        for attempt, verification in pairs:
            lag = self.lag.assess(
                rollout_policy_version=attempt.policy_version,
                active_policy_version=incumbent_version,
                reward_verifier_version=verification.verifier_version,
                active_verifier_version=self.verifier.active_version,
                shift_score=batch_shift_score,
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

        screened_stale = 0
        freshness_assessor = make_experience_freshness_assessor(
            self.lag,
            active_policy_version=incumbent_version,
            active_verifier_version=lambda: self.verifier.active_version,
        )

        def current_freshness(exp: Experience) -> float | None:
            nonlocal screened_stale
            decision = freshness_assessor(exp)
            if not decision.admitted:
                screened_stale += 1
                return None
            return decision.freshness

        # Ineligible high-priority groups must not consume the capacity needed
        # to insert a fresh group. Sampling alone only filters their outputs.
        removed = self.replay.prune_current(
            freshness_fn=current_freshness,
            grouped=self.config.preserve_rollout_groups,
        )
        if removed:
            self.events.emit("replay_experiences_pruned", count=removed)

        if self.config.preserve_rollout_groups:
            admitted_groups: dict[str, list[tuple[Experience, float]]] = {}
            for exp, freshness in zip(admitted, admitted_freshness, strict=True):
                stored_group = exp.attempt.metadata.get("vare_rollout_group")
                if stored_group is None:
                    stored_group = exp.attempt.task.metadata.get("vare_rollout_group")
                key = str(stored_group) if stored_group is not None else f"ungrouped:{id(exp)}"
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
                    experiences = [exp for exp, _ in members]
                    if not complete:
                        reason = "incomplete_after_lag"
                    elif not self.replay.group_has_homogeneous_provenance(experiences):
                        reason = "heterogeneous_provenance"
                    else:
                        reason = "capacity_or_priority"
                    self.events.emit(
                        "replay_group_not_admitted",
                        group_id=group_id,
                        admitted_count=len(members),
                        expected_size=expected_size,
                        reason=reason,
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

        train_batch = self.replay.sample_current(
            self.config.replay_batch_size,
            freshness_fn=current_freshness,
            grouped=self.config.preserve_rollout_groups,
        )
        if screened_stale:
            self.events.emit("replay_experiences_screened", count=screened_stale)
        if verified_experiences is not None and {exp.attempt.metadata.get("trajectory_id") for exp in train_batch} != {exp.attempt.metadata.get("trajectory_id") for exp in verified_experiences}:
            raise RuntimeError("streamed batch was filtered or truncated before training")
        if not train_batch:
            candidate_id = incumbent_id
        else:
            # Authorization occurs at consumption time, after replay
            # resampling and freshness checks. A failed group member denies
            # the entire batch before any optimizer mutation.
            if self.training_evidence_authorizer is not None:
                try:
                    validate_complete_rvl_reward_batch(train_batch)
                    authorizations = [
                        self.training_evidence_authorizer(exp) for exp in train_batch
                    ]
                except Exception as exc:
                    self.events.emit("training_reward_evidence_denied", count=len(train_batch))
                    raise RuntimeError("training_reward_evidence_unverifiable") from exc
                if any(type(ok) is not bool or not ok for ok in authorizations):
                    self.events.emit("training_reward_evidence_denied", count=len(train_batch))
                    raise RuntimeError("unauthenticated_training_reward_evidence")
            if self.safety_interlock is not None:
                self.safety_interlock.require("train", incumbent_id=incumbent_id)
            # A failing training hook may have mutated model/optimizer state
            # without returning a candidate handle. Fence rather than guess.
            self._candidate_unresolved = True
            freshness_train = getattr(
                self.hooks, "train_candidate_with_freshness", None
            )
            if callable(freshness_train):
                candidate_id = await freshness_train(
                    incumbent_id,
                    train_batch,
                    freshness_assessor,
                )
            else:
                candidate_id = await self.hooks.train_candidate(incumbent_id, train_batch)

        evaluation_tasks: list[asyncio.Task] = []
        evaluation_group: asyncio.Future | None = None
        try:
            # Check authority before scheduling any evaluation work. Recovery
            # also owns asynchronous discard if this preflight check fails.
            if self.safety_interlock is not None:
                self.safety_interlock.require("evaluate", incumbent_id=incumbent_id)
            evaluation_tasks = [
                asyncio.create_task(self.hooks.evaluate(incumbent_id)),
                asyncio.create_task(self.hooks.evaluate(candidate_id)),
            ]
            evaluation_group = asyncio.gather(*evaluation_tasks)
            # Own the evaluation lifetime: caller cancellation must not cancel
            # a hook repeatedly while it is unwinding its own resources.
            incumbent_eval, candidate_eval = await asyncio.shield(evaluation_group)
            if self.safety_interlock is not None:
                self.safety_interlock.require("evaluate", incumbent_id=incumbent_id)
            if self.require_current_evaluation_verifier:
                active_verifier_version = getattr(self.verifier, "active_version", None)
                if type(active_verifier_version) is not int:
                    decision = PromotionDecision(
                        accepted=False,
                        reasons=("evaluation_verifier_version_mismatch",),
                        primary_gain=0.0,
                        worst_slice_regression=0.0,
                    )
                elif (
                    incumbent_eval.policy_id != incumbent_id
                    or candidate_eval.policy_id != candidate_id
                ):
                    decision = PromotionDecision(
                        accepted=False,
                        reasons=("evaluation_policy_mismatch",),
                        primary_gain=0.0,
                        worst_slice_regression=0.0,
                    )
                else:
                    decision = self.promotion.decide(
                        incumbent_eval, candidate_eval,
                        expected_verifier_version=active_verifier_version,
                    )
            elif (
                incumbent_eval.policy_id != incumbent_id
                or candidate_eval.policy_id != candidate_id
            ):
                decision = PromotionDecision(
                    accepted=False,
                    reasons=("evaluation_policy_mismatch",),
                    primary_gain=0.0,
                    worst_slice_regression=0.0,
                )
            else:
                decision = self.promotion.decide(incumbent_eval, candidate_eval)
            if verified_experiences is not None:
                assert self.training_evidence_authorizer is not None
                if any(self.training_evidence_authorizer(exp) is not True for exp in train_batch):
                    decision = PromotionDecision(False, ("streamed_training_evidence_changed",),
                                                 decision.primary_gain, decision.worst_slice_regression)
            if decision.accepted and candidate_id != incumbent_id and self.promotion_journal is not None:
                assert self.evidence_trust_root is not None
                assert self.checkpoint_hasher is not None
                expected_old = self.evidence_trust_root.checkpoint_digest(incumbent_eval)
                expected_new = self.evidence_trust_root.checkpoint_digest(candidate_eval)
                if (
                    expected_old is None or expected_new is None
                    or expected_old != self.checkpoint_hasher(incumbent_id)
                    or expected_new != self.checkpoint_hasher(candidate_id)
                ):
                    decision = PromotionDecision(
                        accepted=False,
                        reasons=("checkpoint_attestation_mismatch",),
                        primary_gain=decision.primary_gain,
                        worst_slice_regression=decision.worst_slice_regression,
                        paired_gain=decision.paired_gain,
                        paired_lcb=decision.paired_lcb,
                        paired_n=decision.paired_n,
                    )
            if candidate_id != incumbent_id and decision.accepted and self.safety_interlock is not None:
                try:
                    self.safety_interlock.require(
                        "promote", incumbent_id=incumbent_id, candidate_id=candidate_id
                    )
                except SafetyInterlockDenied:
                    decision = PromotionDecision(
                        accepted=False,
                        reasons=("safety_interlock_denied",),
                        primary_gain=decision.primary_gain,
                        worst_slice_regression=decision.worst_slice_regression,
                        paired_gain=decision.paired_gain,
                        paired_lcb=decision.paired_lcb,
                        paired_n=decision.paired_n,
                    )
        except BaseException:
            async def rollback_evaluation() -> None:
                # gather propagates the first error without stopping siblings.
                # Cancel each live evaluator once, then wait for all hook
                # cleanup before deleting a snapshot it may still be using.
                # A hook already cancelling may be inside asynchronous cleanup.
                for task in evaluation_tasks:
                    if not task.done() and not task.cancelling():
                        task.cancel()
                await asyncio.gather(*evaluation_tasks, return_exceptions=True)
                if evaluation_group is not None and not evaluation_group.cancelled():
                    evaluation_group.exception()  # Retrieve secondary/group errors.
                if candidate_id != incumbent_id:
                    await self._discard_candidate(candidate_id, propagate_cancellation=False)

            recovery = asyncio.create_task(rollback_evaluation())
            while not recovery.done():
                try:
                    await asyncio.shield(recovery)
                except asyncio.CancelledError:
                    continue
            recovery.result()
            raise
        deferred_cancellation_count = 0
        if candidate_id != incumbent_id and decision.accepted:
            transaction_id: str | None = None
            if verified_experiences is not None:
                assert publication_reserve is not None
                try:
                    await publication_reserve()
                except BaseException:
                    await self._discard_candidate(candidate_id, propagate_cancellation=False)
                    raise
            if self.promotion_journal is not None:
                assert self.checkpoint_hasher is not None
                assert self.evidence_trust_root is not None
                old_digest = self.evidence_trust_root.checkpoint_digest(incumbent_eval)
                new_digest = self.evidence_trust_root.checkpoint_digest(candidate_eval)
                assert old_digest is not None and new_digest is not None
                transaction_id = self.evidence_trust_root.promotion_nonce
                assert transaction_id is not None
                try:
                    # The check and durable prepare precede any live policy
                    # mutation. A nonce can be consumed only once.
                    if (self.checkpoint_hasher(incumbent_id) != old_digest
                            or self.checkpoint_hasher(candidate_id) != new_digest):
                        raise PromotionRecoveryRequired("checkpoint changed before prepare")
                    self.promotion_journal.prepare(
                        tx_id=transaction_id,
                        incumbent_id=incumbent_id,
                        incumbent_sha256=old_digest,
                        candidate_id=candidate_id,
                        candidate_sha256=new_digest,
                        evaluation_sha256=sha256_json({
                            "incumbent": incumbent_eval.metadata,
                            "candidate": candidate_eval.metadata,
                        }),
                    )
                except BaseException:
                    await self._discard_candidate(candidate_id, propagate_cancellation=False)
                    raise

            if self.decision_ledger is not None:
                try:
                    self.decision_ledger.append(
                        "promotion_intent",
                        {"round_index": round_index, "incumbent_id": incumbent_id,
                         "candidate_id": candidate_id, "primary_gain": decision.primary_gain},
                    )
                except BaseException:
                    # Promotion has not started. Restore the champion and, if
                    # independently verified, mark this intent aborted.
                    await self._discard_candidate(candidate_id, propagate_cancellation=False)
                    if transaction_id is not None:
                        assert self.promotion_journal is not None
                        assert self.checkpoint_hasher is not None
                        self.promotion_journal.reconcile(
                            tx_id=transaction_id,
                            observed_policy_id=self.hooks.active_policy()[0],
                            observed_checkpoint_sha256=self.checkpoint_hasher(incumbent_id),
                        )
                    raise

            # Publication may mutate the runtime before raising or being
            # cancelled. Never discard a possibly installed candidate or reuse
            # this loop after an unverified publication, even without a journal.
            try:
                deferred_cancellation_count = await self._publish_candidate(
                    candidate_id, defer_caller_cancellation=transaction_id is not None
                )
                if self.hooks.active_policy()[0] != candidate_id:
                    raise PromotionRecoveryRequired(
                        "post-promotion policy acknowledgment mismatch; manual recovery required"
                    )
                if transaction_id is not None:
                    assert self.promotion_journal is not None
                    assert self.checkpoint_hasher is not None
                    assert self.evidence_trust_root is not None
                    expected_new = self.evidence_trust_root.checkpoint_digest(candidate_eval)
                    if self.checkpoint_hasher(candidate_id) != expected_new:
                        raise PromotionRecoveryRequired(
                            "post-promotion checkpoint mismatch; manual recovery required"
                        )
                    self.promotion_journal.commit(transaction_id)
            except BaseException as exc:
                self._promotion_failure = exc
                if isinstance(exc, PromotionRecoveryRequired):
                    raise
                if isinstance(exc, asyncio.CancelledError) and transaction_id is None:
                    raise  # Preserve normal caller cancellation semantics.
                raise PromotionRecoveryRequired(
                    "promotion failed; verify the runtime and reconcile durable evidence"
                ) from exc
            promoted_id = candidate_id
            if deferred_cancellation_count:
                self.events.emit(
                    "promotion_cancellation_deferred",
                    candidate_id=candidate_id,
                    count=deferred_cancellation_count,
                )
            self.events.emit("candidate_promoted", candidate_id=candidate_id, gain=decision.primary_gain)
        else:
            promoted_id = incumbent_id
            if candidate_id != incumbent_id:
                await self._discard_candidate(candidate_id)
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
            deferred_cancellation_count=deferred_cancellation_count,
        )
