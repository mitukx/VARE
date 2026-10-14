from __future__ import annotations

import asyncio
import copy
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
import math
from statistics import fmean
from typing import Any, Callable, Iterator, Sequence

from ..reward_evidence import validate_complete_rvl_reward_batch
from ..types import Attempt, EvaluationReport, Experience, Task, snapshot_verification


ScoreFn = Callable[[Task, str], float]


@dataclass(frozen=True, slots=True)
class _TrainerSnapshot:
    training_state: Any
    module_modes: dict[str, bool]
    module_types: dict[str, type[Any]]


@dataclass(frozen=True, slots=True)
class RVLGRPOConfig:
    train_temperature: float = 0.8
    eval_temperature: float = 0.0
    seed: int = 20261008
    max_backend_concurrency: int = 1


class RVLGRPOHooks:
    """Transactional VARE hooks over RVL's HFLocalBackend + GRPO trainer.

    The adapter intentionally uses RVL's token-exact generation metadata and
    HFCausalLMGRPOTrainer snapshots. A candidate update is never the active
    policy until the VARE promotion gate accepts it. Evaluations are serialized
    because incumbent and candidate share one mutable model instance.

    ``verified_generation_factory`` is injectable for tests; in a real RVL
    checkout it is discovered from ``src.rvl_systems.types``.
    """

    def __init__(
        self,
        *,
        backend: Any,
        trainer: Any,
        eval_tasks: Sequence[Task],
        score_fn: ScoreFn,
        config: RVLGRPOConfig | None = None,
        verified_generation_factory: Callable[..., Any] | None = None,
        generation_factory: Callable[..., Any] | None = None,
        training_evidence_authorizer: Callable[[Experience], bool] | None = None,
        require_authenticated_training: bool = False,
    ) -> None:
        self.backend = backend
        self.trainer = trainer
        self.eval_tasks = list(eval_tasks)
        eval_task_ids = [task.id for task in self.eval_tasks]
        if len(set(eval_task_ids)) != len(eval_task_ids):
            raise ValueError("evaluation task IDs must be unique for per-task scoring")
        self.score_fn = score_fn
        self.config = config or RVLGRPOConfig()
        if self.config.max_backend_concurrency <= 0:
            raise ValueError("max_backend_concurrency must be positive")
        if type(require_authenticated_training) is not bool:
            raise ValueError("require_authenticated_training must be bool")
        if require_authenticated_training and training_evidence_authorizer is None:
            raise ValueError(
                "authenticated training requires a trusted evidence authorizer"
            )
        self.training_evidence_authorizer = training_evidence_authorizer
        self.require_authenticated_training = require_authenticated_training
        self._model_lock = asyncio.Lock()
        self._rollout_sem = asyncio.Semaphore(self.config.max_backend_concurrency)
        self._active_id = "policy-0"
        self._version = 0
        self._counter = 1
        self._states: dict[str, Any] = {self._active_id: self._snapshot_training_state()}
        self._loaded_id: str | None = self._active_id
        resolved_generation = generation_factory
        resolved_verified = verified_generation_factory
        if resolved_generation is None or resolved_verified is None:
            try:
                from src.rvl_systems.types import Generation, VerifiedGeneration
            except ImportError as exc:
                raise RuntimeError(
                    "RVLGRPOHooks requires an importable Recursive-Verification-Lag checkout "
                    "or explicit generation/verified-generation factories"
                ) from exc
            if resolved_generation is None:
                resolved_generation = Generation
            if resolved_verified is None:
                resolved_verified = VerifiedGeneration
        self._generation_factory: Callable[..., Any] = resolved_generation
        self._verified_factory: Callable[..., Any] = resolved_verified

    def active_policy(self) -> tuple[str, int]:
        return self._active_id, self._version

    def _snapshot_training_state(self) -> Any:
        state = self.trainer.snapshot_training_state()
        model = getattr(self.trainer, "model", None)
        named_modules = getattr(model, "named_modules", None)
        if not callable(named_modules):
            return state
        modules = list(named_modules())
        if not modules or any(
            not hasattr(module, "training") or not callable(getattr(module, "train", None))
            for _, module in modules
        ):
            return state
        return _TrainerSnapshot(
            training_state=state,
            module_modes={name: bool(module.training) for name, module in modules},
            module_types={name: type(module) for name, module in modules},
        )

    def _restore_training_state(self, snapshot: Any) -> None:
        if not isinstance(snapshot, _TrainerSnapshot):
            self.trainer.restore_training_state(snapshot)
            return
        model = getattr(self.trainer, "model", None)
        named_modules = getattr(model, "named_modules", None)
        if not callable(named_modules):
            raise RuntimeError("trainer model no longer exposes module modes")
        modules = dict(named_modules())
        if (
            modules.keys() != snapshot.module_modes.keys()
            or modules.keys() != snapshot.module_types.keys()
        ):
            raise RuntimeError("trainer module topology changed since policy snapshot")
        for name, module in modules.items():
            if type(module) is not snapshot.module_types[name]:
                raise RuntimeError(
                    f"trainer module {name!r} class changed since policy snapshot"
                )
        self.trainer.restore_training_state(snapshot.training_state)
        # Assign the captured flags directly. Module.train() and Module.eval()
        # recurse through every parent edge, while named_modules() lists a
        # shared module once. Calling train() on each named module would still
        # overwrite a shared child when a later alias has a different mode.
        # The snapshot stores only the boolean, so a custom train() override's
        # other side effects are not replayed.
        for name, module in modules.items():
            module.training = snapshot.module_modes[name]

    def _restore(self, policy_id: str) -> None:
        if policy_id not in self._states:
            raise KeyError(policy_id)
        if self._loaded_id != policy_id:
            # A restore may partially mutate the shared trainer before it
            # raises. Invalidate first so recovery cannot mistake that partial
            # state for the previously loaded policy and skip a forced restore.
            self._loaded_id = None
            self._restore_training_state(self._states[policy_id])
            self._loaded_id = policy_id

    @contextmanager
    def _preserve_inference_state(self) -> Iterator[None]:
        """Preserve flags and CPU/initialized-CUDA RNG, without tensor copies.

        Backend inference must leave tensors unchanged. The adapter's lock
        serializes access to this model, not unrelated process-global RNG users.
        Other device RNGs and scorer side effects are outside this contract.
        """
        model = getattr(self.trainer, "model", None)
        named_modules = getattr(model, "named_modules", None)
        modes = [
            (module, bool(module.training)) for _, module in named_modules()
        ] if callable(named_modules) else []
        torch = getattr(self.trainer, "torch", None)
        rng_scope: AbstractContextManager[Any] = nullcontext()
        if torch is not None:
            # Do not initialize CUDA merely to snapshot its generators.
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
            rng_scope = torch.random.fork_rng(devices=devices)
        rng_scope.__enter__()
        try:
            yield
        finally:
            try:
                try:
                    rng_scope.__exit__(None, None, None)
                finally:
                    for module, mode in modes:
                        module.training = mode
            except BaseException:
                self._loaded_id = None
                raise

    async def _generate(self, *args: Any, **kwargs: Any) -> Any:
        # Called only under _model_lock. Cancelling an asyncio.to_thread await
        # does not stop its worker. Shield and drain before restoring state or
        # releasing the lock, including when the caller cancels repeatedly.
        with self._preserve_inference_state():
            generation = asyncio.ensure_future(self.backend.generate(*args, **kwargs))
            try:
                return await asyncio.shield(generation)
            except asyncio.CancelledError:
                while not generation.done():
                    try:
                        await asyncio.shield(generation)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
                if not generation.cancelled():
                    generation.exception()  # Retrieve a worker error during cancellation.
                raise

    @staticmethod
    def _generation_fields(generation: Any) -> dict[str, Any]:
        return {
            "prompt_id": str(generation.prompt_id),
            "prompt": str(generation.prompt),
            "response": str(generation.response),
            "logprob": float(generation.logprob),
            "token_count": int(generation.token_count),
            "latency_s": float(generation.latency_s),
            "metadata": copy.deepcopy(dict(generation.metadata)),
        }

    async def rollout(self, task: Task, policy_id: str, policy_version: int, step: int) -> Attempt:
        if policy_id != self._active_id:
            raise ValueError("online rollouts must use the active policy")
        async with self._rollout_sem:
            async with self._model_lock:
                self._restore(policy_id)
                rows = await self._generate(
                    task.id,
                    task.prompt,
                    n=1,
                    temperature=self.config.train_temperature,
                    seed=self.config.seed + step,
                )
        if len(rows) != 1:
            raise RuntimeError("RVL backend returned unexpected generation count")
        generation = rows[0]
        g = self._generation_fields(generation)
        if g["prompt_id"] != task.id or g["prompt"] != task.prompt:
            raise ValueError("RVL generation task identity mismatch")
        return Attempt(
            task=task,
            output=g["response"],
            policy_id=policy_id,
            policy_version=policy_version,
            created_step=step,
            logprob=g["logprob"],
            latency_ms=1000.0 * g["latency_s"],
            metadata={
                "rvl_generation": g,
                "token_count": g["token_count"],
                "sampling_temperature": self.config.train_temperature,
            },
        )

    def _to_verified_generation(self, exp: Experience) -> Any:
        verification = snapshot_verification(
            exp.verification, context="RVL trainer experience verification"
        )
        raw = exp.attempt.metadata.get("rvl_generation")
        if not isinstance(raw, dict):
            raise ValueError("experience lacks token-exact rvl_generation metadata")
        generation = self._generation_factory(**copy.deepcopy(raw))
        return self._verified_factory(
            generation=generation,
            reward=verification.score,
            verifier_latency_s=float(verification.metadata.get("latency_s", 0.0)),
            verifier_version=verification.verifier_version,
            metadata={
                **copy.deepcopy(verification.metadata),
                "vare_disagreement": verification.disagreement,
                "vare_trusted": verification.trusted,
            },
        )

    @staticmethod
    def _group_relative_advantages(
        experiences: Sequence[Experience],
        verified: Sequence[Any],
        *,
        eps: float,
        clip: float,
    ) -> list[float] | None:
        """Normalize RVL rewards by VARE's declared rollout groups.

        RVL's default helper groups by ``Generation.prompt_id``. VARE may draw
        the same task more than once in a round, so distinct complete groups can
        share that ID. When VARE group metadata is present, pass explicit
        group-relative advantages to RVL's supported ``train_step`` override.
        """
        group_members: dict[str, list[tuple[int, float]]] = {}
        expected_sizes: dict[str, int] = {}
        missing_group = False
        for index, (exp, sample) in enumerate(zip(experiences, verified, strict=True)):
            group_id = exp.attempt.metadata.get("vare_rollout_group")
            if group_id is None:
                group_id = exp.attempt.task.metadata.get("vare_rollout_group")
            if group_id is None:
                missing_group = True
                continue
            if not isinstance(group_id, str) or not group_id:
                raise ValueError("VARE rollout group ID must be a nonempty string")

            expected_size = exp.attempt.metadata.get("vare_rollout_group_size")
            if expected_size is None:
                expected_size = exp.attempt.task.metadata.get(
                    "vare_rollout_group_size"
                )
            if type(expected_size) is not int or expected_size <= 0:
                raise ValueError("VARE rollout group size must be a positive integer")
            previous_size = expected_sizes.setdefault(group_id, expected_size)
            if previous_size != expected_size:
                raise ValueError("VARE rollout group has inconsistent declared sizes")
            group_members.setdefault(group_id, []).append((index, float(sample.reward)))

        if not group_members:
            return None
        if missing_group:
            raise ValueError("cannot mix grouped and ungrouped experiences in one GRPO batch")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("RVL advantage epsilon must be finite and positive")
        if not math.isfinite(clip) or clip <= 0.0:
            raise ValueError("RVL advantage clip must be finite and positive")

        advantages = [0.0] * len(verified)
        for group_id, members in group_members.items():
            if len(members) != expected_sizes[group_id]:
                raise ValueError(
                    f"incomplete VARE rollout group {group_id!r}: "
                    f"got {len(members)}, expected {expected_sizes[group_id]}"
                )
            rewards = [reward for _, reward in members]
            mean = fmean(rewards)
            variance = fmean((reward - mean) ** 2 for reward in rewards)
            scale = math.sqrt(variance + eps)
            for index, reward in members:
                advantages[index] = max(-clip, min(clip, (reward - mean) / scale))
        return advantages

    async def train_candidate(self, incumbent_id: str, experiences: Sequence[Experience]) -> str:
        if incumbent_id != self._active_id:
            raise ValueError("candidate must branch from active incumbent")
        if not experiences:
            raise ValueError("candidate training batch cannot be empty")
        # Keep this check at the hook boundary as well as in CapabilityLoop:
        # callers can use RVLGRPOHooks directly, and in that case there is no
        # outer-loop admission check. A configured authorizer always applies;
        # the strict flag additionally makes omitting one a construction error.
        if self.training_evidence_authorizer is not None:
            try:
                validate_complete_rvl_reward_batch(experiences)
                authorizations = [
                    self.training_evidence_authorizer(experience)
                    for experience in experiences
                ]
            except Exception as exc:
                raise RuntimeError("training_reward_evidence_unverifiable") from exc
            if any(type(authorized) is not bool or not authorized for authorized in authorizations):
                raise RuntimeError("unauthenticated_training_reward_evidence")
        elif self.require_authenticated_training:
            # This is defensive; the constructor already rejects this state.
            raise RuntimeError("unauthenticated_training_reward_evidence")
        verified = [self._to_verified_generation(x) for x in experiences]
        train_config = getattr(self.trainer, "config", None)
        advantages = self._group_relative_advantages(
            experiences,
            verified,
            eps=float(getattr(train_config, "advantage_eps", 1e-6)),
            clip=float(getattr(train_config, "clip_advantage", 5.0)),
        )
        async with self._model_lock:
            self._restore(incumbent_id)
            # Keep an immutable pre-update state even if a custom trainer mutates
            # nested optimizer structures during train_step.
            self._states[incumbent_id] = self._snapshot_training_state()
            candidate_id = f"policy-{self._counter}"
            # Invalidate the loaded identity before the first in-place mutation.
            # If training or candidate snapshotting raises, rollback must still
            # restore the saved incumbent rather than trusting a stale ID.
            self._loaded_id = None
            try:
                if advantages is None:
                    self.trainer.train_step(verified)
                else:
                    self.trainer.train_step(verified, advantages=advantages)
                self._states[candidate_id] = self._snapshot_training_state()
                # The candidate now describes the runtime; restore the champion
                # before releasing the lock to online rollout/evaluation code.
                self._loaded_id = candidate_id
                self._restore(incumbent_id)
            except BaseException:
                self._states.pop(candidate_id, None)
                # Force restoration even when an error happened after assigning
                # candidate_id. _restore updates this marker only after the
                # trainer confirms that restoration completed.
                self._loaded_id = None
                try:
                    self._restore(incumbent_id)
                except BaseException as restore_exc:
                    raise RuntimeError(
                        "candidate training failed and incumbent rollback failed; "
                        "trainer state is unknown"
                    ) from restore_exc
                raise
            self._counter += 1
        return candidate_id

    async def evaluate(self, policy_id: str) -> EvaluationReport:
        async with self._model_lock:
            scores: dict[str, float] = {}
            latencies: list[float] = []
            families: dict[str, list[float]] = {}
            try:
                self._restore(policy_id)
                for index, task in enumerate(self.eval_tasks):
                    rows = await self._generate(
                        task.id,
                        task.prompt,
                        n=1,
                        temperature=self.config.eval_temperature,
                        seed=self.config.seed + 1_000_000 + index,
                    )
                    if len(rows) != 1:
                        raise RuntimeError("RVL backend returned unexpected evaluation generation count")
                    row = rows[0]
                    if str(row.prompt_id) != task.id or str(row.prompt) != task.prompt:
                        raise ValueError("RVL evaluation generation task identity mismatch")
                    score = float(self.score_fn(task, str(row.response)))
                    scores[task.id] = score
                    latencies.append(float(row.latency_s))
                    families.setdefault(task.family, []).append(score)
            except BaseException:
                try:
                    self._restore(self._active_id)
                except BaseException as restore_exc:
                    raise RuntimeError(
                        "evaluation failed and incumbent restoration failed; "
                        "trainer state is unknown"
                    ) from restore_exc
                raise
            try:
                # Return the shared runtime to the active champion before the
                # next operation observes it, including after failed evaluation.
                self._restore(self._active_id)
            except BaseException as restore_exc:
                raise RuntimeError(
                    "evaluation completed but incumbent restoration failed; "
                    "trainer state is unknown"
                ) from restore_exc
        return EvaluationReport(
            policy_id=policy_id,
            primary=fmean(scores.values()) if scores else 0.0,
            slices={name: fmean(values) for name, values in families.items()},
            cost=fmean(latencies) if latencies else 0.0,
            verifier_disagreement=0.0,
            n=len(scores),
            metadata={"per_task_scores": scores},
        )

    async def promote(self, candidate_id: str) -> None:
        async with self._model_lock:
            self._restore(candidate_id)
            old_id = self._active_id
            self._active_id = candidate_id
            self._version += 1
            # Drop the previous champion snapshot only after the candidate is
            # installed; no future VARE operation may silently roll back to it.
            if old_id != candidate_id:
                self._states.pop(old_id, None)

    async def discard(self, candidate_id: str) -> None:
        async with self._model_lock:
            self._states.pop(candidate_id, None)
            self._restore(self._active_id)
