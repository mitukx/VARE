from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable, Sequence

from ..types import Attempt, EvaluationReport, Experience, Task


ScoreFn = Callable[[Task, str], float]


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
    ) -> None:
        self.backend = backend
        self.trainer = trainer
        self.eval_tasks = list(eval_tasks)
        self.score_fn = score_fn
        self.config = config or RVLGRPOConfig()
        if self.config.max_backend_concurrency <= 0:
            raise ValueError("max_backend_concurrency must be positive")
        self._model_lock = asyncio.Lock()
        self._rollout_sem = asyncio.Semaphore(self.config.max_backend_concurrency)
        self._active_id = "policy-0"
        self._version = 0
        self._counter = 1
        self._states: dict[str, Any] = {self._active_id: self.trainer.snapshot_training_state()}
        self._loaded_id = self._active_id
        self._generation_factory = generation_factory
        self._verified_factory = verified_generation_factory
        if self._generation_factory is None or self._verified_factory is None:
            try:
                from src.rvl_systems.types import Generation, VerifiedGeneration
            except ImportError as exc:
                raise RuntimeError(
                    "RVLGRPOHooks requires an importable Recursive-Verification-Lag checkout "
                    "or explicit generation/verified-generation factories"
                ) from exc
            self._generation_factory = self._generation_factory or Generation
            self._verified_factory = self._verified_factory or VerifiedGeneration

    def active_policy(self) -> tuple[str, int]:
        return self._active_id, self._version

    def _restore(self, policy_id: str) -> None:
        if policy_id not in self._states:
            raise KeyError(policy_id)
        if self._loaded_id != policy_id:
            # A restore may partially mutate the shared trainer before it
            # raises. Invalidate first so recovery cannot mistake that partial
            # state for the previously loaded policy and skip a forced restore.
            self._loaded_id = None
            self.trainer.restore_training_state(self._states[policy_id])
            self._loaded_id = policy_id

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
                rows = await self.backend.generate(
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
        raw = exp.attempt.metadata.get("rvl_generation")
        if not isinstance(raw, dict):
            raise ValueError("experience lacks token-exact rvl_generation metadata")
        generation = self._generation_factory(**copy.deepcopy(raw))
        return self._verified_factory(
            generation=generation,
            reward=float(exp.verification.score),
            verifier_latency_s=float(exp.verification.metadata.get("latency_s", 0.0)),
            verifier_version=int(exp.verification.verifier_version),
            metadata={
                **copy.deepcopy(exp.verification.metadata),
                "vare_disagreement": exp.verification.disagreement,
                "vare_trusted": exp.verification.trusted,
            },
        )

    async def train_candidate(self, incumbent_id: str, experiences: Sequence[Experience]) -> str:
        if incumbent_id != self._active_id:
            raise ValueError("candidate must branch from active incumbent")
        verified = [self._to_verified_generation(x) for x in experiences]
        async with self._model_lock:
            self._restore(incumbent_id)
            # Keep an immutable pre-update state even if a custom trainer mutates
            # nested optimizer structures during train_step.
            self._states[incumbent_id] = self.trainer.snapshot_training_state()
            candidate_id = f"policy-{self._counter}"
            # Invalidate the loaded identity before the first in-place mutation.
            # If training or candidate snapshotting raises, rollback must still
            # restore the saved incumbent rather than trusting a stale ID.
            self._loaded_id = None
            try:
                self.trainer.train_step(verified)
                self._states[candidate_id] = self.trainer.snapshot_training_state()
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
                    rows = await self.backend.generate(
                        task.id,
                        task.prompt,
                        n=1,
                        temperature=self.config.eval_temperature,
                        seed=self.config.seed + 1_000_000 + index,
                    )
                    if len(rows) != 1:
                        raise RuntimeError("RVL backend returned unexpected evaluation generation count")
                    row = rows[0]
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
