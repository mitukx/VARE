from __future__ import annotations

import asyncio
import copy
import math
import random
from dataclasses import dataclass
from typing import Sequence

from .types import Attempt, EvaluationReport, Experience, Task, Verification


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _features(task: Task) -> tuple[float, float, float, float]:
    x1 = float(task.metadata["x1"])
    x2 = float(task.metadata["x2"])
    return (1.0, x1, x2, x1 * x2)


def _truth(task: Task) -> int:
    x1 = float(task.metadata["x1"])
    x2 = float(task.metadata["x2"])
    return int(0.2 + 1.2 * x1 - 0.9 * x2 + 1.4 * x1 * x2 > 0)


@dataclass
class ToyPolicy:
    weights: list[float]

    def probability(self, task: Task) -> float:
        return _sigmoid(sum(w * x for w, x in zip(self.weights, _features(task))))


class ToyTrustedVerifier:
    name = "toy-oracle"
    trusted = True

    def __init__(self, version: int = 1) -> None:
        self.version = version

    async def verify(self, attempt: Attempt) -> Verification:
        correct = int(attempt.output) == _truth(attempt.task)
        return Verification(score=float(correct), passed=correct, confidence=1.0, verifier_version=self.version, verifier_name=self.name, trusted=True)


class ToyHooks:
    def __init__(self, seed: int = 0, eval_n: int = 512, learning_rate: float = 0.12) -> None:
        self.rng = random.Random(seed)
        self.eval_n = eval_n
        self.learning_rate = learning_rate
        self.version = 0
        self.active_id = "policy-0"
        self.policies: dict[str, ToyPolicy] = {self.active_id: ToyPolicy([0.0, 0.0, 0.0, 0.0])}
        self.eval_tasks = make_tasks(eval_n, seed=seed + 10_000)

    def active_policy(self) -> tuple[str, int]:
        return self.active_id, self.version

    async def rollout(self, task: Task, policy_id: str, policy_version: int, step: int) -> Attempt:
        p = self.policies[policy_id].probability(task)
        action = int(self.rng.random() < p)
        logprob = math.log(max(1e-9, p if action else 1.0 - p))
        return Attempt(task=task, output=str(action), policy_id=policy_id, policy_version=policy_version, created_step=step, logprob=logprob, metadata={"p1": p, "action": action})

    async def train_candidate(self, incumbent_id: str, experiences: Sequence[Experience]) -> str:
        policy = copy.deepcopy(self.policies[incumbent_id])
        # Contextual-bandit REINFORCE with a 0.5 baseline. Priority is used only for sampling upstream,
        # avoiding a second uncalibrated importance correction here.
        for exp in experiences:
            a = int(exp.attempt.output)
            p = float(exp.attempt.metadata["p1"])
            advantage = exp.verification.score - 0.5
            grad_logp = a - p
            for i, x in enumerate(_features(exp.attempt.task)):
                policy.weights[i] += self.learning_rate * advantage * grad_logp * x
        candidate_id = f"policy-{len(self.policies)}"
        self.policies[candidate_id] = policy
        return candidate_id

    async def evaluate(self, policy_id: str) -> EvaluationReport:
        policy = self.policies[policy_id]
        correct = 0
        slices = {"quadrant_pos": [0, 0], "quadrant_neg": [0, 0]}
        per_task_scores: dict[str, float] = {}
        for t in self.eval_tasks:
            pred = int(policy.probability(t) >= 0.5)
            ok = pred == _truth(t)
            correct += int(ok)
            per_task_scores[t.id] = float(ok)
            key = "quadrant_pos" if float(t.metadata["x1"]) * float(t.metadata["x2"]) >= 0 else "quadrant_neg"
            slices[key][0] += int(ok)
            slices[key][1] += 1
        return EvaluationReport(
            policy_id=policy_id,
            primary=correct / len(self.eval_tasks),
            slices={k: c / n for k, (c, n) in slices.items()},
            cost=1.0,
            verifier_disagreement=0.0,
            n=len(self.eval_tasks),
            metadata={"per_task_scores": per_task_scores},
        )

    async def promote(self, candidate_id: str) -> None:
        self.active_id = candidate_id
        self.version += 1

    async def discard(self, candidate_id: str) -> None:
        if candidate_id != self.active_id:
            self.policies.pop(candidate_id, None)


def make_tasks(n: int, seed: int = 0) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    for i in range(n):
        x1, x2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        family = "interaction" if abs(x1 * x2) > 0.35 else "linear"
        tasks.append(Task(id=f"toy-{seed}-{i}", prompt=f"Choose action 0 or 1 for x1={x1:.4f}, x2={x2:.4f}", family=family, metadata={"x1": x1, "x2": x2}))
    return tasks


async def run_demo(rounds: int = 8, rollouts: int = 512, seed: int = 7):
    from .config import EngineConfig, PromotionConfig
    from .engine import CapabilityLoop
    from .verifiers import VerifierEnsemble, VerifierMember

    hooks = ToyHooks(seed=seed)
    verifier = VerifierEnsemble([VerifierMember(ToyTrustedVerifier())])
    cfg = EngineConfig(promotion=PromotionConfig(min_primary_gain=0.002, max_slice_regression=0.08, min_eval_examples=128))
    loop = CapabilityLoop(hooks=hooks, verifier=verifier, config=cfg, seed=seed)
    pool = make_tasks(max(1024, rollouts), seed=seed + 1)
    results = []
    for r in range(rounds):
        results.append(await loop.run_round(pool, round_index=r, rollout_count=rollouts))
    return results
