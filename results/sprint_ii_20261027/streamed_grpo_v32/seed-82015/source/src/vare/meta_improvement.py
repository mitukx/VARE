"""CPU-only improvement-value learning; no language-model capability is implied.

A controller observes *development* outcome deltas from already completed updates.
It never sees the confirmation metric used to report the final result. Each
training intervention performs an actual expected-policy-gradient update on a
small tabular RLVR problem with an imperfect verifier.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Iterable


SKILLS = ("arithmetic", "logic", "tool")
DEV_WEIGHTS = (0.45, 0.35, 0.20)
CONFIRM_WEIGHTS = (0.35, 0.35, 0.30)
FEATURE_SIZE = 5


@dataclass(frozen=True, slots=True)
class Intervention:
    name: str
    skill: int
    cost: int
    repairs_verifier: bool = False


INTERVENTIONS = (
    Intervention("train:arithmetic", 0, 1),
    Intervention("train:logic", 1, 2),
    Intervention("train:tool", 2, 1),
    Intervention("repair-and-train:tool", 2, 3, True),
)


@dataclass(frozen=True, slots=True)
class Candidate:
    intervention: Intervention
    features: tuple[float, ...]
    policy_version: int
    verifier_version: int


@dataclass(frozen=True, slots=True)
class Selection:
    candidate: Candidate
    decision_id: str


@dataclass(frozen=True, slots=True)
class Feedback:
    decision_id: str
    action: str
    before_policy_version: int
    after_policy_version: int
    verifier_version: int
    evaluation_id: str
    development_before: float
    development_after: float
    cost: int

    @property
    def gain_per_cost(self) -> float:
        return (self.development_after - self.development_before) / self.cost


@dataclass(frozen=True, slots=True)
class Result:
    method: str
    seed: int
    budget: int
    cost_spent: int
    development_start: float
    development_end: float
    confirmation_start: float
    confirmation_end: float
    interventions: tuple[str, ...]
    feedback: tuple[Feedback, ...]
    confirmation_reads_during_training: int

    @property
    def confirmation_gain(self) -> float:
        return self.confirmation_end - self.confirmation_start


class LinearImprovementModel:
    """Per-intervention ridge regression with an explicit contextual UCB bonus.

    This is an online contextual bandit, not a trained deep meta-RL agent.
    Samples are retained only when they originate from a completed decision.
    """

    def __init__(self, ridge: float = 0.1, exploration: float = 0.005):
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be positive and finite")
        if not math.isfinite(exploration) or exploration < 0:
            raise ValueError("exploration must be non-negative and finite")
        self.ridge = ridge
        self.exploration = exploration
        self._gram: dict[str, list[list[float]]] = {}
        self._rhs: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}
        self._pending: Selection | None = None
        self._used_decision_ids: set[str] = set()
        self._used_evaluation_ids: set[str] = set()
        self._sequence = 0

    @staticmethod
    def _solve(a: list[list[float]], b: list[float]) -> list[float]:
        """Partial-pivot Gaussian elimination on the small regularized system."""
        n = len(b)
        aug = [list(a[i]) + [b[i]] for i in range(n)]
        for i in range(n):
            pivot = max(range(i, n), key=lambda k: abs(aug[k][i]))
            if abs(aug[pivot][i]) < 1e-12:
                raise ArithmeticError("singular Gram matrix")
            aug[i], aug[pivot] = aug[pivot], aug[i]
            scale = aug[i][i]
            for j in range(i, n + 1):
                aug[i][j] /= scale
            for k in range(n):
                if k == i:
                    continue
                factor = aug[k][i]
                for j in range(i, n + 1):
                    aug[k][j] -= factor * aug[i][j]
        return [aug[i][n] for i in range(n)]

    def _system(self, name: str) -> tuple[list[list[float]], list[float]]:
        if name not in self._gram:
            self._gram[name] = [
                [self.ridge if i == j else 0.0 for j in range(FEATURE_SIZE)]
                for i in range(FEATURE_SIZE)
            ]
            self._rhs[name] = [0.0] * FEATURE_SIZE
        return self._gram[name], self._rhs[name]

    def score(self, candidate: Candidate) -> float:
        x = candidate.features
        if len(x) != FEATURE_SIZE or any(not math.isfinite(v) for v in x):
            raise ValueError("invalid candidate features")
        a, b = self._system(candidate.intervention.name)
        theta = self._solve(a, b)
        inverse_x = self._solve(a, list(x))
        variance = max(0.0, sum(v * u for v, u in zip(x, inverse_x, strict=True)))
        return sum(v * w for v, w in zip(x, theta, strict=True)) + self.exploration * math.sqrt(variance)

    def choose(self, candidates: Iterable[Candidate], remaining_budget: int) -> Selection:
        if self._pending is not None:
            raise RuntimeError("complete or abort the pending intervention first")
        if type(remaining_budget) is not int or remaining_budget < 0:
            raise ValueError("remaining budget must be a non-negative integer")
        options = list(candidates)
        if len({c.intervention.name for c in options}) != len(options):
            raise ValueError("duplicate intervention")
        if any(c.intervention.cost <= 0 for c in options):
            raise ValueError("invalid intervention cost")
        feasible = [c for c in options if c.intervention.cost <= remaining_budget]
        if not feasible:
            raise ValueError("no feasible intervention")
        # Explore each available kind once to avoid treating an untried action as zero-valued.
        unseen = [c for c in feasible if self.counts.get(c.intervention.name, 0) == 0]
        if unseen:
            chosen = min(unseen, key=lambda c: (c.intervention.cost, c.intervention.name))
        else:
            chosen = max(feasible, key=lambda c: (self.score(c), -c.intervention.cost, c.intervention.name))
        self._sequence += 1
        label = f"{self._sequence}:{chosen.intervention.name}:{chosen.policy_version}:{chosen.verifier_version}"
        decision = Selection(chosen, hashlib.sha256(label.encode()).hexdigest())
        self._pending = decision
        return decision

    def abort(self, decision_id: str) -> None:
        if self._pending is None or self._pending.decision_id != decision_id:
            raise ValueError("not the pending decision")
        self._used_decision_ids.add(decision_id)
        self._pending = None

    def observe(self, feedback: Feedback) -> None:
        pending = self._pending
        if pending is None or pending.decision_id != feedback.decision_id:
            raise ValueError("feedback is not bound to the pending decision")
        c = pending.candidate
        if (feedback.decision_id in self._used_decision_ids
                or feedback.evaluation_id in self._used_evaluation_ids):
            raise ValueError("replayed feedback")
        if (feedback.action != c.intervention.name or feedback.cost != c.intervention.cost
                or feedback.before_policy_version != c.policy_version
                or feedback.after_policy_version != c.policy_version + 1
                or feedback.verifier_version != c.verifier_version + int(c.intervention.repairs_verifier)):
            raise ValueError("mismatched action or version provenance")
        if not feedback.evaluation_id or any(
            not math.isfinite(x) or not 0 <= x <= 1
            for x in (feedback.development_before, feedback.development_after)
        ):
            raise ValueError("invalid independent development evaluation")
        x = c.features
        a, b = self._system(c.intervention.name)
        target = feedback.gain_per_cost
        for i in range(FEATURE_SIZE):
            b[i] += x[i] * target
            for j in range(FEATURE_SIZE):
                a[i][j] += x[i] * x[j]
        name = c.intervention.name
        self.counts[name] = self.counts.get(name, 0) + 1
        self._used_decision_ids.add(feedback.decision_id)
        self._used_evaluation_ids.add(feedback.evaluation_id)
        self._pending = None


class TabularRLVR:
    """Three-skill Bernoulli policies updated with an expected REINFORCE step.

    For a good-action probability p and verifier true/false positive rates
    (tpr, fpr), the proxy-reward gradient is p(1-p)(tpr-fpr). If fpr > tpr,
    optimizing verifier reward can *reduce* actual task success.
    """

    def __init__(self, seed: int):
        rng = random.Random(seed)
        self.logits = [-0.65 + rng.uniform(-0.15, 0.15),
                       -1.0 + rng.uniform(-0.15, 0.15),
                       -0.85 + rng.uniform(-0.15, 0.15)]
        self.tpr = [0.95, 0.90, 0.69]
        self.fpr = [0.05, 0.12, 0.85]
        self.policy_version = 0
        self.verifier_version = 0
        self.repaired = False
        self.confirmation_reads = 0

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def probabilities(self) -> tuple[float, ...]:
        return tuple(self._sigmoid(x) for x in self.logits)

    def _weighted_score(self, weights: tuple[float, ...]) -> float:
        return sum(w * p for w, p in zip(weights, self.probabilities(), strict=True))

    def development(self) -> float:
        return self._weighted_score(DEV_WEIGHTS)

    def confirmation(self) -> float:
        # The controller and per-round trace never call this method.
        self.confirmation_reads += 1
        return self._weighted_score(CONFIRM_WEIGHTS)

    def candidates(self) -> tuple[Candidate, ...]:
        ps = self.probabilities()
        options = [action for action in INTERVENTIONS
                   if not (action.repairs_verifier and self.repaired)]
        result = []
        for a in options:
            p = ps[a.skill]
            proxy_accept = self.tpr[a.skill] * p + self.fpr[a.skill] * (1 - p)
            # Purely visible probe and proxy statistics; neither dev nor confirm
            # evaluation score appears here. No clean tpr/fpr is exposed.
            features = (1.0, p, p * (1 - p), proxy_accept,
                        float(self.repaired and a.skill == 2))
            result.append(Candidate(a, features, self.policy_version, self.verifier_version))
        return tuple(result)

    def apply(self, action: Intervention) -> None:
        if action not in INTERVENTIONS:
            raise ValueError("unknown intervention")
        if action.repairs_verifier:
            if self.repaired:
                raise ValueError("verifier was already repaired")
            self.tpr[2], self.fpr[2] = 0.94, 0.08
            self.repaired = True
            self.verifier_version += 1
        skill = action.skill
        p = self.probabilities()[skill]
        expected_gradient = p * (1 - p) * (self.tpr[skill] - self.fpr[skill])
        self.logits[skill] += 0.80 * expected_gradient
        self.policy_version += 1


def _choose_baseline(method: str, candidates: list[Candidate], episode: int,
                     model: LinearImprovementModel | None, budget: int,
                     world: TabularRLVR) -> Selection:
    if method == "value_ucb":
        assert model is not None
        return model.choose(candidates, budget)
    if method == "round_robin":
        action = candidates[episode % len(candidates)]
    elif method == "proxy_greedy":
        action = max(candidates, key=lambda c: c.features[3] / c.intervention.cost)
    elif method == "one_step_oracle":
        # Development-oracle diagnostic, not a deployable controller. It is the
        # only method allowed to inspect counterfactual development outcomes.
        scored = []
        for c in candidates:
            alt = TabularRLVR(0)
            alt.logits = list(world.logits)
            alt.tpr = list(world.tpr)
            alt.fpr = list(world.fpr)
            alt.repaired = world.repaired
            before = alt.development()
            alt.apply(c.intervention)
            scored.append(((alt.development() - before) / c.intervention.cost, c))
        action = max(scored, key=lambda item: item[0])[1]
    else:
        raise ValueError(f"unknown strategy: {method}")
    label = f"baseline:{method}:{episode}:{action.intervention.name}"
    return Selection(action, hashlib.sha256(label.encode()).hexdigest())


def run_episode(method: str, seed: int, budget: int = 32) -> Result:
    """Conduct updates using dev-only feedback; confirmation measured at ends."""
    if method not in {"value_ucb", "round_robin", "proxy_greedy", "one_step_oracle"}:
        raise ValueError("unknown method")
    if type(seed) is not int or type(budget) is not int or budget < 3:
        raise ValueError("seed must be int and budget >= 3")
    world = TabularRLVR(seed)
    model = LinearImprovementModel() if method == "value_ucb" else None
    confirmation_start = world.confirmation()
    start = world.development()
    spent = 0
    actions: list[str] = []
    evidence: list[Feedback] = []
    while True:
        feasible = [c for c in world.candidates() if c.intervention.cost <= budget - spent]
        if not feasible:
            break
        selected = _choose_baseline(method, feasible, len(actions), model, budget - spent, world)
        action = selected.candidate.intervention
        before = world.development()
        old_version = world.policy_version
        world.apply(action)
        after = world.development()
        feedback = Feedback(
            decision_id=selected.decision_id, action=action.name,
            before_policy_version=old_version, after_policy_version=world.policy_version,
            verifier_version=world.verifier_version, evaluation_id=f"dev:{seed}:{method}:{old_version + 1}",
            development_before=before, development_after=after, cost=action.cost,
        )
        if model is not None:
            model.observe(feedback)
        evidence.append(feedback)
        actions.append(action.name)
        spent += action.cost
    # Only the start and terminal confirmation results are accessed.
    reads_during_training = world.confirmation_reads - 1
    end = world.confirmation()
    return Result(method, seed, budget, spent, start, world.development(),
                  confirmation_start, end, tuple(actions), tuple(evidence), reads_during_training)


def summarize(seeds: Iterable[int], budget: int = 32) -> dict[str, object]:
    """Paired comparison with locked seed sets; no post-hoc tuning."""
    seed_list = list(seeds)
    if not seed_list or len(set(seed_list)) != len(seed_list):
        raise ValueError("nonempty unique seeds required")
    methods = ("value_ucb", "round_robin", "proxy_greedy", "one_step_oracle")
    episodes = {m: [run_episode(m, seed, budget) for seed in seed_list] for m in methods}
    means = {m: sum(r.confirmation_gain for r in rows) / len(rows)
             for m, rows in episodes.items()}
    rng = random.Random(20261013)
    paired = {}
    for other in methods[1:]:
        differences = [a.confirmation_gain - b.confirmation_gain for a, b in
                       zip(episodes["value_ucb"], episodes[other], strict=True)]
        draws = []
        for _ in range(1000):
            draws.append(sum(differences[rng.randrange(len(differences))]
                             for _ in differences) / len(differences))
        draws.sort()
        paired[other] = {
            "mean_gain_difference": sum(differences) / len(differences),
            "paired_bootstrap_95_percentile": [draws[25], draws[974]],
        }
    return {
        "study": "meta_improvement_cpu_v1", "type": "synthetic_exact_expected_rlvr",
        "seeds": seed_list, "budget_per_seed": budget,
        "development_weights": list(DEV_WEIGHTS),
        "confirmation_weights": list(CONFIRM_WEIGHTS),
        "mean_confirmation_gain": means,
        "paired_vs_value_ucb": paired,
        "episodes": [{"method": r.method, "seed": r.seed, "cost": r.cost_spent,
                      "development_gain": r.development_end - r.development_start,
                      "confirmation_gain": r.confirmation_gain,
                      "actions": list(r.interventions),
                      "confirmation_reads_during_training": r.confirmation_reads_during_training}
                     for m in methods for r in episodes[m]],
        "limitation": "Tabular expected-gradient dynamics only; no actual LLM improvement, no RSI capability claim.",
    }


def canonical_summary_json(seeds: Iterable[int], budget: int = 32) -> str:
    return json.dumps(summarize(seeds, budget), indent=2, sort_keys=True, allow_nan=False) + "\n"
