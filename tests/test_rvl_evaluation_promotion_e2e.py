"""One CPU integration contract from RVL evaluation through promotion/rollback."""

import asyncio
from dataclasses import dataclass, field, replace
import math
from types import SimpleNamespace

import pytest

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.integrations.rvl_grpo import RVLGRPOConfig, RVLGRPOHooks
from vare.types import Task, Verification
from vare.verifiers import VerifierEnsemble, VerifierMember


@dataclass
class Generation:
    prompt_id: str
    prompt: str
    response: str
    logprob: float = -0.1
    token_count: int = 1
    latency_s: float = 0.001
    metadata: dict = field(default_factory=lambda: {
        "sampling_temperature": 0.8,
        "prompt_token_ids": [1],
        "response_token_ids": [2],
        "response_token_logprobs": [-0.1],
    })


@dataclass
class VerifiedGeneration:
    generation: Generation
    reward: float
    verifier_latency_s: float
    verifier_version: int
    metadata: dict = field(default_factory=dict)


class TinyTrainer:
    """Exact state snapshots let the test check the full rollback contract."""

    config = SimpleNamespace(advantage_eps=1e-6, clip_advantage=5.0)

    def __init__(self, *, update_weight=True):
        self.weight = 0
        self.optimizer_step = 0
        self.update_weight = update_weight

    def snapshot_training_state(self):
        return {"weight": self.weight, "optimizer_step": self.optimizer_step}

    def restore_training_state(self, state):
        self.weight = state["weight"]
        self.optimizer_step = state["optimizer_step"]

    def train_step(self, samples, *, advantages=None):
        assert len(samples) == 1
        assert advantages == [0.0]
        if self.update_weight:
            self.weight += 1
        self.optimizer_step += 1


class LocalBackend:
    def __init__(self, trainer, *, misbind_candidate=False):
        self.trainer = trainer
        self.misbind_candidate = misbind_candidate
        self.calls = []

    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        assert n == 1
        self.calls.append((prompt_id, prompt, self.trainer.weight))
        if self.misbind_candidate and self.trainer.weight and prompt_id.startswith("eval-"):
            return [Generation("other-task", prompt, "1")]
        answer = "1" if self.trainer.weight else "0"
        return [Generation(prompt_id, prompt, answer)]


class AlwaysVerifier:
    name = "integration-oracle"
    version = 0
    trusted = True

    async def verify(self, attempt):
        return Verification(1.0, True, 1.0, self.version, self.name, trusted=True)


class EvidenceMutationHooks(RVLGRPOHooks):
    """Mutations occur only after the actual RVL evaluator built its report."""

    def __init__(self, *, evidence_case, paired_gate, **kwargs):
        self.evidence_case = evidence_case
        super().__init__(**kwargs)

    async def evaluate(self, policy_id):
        report = await super().evaluate(policy_id)
        scores = report.metadata["per_task_scores"]
        case = self.evidence_case
        if case == "underreported":
            # Reproduce the dangerous contract violation: n and primary still
            # describe both evaluated rows, while only one row is retained.
            report.metadata["per_task_scores"] = {next(iter(scores)): next(iter(scores.values()))}
        elif case == "missing" and policy_id != self.active_policy()[0]:
            report.metadata.pop("per_task_scores")
        elif case == "malformed" and policy_id != self.active_policy()[0]:
            report.metadata["per_task_scores"] = {"eval-0": math.nan, "eval-1": 1.0}
        elif case == "mismatched" and policy_id != self.active_policy()[0]:
            report.metadata["per_task_scores"] = {f"other-{key}": value for key, value in scores.items()}
        elif case in {"aggregate-only", "aggregate-only-paired"}:
            report.metadata.pop("per_task_scores")
        elif case == "wrong-policy-id" and policy_id != self.active_policy()[0]:
            report = replace(report, policy_id="unrequested-policy")
        return report


def _run_case(case, *, paired_gate=True, misbind_candidate=False):
    trainer = TinyTrainer(update_weight=case != "no-improvement")
    backend = LocalBackend(trainer, misbind_candidate=misbind_candidate)
    eval_tasks = [
        Task("eval-0", "eval prompt 0", "math"),
        Task("eval-1", "eval prompt 1", "math"),
    ]
    hooks = EvidenceMutationHooks(
        backend=backend,
        trainer=trainer,
        eval_tasks=eval_tasks,
        score_fn=lambda task, response: float(response == "1"),
        config=RVLGRPOConfig(seed=19),
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
        evidence_case=case,
        paired_gate=paired_gate,
    )
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=VerifierEnsemble([VerifierMember(AlwaysVerifier())]),
        config=EngineConfig(
            replay_batch_size=1,
            promotion=PromotionConfig(
                min_primary_gain=0.01,
                min_eval_examples=2,
                paired_confidence_gate=paired_gate,
                min_paired_examples=1,
                paired_bootstrap_samples=100,
            ),
        ),
        seed=19,
    )
    return loop, hooks, trainer, backend


def _assert_exact_incumbent_rollback(hooks, trainer):
    assert hooks.active_policy() == ("policy-0", 0)
    assert hooks._loaded_id == "policy-0"
    assert trainer.snapshot_training_state() == {"weight": 0, "optimizer_step": 0}
    assert set(hooks._states) == {"policy-0"}


def test_rvl_evaluate_to_promotion_is_fail_closed_and_transactional():
    """Exercise real hooks, reports, gate, and engine promote/discard behavior."""

    # Independent expected-score oracle: the local backend emits two zeros for
    # policy-0 and two ones for policy-1; do not derive expectations from the
    # report under test.
    cases = [
        ("valid", True, True, None),
        ("no-improvement", True, False, "insufficient_primary_gain"),
        ("underreported", True, False, "invalid_evaluation_metrics"),
        ("missing", True, False, "missing_paired_evidence"),
        ("malformed", True, False, "invalid_evaluation_metrics"),
        ("mismatched", True, False, "paired_identity_mismatch"),
        ("aggregate-only-paired", True, False, "missing_paired_evidence"),
        ("aggregate-only", False, True, None),
        ("wrong-policy-id", True, False, "evaluation_policy_mismatch"),
    ]

    for case, paired_gate, should_promote, rejection_reason in cases:
        loop, hooks, trainer, backend = _run_case(case, paired_gate=paired_gate)
        result = asyncio.run(loop.run_round([Task("train", "train prompt", "math")], round_index=0))
        if should_promote:
            assert result.promoted_id == "policy-1", case
            assert result.incumbent_eval.primary == 0.0
            assert result.candidate_eval.primary == 1.0
            assert hooks.active_policy() == ("policy-1", 1)
            assert hooks._loaded_id == "policy-1"
            assert trainer.snapshot_training_state() == {"weight": 1, "optimizer_step": 1}
            assert set(hooks._states) == {"policy-1"}
        else:
            assert not result.decision.accepted, case
            assert rejection_reason in result.decision.reasons, case
            _assert_exact_incumbent_rollback(hooks, trainer)
        # Both actual reports came from two distinct requested task identities.
        assert sum(call[0].startswith("eval-") for call in backend.calls) == 4

    # Misbound actual backend output raises inside RVLGRPOHooks.evaluate rather
    # than returning a report. The engine must still discard the candidate.
    loop, hooks, trainer, backend = _run_case("valid", misbind_candidate=True)
    with pytest.raises(ValueError, match="evaluation generation task identity mismatch"):
        asyncio.run(loop.run_round([Task("train", "train prompt", "math")], round_index=0))
    _assert_exact_incumbent_rollback(hooks, trainer)
    assert any(prompt_id.startswith("eval-") and weight == 1 for prompt_id, _, weight in backend.calls)

    # Duplicate task identities are rejected at the real RVL adapter boundary,
    # before a report can be formed or a backend call can consume the split.
    trainer = TinyTrainer()
    backend = LocalBackend(trainer)
    with pytest.raises(ValueError, match="evaluation task IDs must be unique"):
        RVLGRPOHooks(
            backend=backend,
            trainer=trainer,
            eval_tasks=[Task("dup", "first"), Task("dup", "second")],
            score_fn=lambda task, response: 0.0,
            generation_factory=Generation,
            verified_generation_factory=VerifiedGeneration,
        )
    assert backend.calls == []
