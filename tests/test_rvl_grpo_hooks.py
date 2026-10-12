from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import math

import pytest

from vare.integrations.rvl_grpo import RVLGRPOConfig, RVLGRPOHooks
from vare.types import Attempt, Experience, Task, Verification


@dataclass
class Generation:
    prompt_id: str
    prompt: str
    response: str
    logprob: float
    token_count: int
    latency_s: float
    metadata: dict = field(default_factory=dict)


@dataclass
class VerifiedGeneration:
    generation: Generation
    reward: float
    verifier_latency_s: float
    verifier_version: int
    metadata: dict = field(default_factory=dict)


class FakeTrainer:
    def __init__(self):
        self.weight = 0
        self.optimizer_step = 0

    def snapshot_training_state(self):
        return {"weight": self.weight, "optimizer_step": self.optimizer_step}

    def restore_training_state(self, state):
        self.weight = state["weight"]
        self.optimizer_step = state["optimizer_step"]

    def train_step(self, samples):
        assert samples and isinstance(samples[0], VerifiedGeneration)
        self.weight += 1
        self.optimizer_step += 1
        return {"loss": 0.0}


class PartiallyFailingTrainer(FakeTrainer):
    def train_step(self, samples):
        super().train_step(samples)
        raise RuntimeError("simulated failure after mutating policy state")


class PartiallyFailingRestoreTrainer(FakeTrainer):
    def __init__(self):
        super().__init__()
        self.fail_candidate_restore = False
        self.restore_targets = []

    def restore_training_state(self, state):
        self.restore_targets.append(state["weight"])
        # Simulate a restore that changes part of the live trainer before
        # failing. The adapter must then force a second restore to the incumbent.
        self.weight = state["weight"]
        self.optimizer_step = state["optimizer_step"]
        if state["weight"] > 0 and self.fail_candidate_restore:
            self.fail_candidate_restore = False
            raise RuntimeError("simulated partial candidate restore failure")


class FakeBackend:
    def __init__(self, trainer):
        self.trainer = trainer
        self.fail = False

    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        assert n == 1
        if self.fail:
            raise RuntimeError("simulated evaluation generation failure")
        return [Generation(
            prompt_id=prompt_id,
            prompt=prompt,
            response="1" if self.trainer.weight > 0 else "0",
            logprob=-0.1,
            token_count=1,
            latency_s=0.001,
            metadata={
                "sampling_temperature": max(temperature, 0.1),
                "prompt_token_ids": [1],
                "response_token_ids": [2],
                "response_token_logprobs": [-0.1],
            },
        )]


def test_rvl_grpo_hooks_candidate_is_transactional():
    trainer = FakeTrainer()
    backend = FakeBackend(trainer)
    eval_tasks = [Task(id=f"e{i}", prompt="answer", family="math") for i in range(4)]
    hooks = RVLGRPOHooks(
        backend=backend,
        trainer=trainer,
        eval_tasks=eval_tasks,
        score_fn=lambda task, response: float(response == "1"),
        config=RVLGRPOConfig(seed=1),
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )
    task = Task(id="t", prompt="train", family="math")
    raw = {
        "prompt_id": "t",
        "prompt": "train",
        "response": "0",
        "logprob": -0.1,
        "token_count": 1,
        "latency_s": 0.001,
        "metadata": {
            "sampling_temperature": 0.8,
            "prompt_token_ids": [1],
            "response_token_ids": [2],
            "response_token_logprobs": [-0.1],
        },
    }
    exp = Experience(
        attempt=Attempt(task, "0", "policy-0", 0, 0, logprob=-0.1, metadata={"rvl_generation": raw}),
        verification=Verification(1.0, True, 1.0, 1, "oracle", trusted=True),
        policy_lag=0,
        verifier_lag=0,
        shift_score=0.0,
    )

    async def run():
        candidate = await hooks.train_candidate("policy-0", [exp])
        assert trainer.weight == 0  # candidate is not silently installed
        backend.fail = True
        try:
            await hooks.evaluate(candidate)
        except RuntimeError as exc:
            assert "simulated evaluation generation failure" in str(exc)
        else:
            raise AssertionError("evaluation generation failure should propagate")
        finally:
            backend.fail = False
        assert hooks.active_policy() == ("policy-0", 0)
        assert hooks._loaded_id == "policy-0"
        assert trainer.weight == 0
        inc = await hooks.evaluate("policy-0")
        cand = await hooks.evaluate(candidate)
        assert inc.primary == 0.0
        assert cand.primary == 1.0
        assert trainer.weight == 0
        await hooks.promote(candidate)
        assert hooks.active_policy() == (candidate, 1)
        assert trainer.weight == 1

    asyncio.run(run())


@pytest.mark.parametrize("score", [2.0, math.nan, math.inf])
def test_rvl_trainer_boundary_rejects_mutated_reward_before_optimizer(score):
    trainer = FakeTrainer()
    hooks = RVLGRPOHooks(
        backend=FakeBackend(trainer),
        trainer=trainer,
        eval_tasks=[],
        score_fn=lambda task, response: 0.0,
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )
    verification = Verification(0.75, True, 1.0, 1, "fixture", trusted=True)
    verification.score = score
    exp = Experience(
        attempt=Attempt(
            Task("train", "prompt"),
            "answer",
            "policy-0",
            0,
            0,
            metadata={
                "rvl_generation": {
                    "prompt_id": "train",
                    "prompt": "prompt",
                    "response": "answer",
                    "logprob": -0.1,
                    "token_count": 1,
                    "latency_s": 0.001,
                    "metadata": {},
                }
            },
        ),
        verification=verification,
        policy_lag=0,
        verifier_lag=0,
        shift_score=0.0,
    )
    with pytest.raises(ValueError, match="score"):
        asyncio.run(hooks.train_candidate("policy-0", [exp]))
    assert trainer.optimizer_step == 0


def test_rvl_grpo_hooks_restores_incumbent_after_partial_train_failure():
    trainer = PartiallyFailingTrainer()
    backend = FakeBackend(trainer)
    task = Task(id="t", prompt="train", family="math")
    raw = {
        "prompt_id": "t",
        "prompt": "train",
        "response": "0",
        "logprob": -0.1,
        "token_count": 1,
        "latency_s": 0.001,
        "metadata": {
            "sampling_temperature": 0.8,
            "prompt_token_ids": [1],
            "response_token_ids": [2],
            "response_token_logprobs": [-0.1],
        },
    }
    exp = Experience(
        attempt=Attempt(task, "0", "policy-0", 0, 0, logprob=-0.1, metadata={"rvl_generation": raw}),
        verification=Verification(1.0, True, 1.0, 1, "oracle", trusted=True),
        policy_lag=0,
        verifier_lag=0,
        shift_score=0.0,
    )
    hooks = RVLGRPOHooks(
        backend=backend,
        trainer=trainer,
        eval_tasks=[task],
        score_fn=lambda task, response: float(response == "1"),
        config=RVLGRPOConfig(seed=1),
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )

    async def run():
        try:
            await hooks.train_candidate("policy-0", [exp])
        except RuntimeError as exc:
            assert "simulated failure" in str(exc)
        else:
            raise AssertionError("partial training failure should propagate")

        assert hooks.active_policy() == ("policy-0", 0)
        assert trainer.weight == 0
        assert trainer.optimizer_step == 0
        assert hooks._loaded_id == "policy-0"
        assert set(hooks._states) == {"policy-0"}
        assert hooks._counter == 1
        attempt = await hooks.rollout(task, "policy-0", 0, 1)
        assert attempt.output == "0"
        assert trainer.weight == 0
        assert trainer.optimizer_step == 0

    asyncio.run(run())


def test_rvl_grpo_hooks_forces_incumbent_restore_after_partial_candidate_restore_failure():
    trainer = PartiallyFailingRestoreTrainer()
    backend = FakeBackend(trainer)
    task = Task(id="t", prompt="train", family="math")
    raw = {
        "prompt_id": "t",
        "prompt": "train",
        "response": "0",
        "logprob": -0.1,
        "token_count": 1,
        "latency_s": 0.001,
        "metadata": {
            "sampling_temperature": 0.8,
            "prompt_token_ids": [1],
            "response_token_ids": [2],
            "response_token_logprobs": [-0.1],
        },
    }
    exp = Experience(
        attempt=Attempt(task, "0", "policy-0", 0, 0, logprob=-0.1, metadata={"rvl_generation": raw}),
        verification=Verification(1.0, True, 1.0, 1, "oracle", trusted=True),
        policy_lag=0,
        verifier_lag=0,
        shift_score=0.0,
    )
    hooks = RVLGRPOHooks(
        backend=backend,
        trainer=trainer,
        eval_tasks=[task],
        score_fn=lambda task, response: float(response == "1"),
        config=RVLGRPOConfig(seed=1),
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )

    async def run():
        candidate = await hooks.train_candidate("policy-0", [exp])
        trainer.fail_candidate_restore = True

        try:
            await hooks.evaluate(candidate)
        except RuntimeError as exc:
            assert "simulated partial candidate restore failure" in str(exc)
        else:
            raise AssertionError("partial candidate restore failure should propagate")

        # The failed candidate restore changed live state to the candidate.
        # Recovery must issue a fresh incumbent restore despite that partial
        # mutation, then leave the adapter's identity marker in sync.
        assert trainer.restore_targets[-2:] == [1, 0]
        assert trainer.weight == 0
        assert trainer.optimizer_step == 0
        assert hooks._loaded_id == "policy-0"
        assert hooks.active_policy() == ("policy-0", 0)

        # A subsequent candidate evaluation remains possible after recovery.
        report = await hooks.evaluate(candidate)
        assert report.primary == 1.0
        assert trainer.weight == 0

    asyncio.run(run())


class ContractModule:
    """Stand-in for the PyTorch module contract used by mode rollback.

    ``named_modules`` yields each object once. ``train`` writes ``training``
    and recurses through every registered child, including a child already
    visited through another parent. This matches ``nn.Module`` traversal and
    ``Module.train``; it is not a PyTorch installation.
    """

    def __init__(self) -> None:
        self.training = True
        self._children: dict[str, ContractModule] = {}

    def add(self, name: str, module: ContractModule) -> None:
        self._children[name] = module

    def named_modules(self, memo: set[int] | None = None, prefix: str = ""):
        if memo is None:
            memo = set()
        if id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for name, child in self._children.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(memo, child_prefix)

    def train(self, mode: bool = True):
        if type(mode) is not bool:
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for child in self._children.values():
            child.train(mode)
        return self


def _module_modes(model: ContractModule) -> dict[str, bool]:
    return {name: module.training for name, module in model.named_modules()}


class ModeRollbackTrainer:
    def __init__(self, model: ContractModule, probe: ContractModule):
        self.model = model
        self.probe = probe
        self.step = 0
        self.probe_training_during_update: bool | None = None

    def snapshot_training_state(self):
        return {"step": self.step}

    def restore_training_state(self, state):
        self.step = state["step"]

    def train_step(self, samples):
        assert samples
        self.model.train(True)
        self.probe_training_during_update = self.probe.training
        self.step += 1
        return {"loss": 0.0}


def _mode_rollback_experience() -> Experience:
    task = Task(id="t", prompt="train", family="math")
    raw = {
        "prompt_id": "t",
        "prompt": "train",
        "response": "0",
        "logprob": -0.1,
        "token_count": 1,
        "latency_s": 0.001,
        "metadata": {
            "sampling_temperature": 0.8,
            "prompt_token_ids": [1],
            "response_token_ids": [2],
            "response_token_logprobs": [-0.1],
        },
    }
    return Experience(
        attempt=Attempt(task, "0", "policy-0", 0, 0, logprob=-0.1, metadata={"rvl_generation": raw}),
        verification=Verification(1.0, True, 1.0, 1, "oracle", trusted=True),
        policy_lag=0,
        verifier_lag=0,
        shift_score=0.0,
    )


def _mode_rollback_hooks(trainer: ModeRollbackTrainer) -> RVLGRPOHooks:
    return RVLGRPOHooks(
        backend=FakeBackend(trainer),
        trainer=trainer,
        eval_tasks=[Task(id="t", prompt="train", family="math")],
        score_fn=lambda task, response: float(response == "1"),
        config=RVLGRPOConfig(seed=1),
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )


def _shared_child_graph() -> tuple[ContractModule, ContractModule]:
    root = ContractModule()
    shared = ContractModule()
    left = ContractModule()
    right = ContractModule()
    left.add("drop", shared)
    right.add("drop", shared)
    root.add("left", left)
    root.add("right", right)
    # Assign flags directly so setup does not recurse through the alias.
    root.training = False
    left.training = False
    right.training = True
    shared.training = False
    return root, shared


def _nested_mixed_graph() -> tuple[ContractModule, ContractModule]:
    root = ContractModule()
    parent = ContractModule()
    child = ContractModule()
    parent.add("child", child)
    root.add("parent", parent)
    root.training = False
    parent.training = False
    child.training = True
    return root, child


def test_rvl_grpo_rollback_preserves_shared_child_module_mode():
    model, shared = _shared_child_graph()
    saved = _module_modes(model)
    assert saved == {"": False, "left": False, "left.drop": False, "right": True}
    trainer = ModeRollbackTrainer(model, shared)
    hooks = _mode_rollback_hooks(trainer)

    async def run():
        candidate = await hooks.train_candidate("policy-0", [_mode_rollback_experience()])
        assert trainer.probe_training_during_update is True
        assert trainer.step == 0
        assert _module_modes(model) == saved
        await hooks.promote(candidate)
        assert trainer.step == 1
        assert _module_modes(model) == {name: True for name in saved}

    asyncio.run(run())


def test_rvl_grpo_rollback_preserves_distinct_nested_module_modes():
    model, child = _nested_mixed_graph()
    saved = _module_modes(model)
    assert saved == {"": False, "parent": False, "parent.child": True}
    trainer = ModeRollbackTrainer(model, child)
    hooks = _mode_rollback_hooks(trainer)

    async def run():
        await hooks.train_candidate("policy-0", [_mode_rollback_experience()])
        assert trainer.probe_training_during_update is True
        assert trainer.step == 0
        assert _module_modes(model) == saved

    asyncio.run(run())
