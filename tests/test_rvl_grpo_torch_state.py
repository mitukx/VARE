"""CPU state contracts at the VARE adapter boundary (optional PyTorch)."""
import asyncio
import copy
from types import SimpleNamespace

import pytest

from vare.integrations.rvl_grpo import RVLGRPOHooks
from vare.types import Attempt, Experience, Task, Verification

torch = pytest.importorskip("torch")


class SharedModel(torch.nn.Module):
    def __init__(self, shared=True):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.first = torch.nn.Module()
        self.first.dropout = torch.nn.Dropout(0.5)
        self.last = torch.nn.Module()
        self.last.dropout = self.first.dropout if shared else torch.nn.Dropout(0.5)
        self.train()
        self.first.training = False
        self.first.dropout.training = False

    def forward(self, x):
        return self.first.dropout(self.linear(x))


class TensorTrainer:
    """Real AdamW/tensor/RNG state with a small injectable trainer contract."""
    torch = torch

    def __init__(self, shared=True):
        self.model = SharedModel(shared)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.fail_step = False
        self.train_step([None])  # Populate optimizer moments before snapshot.
        self.model.first.training = False
        self.model.first.dropout.training = False

    def snapshot_training_state(self):
        return copy.deepcopy({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": torch.get_rng_state(),
        })

    def restore_training_state(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        torch.set_rng_state(state["rng"])

    def train_step(self, samples):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.model(torch.ones(3, 4)).square().mean().backward()
        self.optimizer.step()
        if self.fail_step:
            raise RuntimeError("injected after AdamW step")


class ModeChangingBackend:
    """Mirror HFLocalBackend's seeded generation and root-mode restoration."""
    def __init__(self, trainer):
        self.trainer = trainer
        self.fail = False

    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        model = self.trainer.model
        was_training = model.training
        torch.manual_seed(seed)
        model.eval()
        try:
            model(torch.ones(3, 4))
            if self.fail:
                raise RuntimeError("injected generation failure")
            return [SimpleNamespace(
                prompt_id=prompt_id, prompt=prompt, response="ok", logprob=-0.1,
                token_count=1, latency_s=0.001, metadata={},
            )]
        finally:
            model.train(was_training)


def make_hooks(trainer, backend=None):
    return RVLGRPOHooks(
        trainer=trainer, backend=backend or ModeChangingBackend(trainer),
        eval_tasks=[Task("eval", "prompt")], score_fn=lambda *_: 1.0,
        generation_factory=SimpleNamespace, verified_generation_factory=SimpleNamespace,
        current_verifier_version=0,
    )


def experience():
    raw = dict(prompt_id="train", prompt="prompt", response="ok", logprob=-0.1,
               token_count=1, latency_s=0.001,
               metadata={"prompt_token_ids": [1], "response_token_ids": [2]})
    return Experience(
        Attempt(Task("train", "prompt"), "ok", "policy-0", 0, 0,
                metadata={"rvl_generation": raw}),
        Verification(1.0, True, 1.0, 0, "fixture", trusted=True), 0, 0, 0.0,
    )


def modes(model):
    return {name: module.training for name, module in model.named_modules()}


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("fail_step", [True, False])
def test_candidate_restores_exact_modes_optimizer_rng_and_next_forward(shared, fail_step):
    torch.manual_seed(13)
    trainer = TensorTrainer(shared)
    hooks = make_hooks(trainer)
    before = trainer.snapshot_training_state()
    before_modes = modes(trainer.model)
    expected = trainer.model(torch.ones(3, 4)).detach().clone()
    trainer.restore_training_state(before)
    trainer.fail_step = fail_step

    if fail_step:
        with pytest.raises(RuntimeError, match="injected after AdamW"):
            asyncio.run(hooks.train_candidate("policy-0", [experience()]))
        assert set(hooks._states) == {"policy-0"}
    else:
        candidate = asyncio.run(hooks.train_candidate("policy-0", [experience()]))
        assert candidate in hooks._states
    assert_equal(before, trainer.snapshot_training_state())
    assert modes(trainer.model) == before_modes
    assert torch.equal(trainer.model(torch.ones(3, 4)).detach(), expected)
    assert hooks.active_policy() == ("policy-0", 0)
    assert hooks._loaded_id == "policy-0"


@pytest.mark.parametrize("operation", ["evaluate", "rollout"])
def test_rng_cleanup_failure_invalidates_or_recovers_loaded_state(monkeypatch, operation):
    trainer = TensorTrainer()
    hooks = make_hooks(trainer)
    before = trainer.snapshot_training_state()
    before_modes = modes(trainer.model)
    real_set_rng = torch.set_rng_state
    fail_once = True

    def set_rng(state):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("injected RNG cleanup failure")
        real_set_rng(state)

    monkeypatch.setattr(torch, "set_rng_state", set_rng)

    async def run():
        with pytest.raises(RuntimeError, match="injected RNG cleanup"):
            if operation == "evaluate":
                await hooks.evaluate("policy-0")
            else:
                await hooks.rollout(Task("first", "prompt"), "policy-0", 0, 0)
        assert hooks._loaded_id == ("policy-0" if operation == "evaluate" else None)
        await hooks.rollout(Task("recovery", "prompt"), "policy-0", 0, 1)

    asyncio.run(run())
    assert_equal(before, trainer.snapshot_training_state())
    assert modes(trainer.model) == before_modes
    assert hooks._loaded_id == "policy-0"


@pytest.mark.parametrize("operation", ["evaluate", "rollout"])
@pytest.mark.parametrize("fail", [False, True])
def test_active_inference_preserves_modes_rng_and_tensor_state(operation, fail):
    torch.manual_seed(23)
    trainer = TensorTrainer()
    backend = ModeChangingBackend(trainer)
    hooks = make_hooks(trainer, backend)
    before = trainer.snapshot_training_state()
    before_modes = modes(trainer.model)
    backend.fail = fail

    async def run():
        if operation == "evaluate":
            return await hooks.evaluate("policy-0")
        return await hooks.rollout(Task("rollout", "prompt"), "policy-0", 0, 1)

    if fail:
        with pytest.raises(RuntimeError, match="injected generation"):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert modes(trainer.model) == before_modes
    assert_equal(before, trainer.snapshot_training_state())
    assert hooks._loaded_id == "policy-0"
