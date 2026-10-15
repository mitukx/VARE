"""Adapter rollback must not recursively overwrite a shared child mode.

RVL's trainer snapshot had this defect: ``named_modules()`` lists a shared
module once, while ``Module.train()`` still walks every parent edge. Restoring
a later alias with a different mode puts the shared child back into the wrong
mode. ``RVLGRPOHooks`` keeps its own mode map and applies it after
``trainer.restore_training_state()``, so a trainer that already assigns flags
directly is not sufficient.
"""

import asyncio
from dataclasses import dataclass, field

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


class GraphModule:
    """Minimal module graph with PyTorch's shared-module traversal rules.

    ``named_modules`` records each object once. ``train`` still recurses through
    every registered parent, which is the overwrite ``RVLGRPOHooks`` must avoid.
    ``mode_dependent_output`` stands in for eval-mode Dropout: eval returns the
    input unchanged and train can zero it. It is not a measured model forward.
    """

    def __init__(self) -> None:
        self.training = True
        self.train_calls = 0
        self._children: list[tuple[str, GraphModule]] = []

    def add(self, name: str, module: "GraphModule") -> "GraphModule":
        self._children.append((name, module))
        return module

    def named_modules(self, memo: set[int] | None = None, prefix: str = ""):
        if memo is None:
            memo = set()
        if id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for name, child in self._children:
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(memo=memo, prefix=child_prefix)

    def train(self, mode: bool = True) -> "GraphModule":
        self.train_calls += 1
        self.training = bool(mode)
        for _, child in self._children:
            child.train(mode)
        return self

    def mode_dependent_output(self, value: float = 1.0) -> float:
        return value if not self.training else 0.0


class DirectFlagTrainer:
    """Trainer restore assigns flags directly, as in the patched RVL snapshot."""

    def __init__(self, model: GraphModule) -> None:
        self.model = model
        self.weight = 0
        self.calls_after_step: dict[str, int] | None = None

    def snapshot_training_state(self) -> dict:
        return {
            "weight": self.weight,
            "modes": {name: module.training for name, module in self.model.named_modules()},
        }

    def restore_training_state(self, state: dict) -> None:
        self.weight = state["weight"]
        modules = dict(self.model.named_modules())
        for name, mode in state["modes"].items():
            modules[name].training = mode

    def train_step(self, samples, advantages=None) -> dict[str, float]:
        assert samples and isinstance(samples[0], VerifiedGeneration)
        self.weight += 1
        self.model.train(True)
        self.calls_after_step = {
            name: module.train_calls for name, module in self.model.named_modules()
        }
        raise RuntimeError("injected after mode mutation")


def _modes(root: GraphModule) -> dict[str, bool]:
    return {name: module.training for name, module in root.named_modules()}


def _shared_graph(*, alias_first: bool) -> tuple[GraphModule, GraphModule]:
    drop = GraphModule()
    transformer = GraphModule()
    transformer.add("drop", drop)
    alias = GraphModule()
    alias.add("drop", drop)
    root = GraphModule()
    if alias_first:
        root.add("shared_dropout_alias", alias)
        root.add("transformer", transformer)
    else:
        root.add("transformer", transformer)
        root.add("shared_dropout_alias", alias)
    root.train(False)
    alias.train(True)
    drop.train(False)
    return root, drop


def _nested_graph() -> tuple[GraphModule, GraphModule]:
    child = GraphModule()
    parent = GraphModule()
    parent.add("child", child)
    root = GraphModule()
    root.add("parent", parent)
    root.train(True)
    child.train(False)
    return root, child


def _experience() -> Experience:
    task = Task(id="t", prompt="train", family="math")
    raw = {
        "prompt_id": "t",
        "prompt": "train",
        "response": "0",
        "logprob": -0.1,
        "token_count": 1,
        "latency_s": 0.001,
        "metadata": {
            "prompt_token_ids": [1],
            "response_token_ids": [2],
        },
    }
    return Experience(
        attempt=Attempt(
            task,
            "0",
            "policy-0",
            0,
            0,
            logprob=-0.1,
            metadata={"rvl_generation": raw},
        ),
        verification=Verification(1.0, True, 1.0, 1, "oracle", trusted=True),
        policy_lag=0,
        verifier_lag=0,
        shift_score=0.0,
    )


@pytest.mark.parametrize(
    ("graph_case",),
    [
        ("later_alias_differs",),
        ("earlier_alias_masks",),
        ("nested_unshared",),
    ],
)
def test_candidate_rollback_preserves_shared_child_mode(graph_case: str) -> None:
    if graph_case == "later_alias_differs":
        root, probe = _shared_graph(alias_first=False)
    elif graph_case == "earlier_alias_masks":
        root, probe = _shared_graph(alias_first=True)
    else:
        root, probe = _nested_graph()

    saved_modes = _modes(root)
    expected_output = probe.mode_dependent_output()
    if graph_case == "nested_unshared":
        assert saved_modes["parent"] is True
        assert saved_modes["parent.child"] is False
    else:
        assert saved_modes[""] is False
        assert saved_modes["shared_dropout_alias"] is True
        assert any(name.endswith("drop") and mode is False for name, mode in saved_modes.items())
        assert sum(name.endswith("drop") for name in saved_modes) == 1

    trainer = DirectFlagTrainer(root)
    hooks = RVLGRPOHooks(
        backend=object(),
        trainer=trainer,
        eval_tasks=[],
        score_fn=lambda task, response: 0.0,
        config=RVLGRPOConfig(seed=1),
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )

    async def run() -> None:
        with pytest.raises(RuntimeError, match="injected after mode mutation"):
            await hooks.train_candidate("policy-0", [_experience()])

    asyncio.run(run())

    failures: list[str] = []
    restored = _modes(root)
    if restored != saved_modes:
        failures.append(f"per-module modes: saved={saved_modes} restored={restored}")
    if probe.mode_dependent_output() != expected_output:
        failures.append("subsequent mode-dependent forward")
    calls = {name: module.train_calls for name, module in root.named_modules()}
    if calls != trainer.calls_after_step:
        failures.append("restore invoked Module.train")
    if trainer.weight != 0:
        failures.append("trainer weight")
    if failures:
        raise AssertionError("\n".join(failures))
    assert hooks.active_policy() == ("policy-0", 0)


def test_restore_rejects_a_module_whose_class_changed() -> None:
    root, _probe = _nested_graph()
    trainer = DirectFlagTrainer(root)
    hooks = RVLGRPOHooks(
        backend=object(),
        trainer=trainer,
        eval_tasks=[],
        score_fn=lambda task, response: 0.0,
        config=RVLGRPOConfig(seed=1),
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )

    class Replacement(GraphModule):
        pass

    parent = dict(root.named_modules())["parent"]
    replacement = Replacement()
    parent._children = [("child", replacement)]
    with pytest.raises(RuntimeError, match="class changed"):
        hooks._restore_training_state(hooks._states["policy-0"])
    assert replacement.training is True


def test_torch_shared_module_mode_restores_without_calling_train() -> None:
    torch = pytest.importorskip("torch")

    class Flag(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.train_calls = 0

        def train(self, mode: bool = True):  # type: ignore[override]
            self.train_calls += 1
            return super().train(mode)

    class Holder(torch.nn.Module):
        def __init__(self, shared: torch.nn.Module) -> None:
            super().__init__()
            self.left = torch.nn.Sequential()
            self.left.add_module("drop", shared)
            self.right = torch.nn.Sequential()
            self.right.add_module("drop", shared)

    shared = Flag()
    root = Holder(shared)
    root.eval()
    root.right.train(True)
    shared.train(False)
    assert shared.training is False
    assert root.right.training is True

    class Trainer:
        def __init__(self, model: torch.nn.Module) -> None:
            self.model = model

        def snapshot_training_state(self) -> dict[str, int]:
            return {"marker": 1}

        def restore_training_state(self, state: dict[str, int]) -> None:
            if state != {"marker": 1}:
                raise AssertionError(state)

    trainer = Trainer(root)
    hooks = RVLGRPOHooks(
        backend=object(),
        trainer=trainer,
        eval_tasks=[],
        score_fn=lambda task, response: 0.0,
        config=RVLGRPOConfig(seed=1),
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )
    saved = {name: module.training for name, module in root.named_modules()}
    root.train(True)
    calls_before_restore = shared.train_calls
    hooks._restore_training_state(hooks._states["policy-0"])
    restored = {name: module.training for name, module in root.named_modules()}
    assert restored == saved
    assert shared.training is False
    assert root.right.training is True
    assert shared.train_calls == calls_before_restore
