import asyncio
from dataclasses import dataclass, field

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

    def snapshot_training_state(self):
        return {"weight": self.weight}

    def restore_training_state(self, state):
        self.weight = state["weight"]

    def train_step(self, samples):
        assert samples and isinstance(samples[0], VerifiedGeneration)
        self.weight += 1
        return {"loss": 0.0}


class FakeBackend:
    def __init__(self, trainer):
        self.trainer = trainer

    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        assert n == 1
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
        inc = await hooks.evaluate("policy-0")
        cand = await hooks.evaluate(candidate)
        assert inc.primary == 0.0
        assert cand.primary == 1.0
        assert trainer.weight == 0
        await hooks.promote(candidate)
        assert hooks.active_policy() == (candidate, 1)
        assert trainer.weight == 1

    asyncio.run(run())
