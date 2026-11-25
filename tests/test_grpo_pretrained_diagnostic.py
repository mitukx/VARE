"""No-model-download contract checks for the pretrained alignment diagnostic."""
from __future__ import annotations

from dataclasses import replace

import pytest

from scripts import diagnose_grpo_pretrained_sampling as D
from scripts.long_horizon_eval import TraceGeneration
from vare import grpo_mlx as G


class RecordingPolicy:
    def __init__(self, **overrides):
        self.overrides = overrides
        self.calls = []

    def generate(self, messages, *, max_tokens, temperature, top_p, seed):
        self.calls.append((list(messages), seed, max_tokens, temperature, top_p))
        prompt = [1, 2] if len(messages) == 1 else [1, 2, 3, 4]
        completion = [5, 6]
        result = TraceGeneration(
            text="neutral text",
            prompt_tokens=len(prompt),
            completion_tokens=len(completion),
            seconds=0.1,
            finish="length",
            prompt_ids=prompt,
            completion_ids=completion,
            sampling_temperature=temperature,
            sampling_top_p=top_p,
            sampling_behavior_logprobs=[-1.0, -2.0],
        )
        return replace(result, **self.overrides)


def test_collect_group_consumes_real_trace_generation_contract():
    p = D.load_protocol()
    policy = RecordingPolicy()
    samples, raw = D.collect_group(policy, p, p["contexts"][0], "checkpoint")
    assert len(samples) == len(p["seeds"]) == 2
    assert len(raw) == len(policy.calls) == 4
    assert [row["finish_reason"] for row in raw] == ["length"] * 4
    assert [row["seed"] for row in raw] == [
        seed + turn * p["second_turn_seed_offset"]
        for seed in p["seeds"] for turn in range(2)
    ]
    assert [row["prompt_ids"] for row in raw] == [[1, 2], [1, 2, 3, 4]] * 2
    assert all(sample.reward == 0 and len(sample.trajectory_segments) == 2 for sample in samples)
    assert len(G.group_samples(samples, group_size=2)) == 1
    assert not G.group_advantages([sample.reward for sample in samples])[1]
    assert policy.calls[1][0][-1] == {"role": "user", "content": p["observation"]}


@pytest.mark.parametrize("overrides", [
    {"finish": "prompt_too_long"},
    {"prompt_tokens": 3},
    {"completion_tokens": 1},
    {"sampling_behavior_logprobs": [-1.0]},
    {"sampling_temperature": 1.0},
    {"sampling_top_p": 0.9},
])
def test_collect_group_rejects_inconsistent_trace_generation(overrides):
    p = D.load_protocol()
    with pytest.raises(ValueError, match="incomplete sampled diagnostic action"):
        D.collect_group(RecordingPolicy(**overrides), p, p["contexts"][0], "checkpoint")
