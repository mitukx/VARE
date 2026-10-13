"""CPU-only regressions for fail-closed GRPO rollout validation.

An invalid rollout must be rejected before the trainer can mutate weights.
This deliberately uses a fake backend rather than claiming a real-model result.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from minigrpo.grpo import TokenRow, sequence_loss
from minigrpo.loop import Rollout, RunConfig, execute
from minigrpo.splits import train_tasks


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (TokenRow(logp=(math.nan,), logp_old=(0.0,), mask=(1,), advantage=1.0), "non-finite"),
        (TokenRow(logp=(0.0,), logp_old=(math.inf,), mask=(1,), advantage=1.0), "non-finite"),
        (TokenRow(logp=(0.0,), logp_old=(0.0,), mask=(2,), advantage=1.0), "binary"),
        (TokenRow(logp=(0.0,), logp_old=(0.0,), mask=(1,), advantage=math.nan), "non-finite"),
        (TokenRow(logp=(0.0,), logp_old=(0.0,), mask=(1,), advantage=1.0, entropy=()), "entropy length"),
        (TokenRow(logp=(0.0,), logp_old=(0.0,), mask=(1,), advantage=1.0, entropy=(math.inf,)), "non-finite"),
        (TokenRow(logp=(0.0,), logp_old=(0.0,), mask=(1,), advantage=1.0, logp_ref=(math.nan,)), "non-finite"),
    ],
)
def test_grpo_objective_rejects_invalid_active_inputs(row: TokenRow, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        sequence_loss([row])


def test_masked_nonfinite_padding_is_allowed() -> None:
    row = TokenRow(
        logp=(math.nan, 0.0),
        logp_old=(math.nan, 0.0),
        mask=(0, 1),
        advantage=1.0,
        entropy=(math.nan, 0.0),
    )
    assert sequence_loss([row])["loss"] == -1.0


@pytest.mark.parametrize(
    ("old_logp", "mask", "entropy", "reference", "kl_coef", "reason"),
    [
        ((math.nan,), (1,), (0.0,), None, 0.0, "non-finite"),
        ((0.0,), (2,), (0.0,), None, 0.0, "binary"),
        ((0.0,), (1,), (), None, 0.0, "entropy length"),
        ((0.0,), (1,), (0.0,), None, 0.02, "reference"),
        ((0.0,), (1,), (math.inf,), None, 0.0, "non-finite"),
    ],
)
def test_invalid_rollout_is_rejected_before_optimizer_update(
    tmp_path: Path,
    old_logp: tuple[float, ...],
    mask: tuple[int, ...],
    entropy: tuple[float, ...],
    reference: tuple[float, ...] | None,
    kl_coef: float,
    reason: str,
) -> None:
    calls: list[str] = []

    class RecordingTrainer:
        def prepare(self, resume: Path | None) -> None:
            calls.append("prepare")

        def generate(self, prompt: str, seed: int) -> str:
            return "#### 2"

        def encode(self, prompt: str, completion: str) -> Rollout:
            return Rollout(
                text=completion, old_logp=old_logp, mask=mask,
                entropy=entropy, ref_logp=reference,
            )

        def update(self, rollouts: list[Rollout], advantages: list[float], *, epochs: int) -> float:
            calls.append("update")
            return 0.0

        def save(self, directory: Path) -> None:
            calls.append("save")

    config = RunConfig(
        backend="cpu", model="fake", revision="", steps=1, group_size=2,
        max_new_tokens=8, learning_rate=1e-5, kl_coef=kl_coef, clip_eps=0.2,
        dr_grpo=False, epochs=1, lora_rank=8, lora_scale=20.0,
        lora_dropout=0.0, num_layers=1, seed=7, temperature=0.7,
        output=str(tmp_path), eval_dev=False,
    )
    with pytest.raises(ValueError, match=reason):
        execute(RecordingTrainer(), config, train_tasks()[:1], [])
    assert calls == ["prepare"]
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "metrics.jsonl").exists()
