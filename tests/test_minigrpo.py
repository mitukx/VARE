from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from minigrpo.expr import evaluate_expression, parse_final_integer
from minigrpo.grpo import (
    TokenRow,
    clipped_surrogate,
    correctness_reward,
    cpu_policy_step,
    format_reward,
    group_advantages,
    sequence_loss,
)
from minigrpo.loop import Rollout, RunConfig, check_payload, dev_sample_seed, execute, rollout_seed
from minigrpo.plot import render_svg
from minigrpo.splits import build_split, dev_tasks, derive_seeds, task_from_seed, train_tasks

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_minigrpo.py"


def test_seed_split_matches_the_known_train_row() -> None:
    task = task_from_seed(derive_seeds("train")[0], "train")
    assert task.seed == 1_029_953
    assert task.expression == "23544 + 19091 - 10702"
    assert task.value == 31_933
    assert evaluate_expression(task.expression) == task.value
    assert len(train_tasks()) == 128
    assert len(dev_tasks()) == 32
    assert len(derive_seeds("confirm")) == 64
    confirm = {task.expression for task in build_split("confirm")}
    assert confirm.isdisjoint({task.expression for task in train_tasks()})
    assert confirm.isdisjoint({task.expression for task in dev_tasks()})


def test_check_does_not_use_confirm() -> None:
    payload = check_payload()
    assert payload == {
        "confirm_tasks_used": 0,
        "dev_tasks": 32,
        "status": "checked",
        "train_tasks": 128,
    }
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import mlx" not in source.split("def main")[0]
    assert "import torch" not in source.split("def main")[0]


def test_advantages_standardize_and_keep_ties() -> None:
    values = group_advantages([1.0, 0.0, 1.0, 1.0], 2)
    assert len(values) == 4
    assert math.isclose(values[0], 0.5 / (0.5 + 1e-8))
    assert math.isclose(values[1], -0.5 / (0.5 + 1e-8))
    assert values[2] == 0.0 and values[3] == 0.0
    plain = group_advantages([1.0, 0.0], 2, dr_grpo=True)
    assert plain == [0.5, -0.5]
    try:
        group_advantages([1.0, 0.0, 1.0], 2)
    except ValueError:
        pass
    else:
        raise AssertionError("a partial group was accepted")


def test_loss_ignores_masked_tokens_and_clips() -> None:
    base = TokenRow(logp=(-0.2, 0.1), logp_old=(-0.2, 0.0), mask=(0, 1), advantage=1.0)
    masked = TokenRow(logp=(-9.0, 0.1), logp_old=(-0.2, 0.0), mask=(0, 1), advantage=1.0)
    active = TokenRow(logp=(-0.2, 0.4), logp_old=(-0.2, 0.0), mask=(0, 1), advantage=1.0)
    assert sequence_loss([base])["loss"] == sequence_loss([masked])["loss"]
    assert sequence_loss([base])["loss"] != sequence_loss([active])["loss"]
    try:
        sequence_loss([TokenRow(logp=(0.0,), logp_old=(0.0,), mask=(0,), advantage=1.0)])
    except ValueError:
        pass
    else:
        raise AssertionError("an empty completion mask was accepted")
    positive = clipped_surrogate(math.log(2.0), 0.0, 1.0, 0.2)
    assert math.isclose(positive, -1.2)
    negative = clipped_surrogate(math.log(0.5), 0.0, -1.0, 0.2)
    assert math.isclose(negative, 0.8)


def test_dr_grpo_does_not_divide_by_length() -> None:
    short = TokenRow(logp=(0.0,), logp_old=(0.0,), mask=(1,), advantage=1.0)
    long = TokenRow(logp=(0.0, 0.0, 0.0), logp_old=(0.0, 0.0, 0.0), mask=(1, 1, 1), advantage=1.0)
    standard = sequence_loss([short, long])["loss"]
    doctor = sequence_loss([short, long], dr_grpo=True)["loss"]
    assert math.isclose(float(standard), -1.0)
    assert math.isclose(float(doctor), -2.0)


def test_cpu_smoke_and_metric_log(tmp_path: Path) -> None:
    smoke = cpu_policy_step()
    assert smoke["theta"] > 0.0 and smoke["loss_after"] < smoke["loss_before"]
    tasks = train_tasks()[:2]
    dev = dev_tasks()[:2]

    class Fake:
        def prepare(self, resume: Path | None) -> None:
            return None

        def generate(self, prompt: str, seed: int) -> str:
            return "no integer here"

        def encode(self, prompt: str, completion: str) -> Rollout:
            return Rollout(text=completion, old_logp=(0.0,), mask=(1,), entropy=(0.2,))

        def update(self, rollouts: list[Rollout], advantages: list[float], *, epochs: int) -> float:
            assert len(rollouts) == len(advantages) == 2
            loss = sequence_loss(
                [
                    TokenRow(
                        logp=rollout.old_logp,
                        logp_old=rollout.old_logp,
                        mask=rollout.mask,
                        advantage=advantage,
                        entropy=rollout.entropy,
                    )
                    for rollout, advantage in zip(rollouts, advantages, strict=True)
                ]
            )["loss"]
            assert isinstance(loss, float)
            return loss

        def save(self, directory: Path) -> None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "ok").write_text("ok", encoding="utf-8")

    config = RunConfig(
        backend="cpu",
        model="cpu-smoke",
        revision="",
        steps=2,
        group_size=2,
        max_new_tokens=8,
        learning_rate=1e-5,
        kl_coef=0.0,
        clip_eps=0.2,
        dr_grpo=False,
        epochs=1,
        lora_rank=8,
        lora_scale=20.0,
        lora_dropout=0.0,
        num_layers=1,
        seed=7,
        temperature=0.7,
        output=str(tmp_path),
    )
    execute(Fake(), config, tasks, dev)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    train = [row for row in rows if row["kind"] == "train"]
    assert [row["step"] for row in train] == [1, 2]
    for row in train:
        for key in ("reward_mean", "entropy_mean", "response_length_mean", "kl_mean", "format_compliance_rate"):
            assert key in row
        assert row["group_size"] == 2
        assert row["split"] == "train"
    summary = [row for row in rows if row["kind"] == "dev_summary"]
    assert summary[-1]["n"] == 2
    assert summary[-1]["split"] == "dev"
    svg = render_svg(rows)
    assert "reward_mean" in svg and "format_compliance_rate" in svg
    assert rollout_seed(7, 2, 1, 2) != dev_sample_seed(7, 0)
    assert parse_final_integer("#### 3") == 3
    assert format_reward("#### 1") == 1.0
    assert correctness_reward("1 + 1", "#### 1") == 0.0
    assert correctness_reward("1 + 1", "#### 2") == 1.0


def test_check_command() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "mlx", "--check"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["confirm_tasks_used"] == 0
    assert payload["train_tasks"] == 128
