"""Frozen five-arm comparison after the accepted update-cost smoke.

The decision set is the committed confirm split. The 64-task pack is scored
as a secondary curve and cannot change the decision. This module does not
load a model.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Iterable, Mapping, Sequence

from vare.answer_scoring import score_output, summarize_two_tier
from vare.eval_stats import (
    mean_unbiased_pass_at_k,
    mcnemar_exact,
    paired_bootstrap_interval,
    unbiased_pass_at_k,
)
from vare.heldout_exact_integer import (
    grade_output,
    load_task_pack,
    parse_final_integer,
    verify_task_pack,
)
from vare.integer_seed_split import derive_seeds, split_sha256, task_from_seed
from vare.update_cost_smoke import HELDOUT_PACK_SHA256, train_example

PROTOCOL_ID = "format-vs-correctness-comparison-v2"
PROTOCOL_SHA256 = "8214a9c784035f47b20d9e44fc6bbbd3cef45f821d86d6eb2f69936cc25eb6b4"
SMOKE_MANIFEST_SHA256 = "8926c6c6383201bb3f1a067707210db7abd94c90a10f40ae0686784bc4ce511e"
SMOKE_WALL_SECONDS = 8.762955541998963
SMOKE_PEAK_MEMORY_BYTES = 1959434978.0
PLANNING_SECONDS_PER_STEP = 9
ARMS = (
    "base",
    "format_sft",
    "format_reward_rl",
    "correctness_reward_rl",
    "random_reward_rl",
)
UPDATED_ARMS = ARMS[1:]
RL_ARMS = ("format_reward_rl", "correctness_reward_rl", "random_reward_rl")
RUN_SEEDS = (20261016, 20261017, 20261018, 20261019, 20261020)
OPTIMIZER_STEPS = 16
CURVE_STEPS = (8, 16)
GROUP_SIZE = 2
EVAL_SAMPLES = 4
PRIMARY_K = 1
CURVE_KS = (1, 4)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261016
ADVANTAGE_EPSILON = 1e-8
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 48
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-5
LORA_KEYS = ("self_attn.q_proj", "self_attn.v_proj")
PLANNING_OPTIMIZER_SECONDS = len(UPDATED_ARMS) * len(RUN_SEEDS) * OPTIMIZER_STEPS * PLANNING_SECONDS_PER_STEP
RL_GENERATIONS = len(RL_ARMS) * len(RUN_SEEDS) * OPTIMIZER_STEPS * GROUP_SIZE


def comparison_train_examples() -> tuple[dict[str, Any], ...]:
    """Sixteen train-split rows after the consumed smoke example."""
    consumed_seed = train_example()["seed"]
    if consumed_seed != 1029953:
        raise RuntimeError("the consumed smoke train seed changed")
    seeds = [seed for seed in derive_seeds("train") if seed != consumed_seed]
    chosen = tuple(seeds[:OPTIMIZER_STEPS])
    if len(chosen) != OPTIMIZER_STEPS:
        raise RuntimeError("the train split does not have 16 seeds after the smoke row")
    rows: list[dict[str, Any]] = []
    for seed in chosen:
        task = task_from_seed(seed, "train")
        rows.append(
            {
                "completion": f"#### {task.value}",
                "expression": task.expression,
                "prompt": task.prompt,
                "seed": task.seed,
                "value": task.value,
            }
        )
    return tuple(rows)


def confirm_eval_tasks() -> list[dict[str, str]]:
    rows = []
    for seed in derive_seeds("confirm"):
        task = task_from_seed(seed, "confirm")
        rows.append(
            {
                "expression": task.expression,
                "prompt": task.prompt,
                "split": "confirm",
                "task_id": f"confirm-{seed}",
            }
        )
    return rows


def pack_eval_tasks(pack_text: str) -> list[dict[str, str]]:
    rows = load_task_pack(pack_text)
    verify_task_pack(rows)
    return [{**row, "split": "pack"} for row in rows]


def assert_training_rows_are_disjoint(pack_text: str) -> None:
    train_rows = comparison_train_examples()
    train_expressions = {str(row["expression"]) for row in train_rows}
    confirm_expressions = {row["expression"] for row in confirm_eval_tasks()}
    pack_expressions = {row["expression"] for row in load_task_pack(pack_text)}
    dev_expressions = {task_from_seed(seed, "dev").expression for seed in derive_seeds("dev")}
    if train_expressions & confirm_expressions:
        raise ValueError("a train row collides with confirm")
    if train_expressions & pack_expressions:
        raise ValueError("a train row collides with the held-out pack")
    if train_expressions & dev_expressions:
        raise ValueError("a train row collides with dev")
    if 1029953 in {int(row["seed"]) for row in train_rows}:
        raise ValueError("the consumed smoke row is in the comparison train set")


def format_reward(text: str) -> float:
    return 1.0 if parse_final_integer(text) is not None else 0.0


def correctness_reward(expression: str, text: str) -> float:
    return 1.0 if grade_output(expression, text) else 0.0


def random_group_rewards(run_seed: int, step: int) -> list[float]:
    rng = random.Random(f"format-vs-correctness-v2-random-reward:{run_seed}:{step}")
    return [1.0 if rng.random() < 0.5 else 0.0 for _ in range(GROUP_SIZE)]


def rl_rewards(arm: str, expression: str, outputs: Sequence[str], run_seed: int, step: int) -> list[float]:
    if len(outputs) != GROUP_SIZE:
        raise ValueError("a reward group must keep every sample")
    if arm == "format_reward_rl":
        return [format_reward(text) for text in outputs]
    if arm == "correctness_reward_rl":
        return [correctness_reward(expression, text) for text in outputs]
    if arm == "random_reward_rl":
        return random_group_rewards(run_seed, step)
    raise ValueError(f"{arm} is not a reward arm")


def group_advantages(rewards: Sequence[float]) -> list[float]:
    if len(rewards) != GROUP_SIZE or any(reward not in (0.0, 1.0) for reward in rewards):
        raise ValueError("advantages require an intact 0/1 group")
    mean = sum(rewards) / GROUP_SIZE
    variance = sum((reward - mean) ** 2 for reward in rewards) / GROUP_SIZE
    std = math.sqrt(variance)
    if std == 0.0:
        return [0.0] * GROUP_SIZE
    scale = std + ADVANTAGE_EPSILON
    return [(reward - mean) / scale for reward in rewards]


def rollout_seed(arm: str, run_seed: int, step: int, group_index: int) -> int:
    if arm not in RL_ARMS or run_seed not in RUN_SEEDS:
        raise ValueError("rollout seed is outside the frozen grid")
    if not 1 <= step <= OPTIMIZER_STEPS or group_index not in range(GROUP_SIZE):
        raise ValueError("rollout seed is outside the frozen grid")
    arm_index = RL_ARMS.index(arm)
    return 60_061_016 + arm_index * 10_000_000_000 + run_seed * 1_000 + step * 10 + group_index


def eval_sample_seed(run_seed: int, checkpoint_step: int, task_index: int, sample_index: int) -> int:
    if run_seed not in RUN_SEEDS or sample_index not in range(EVAL_SAMPLES):
        raise ValueError("eval seed is outside the frozen grid")
    if checkpoint_step not in (0, *CURVE_STEPS):
        raise ValueError("eval checkpoint is outside the frozen grid")
    return 80_061_016 + run_seed * 100_000 + checkpoint_step * 1_000 + task_index * 10 + sample_index


def checkpoints_for(arm: str) -> tuple[int, ...]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    if arm == "base":
        return (0,)
    return CURVE_STEPS


def final_checkpoint(arm: str) -> int:
    return 0 if arm == "base" else OPTIMIZER_STEPS


def iter_train_units() -> list[dict[str, Any]]:
    examples = comparison_train_examples()
    units: list[dict[str, Any]] = []
    for run_seed in RUN_SEEDS:
        for arm in UPDATED_ARMS:
            for step, example in enumerate(examples, start=1):
                units.append(
                    {
                        "arm": arm,
                        "kind": "train",
                        "run_seed": run_seed,
                        "step": step,
                        "train_seed": example["seed"],
                    }
                )
    return units


def iter_eval_units(confirm: Sequence[Mapping[str, str]], pack: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for run_seed in RUN_SEEDS:
        for arm in ARMS:
            for checkpoint_step in checkpoints_for(arm):
                splits: list[tuple[str, Sequence[Mapping[str, str]]]] = [("confirm", confirm)]
                if checkpoint_step == final_checkpoint(arm):
                    splits.append(("pack", pack))
                for split, tasks in splits:
                    for task_index, task in enumerate(tasks):
                        for sample_index in range(EVAL_SAMPLES):
                            units.append(
                                {
                                    "arm": arm,
                                    "checkpoint_step": checkpoint_step,
                                    "kind": "eval",
                                    "run_seed": run_seed,
                                    "sample_index": sample_index,
                                    "sample_seed": eval_sample_seed(
                                        run_seed, checkpoint_step, task_index, sample_index
                                    ),
                                    "split": split,
                                    "task_id": task["task_id"],
                                }
                            )
    return units


def unit_key(unit: Mapping[str, object]) -> tuple[object, ...]:
    if unit["kind"] == "train":
        return ("train", unit["arm"], unit["run_seed"], unit["step"])
    return (
        "eval",
        unit["arm"],
        unit["run_seed"],
        unit["checkpoint_step"],
        unit["split"],
        unit["task_id"],
        unit["sample_index"],
    )


def row_key(row: Mapping[str, object]) -> tuple[object, ...]:
    if row.get("kind") == "train":
        return ("train", row.get("arm"), row.get("run_seed"), row.get("step"))
    return (
        "eval",
        row.get("arm"),
        row.get("run_seed"),
        row.get("checkpoint_step"),
        row.get("split"),
        row.get("task_id"),
        row.get("sample_index"),
    )


def resume_checkpoint(completed_steps: Iterable[int]) -> int:
    """Last contiguous completed step, or 0 when training has not started."""
    done = set(completed_steps)
    step = 0
    while step + 1 in done:
        step += 1
    if done - set(range(1, step + 1)):
        raise ValueError("train progress has a gap")
    return step


def protocol_document() -> dict[str, object]:
    examples = comparison_train_examples()
    return {
        "advantage": {
            "epsilon": ADVANTAGE_EPSILON,
            "group_size": GROUP_SIZE,
            "standardize": "population_std",
        },
        "arms": list(ARMS),
        "authorized": True,
        "backends_are_replicates": False,
        "bootstrap_alpha": 0.05,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confirm_split_sha256": split_sha256(),
        "curve_checkpoints": list(CURVE_STEPS),
        "curve_ks": list(CURVE_KS),
        "decision_rule": (
            "positive_contrast when the confirm bootstrap low end is above 0, "
            "negative_contrast when the high end is below 0, otherwise "
            "interval_includes_zero. Pack rows cannot change the decision."
        ),
        "decision_split": "confirm",
        "does_not_establish": (
            "A positive contrast is a difference on the confirm split under this "
            "budget. It does not promote weights and it does not authorize a "
            "larger step count after the interval is observed."
        ),
        "eval_samples_per_task": EVAL_SAMPLES,
        "format_sft_target": "correct strict line #### <value>",
        "frozen_before_arm_output": True,
        "heldout_pack_sha256": HELDOUT_PACK_SHA256,
        "learning_rate": LEARNING_RATE,
        "lora": {"dropout": 0.0, "keys": list(LORA_KEYS), "rank": 8, "scale": 20.0},
        "matched_budget": "optimizer_steps",
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "models": {
            "mlx": {
                "package": "mlx-lm==0.32.0",
                "repo_id": "mlx-community/Qwen2.5-3B-Instruct-4bit",
                "revision": "4f83f8f146fdf28b512a06562b671d7af4fab457",
            },
            "transformers": {
                "package_note": "Runtime torch plus peft and transformers. Not a replicate of the 4-bit mlx checkpoint.",
                "repo_id": "Qwen/Qwen2.5-3B-Instruct",
                "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            },
        },
        "num_layers": 1,
        "optimizer": "adam",
        "optimizer_steps": OPTIMIZER_STEPS,
        "planning_optimizer_seconds": PLANNING_OPTIMIZER_SECONDS,
        "planning_seconds_per_step": PLANNING_SECONDS_PER_STEP,
        "primary_contrast": (
            "correctness_reward_rl minus format_sft, unbiased strict pass@1, "
            "confirm tasks pooled across the five run seeds, paired bootstrap"
        ),
        "primary_k": PRIMARY_K,
        "protocol_id": PROTOCOL_ID,
        "rl_generations": RL_GENERATIONS,
        "run_seeds": list(RUN_SEEDS),
        "scores_dev_split": False,
        "smoke_manifest_sha256": SMOKE_MANIFEST_SHA256,
        "smoke_peak_memory_bytes": SMOKE_PEAK_MEMORY_BYTES,
        "smoke_wall_seconds": SMOKE_WALL_SECONDS,
        "temperature": TEMPERATURE,
        "train_seeds": [int(row["seed"]) for row in examples],
        "uses_confirm_for_training": False,
        "uses_consumed_smoke_row": False,
        "uses_pack_for_training": False,
    }


def canonical_protocol_bytes() -> bytes:
    return (json.dumps(protocol_document(), indent=2, sort_keys=True) + "\n").encode()


def validate_protocol(protocol: Mapping[str, object]) -> None:
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("authorized") is not True:
        raise ValueError("unexpected comparison protocol")
    if protocol.get("optimizer_steps") != OPTIMIZER_STEPS or protocol.get("run_seeds") != list(RUN_SEEDS):
        raise ValueError("step count or seeds drifted")
    if protocol.get("decision_split") != "confirm" or protocol.get("uses_pack_for_training") is not False:
        raise ValueError("the pack is not a training set and not the decision split")
    if protocol.get("uses_confirm_for_training") is not False or protocol.get("scores_dev_split") is not False:
        raise ValueError("confirm and dev are not training data")
    if protocol.get("train_seeds") != [int(row["seed"]) for row in comparison_train_examples()]:
        raise ValueError("train seed list drifted")
    if protocol.get("smoke_manifest_sha256") != SMOKE_MANIFEST_SHA256:
        raise ValueError("smoke manifest pin drifted")
    if protocol.get("heldout_pack_sha256") != HELDOUT_PACK_SHA256:
        raise ValueError("held-out pack pin drifted")
    if protocol.get("confirm_split_sha256") != split_sha256():
        raise ValueError("confirm split pin drifted")
    if protocol.get("frozen_before_arm_output") is not True:
        raise ValueError("the comparison protocol was not frozen before arm output")
    train_seeds = protocol.get("train_seeds")
    if not isinstance(train_seeds, list) or any(seed == 1029953 for seed in train_seeds):
        raise ValueError("the consumed smoke row is in the train list")


def _index_eval(rows: Sequence[Mapping[str, object]]) -> dict[tuple[object, ...], list[Mapping[str, object]]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("kind") != "eval":
            continue
        key = (
            row.get("arm"),
            row.get("checkpoint_step"),
            row.get("split"),
            row.get("task_id"),
            row.get("run_seed"),
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def _regraded(task: Mapping[str, str], row: Mapping[str, object]) -> dict[str, bool]:
    output = row.get("output")
    if not isinstance(output, str):
        raise ValueError("eval row is missing output text")
    return score_output(task["expression"], output)


def _pass_at(
    grouped: Mapping[tuple[object, ...], Sequence[Mapping[str, object]]],
    tasks: Sequence[Mapping[str, str]],
    *,
    arm: str,
    checkpoint_step: int,
    split: str,
    k: int,
    run_seed: int | None = None,
) -> float:
    counts: list[tuple[int, int]] = []
    seeds = (run_seed,) if run_seed is not None else RUN_SEEDS
    for task in tasks:
        samples: list[Mapping[str, object]] = []
        for seed in seeds:
            samples.extend(grouped.get((arm, checkpoint_step, split, task["task_id"], seed), ()))
        expected = EVAL_SAMPLES if run_seed is not None else EVAL_SAMPLES * len(RUN_SEEDS)
        if len(samples) != expected:
            raise ValueError(f"missing samples for {arm} {split} {task['task_id']}")
        correct = 0
        for sample in samples:
            scored = _regraded(task, sample)
            correct += int(scored["strict_correct"])
        counts.append((expected, correct))
    return mean_unbiased_pass_at_k(counts, k)


def contrast_decision(low: float, high: float) -> str:
    if low > 0:
        return "positive_contrast"
    if high < 0:
        return "negative_contrast"
    return "interval_includes_zero"


def summarize_comparison(
    confirm: Sequence[Mapping[str, str]],
    pack: Sequence[Mapping[str, str]],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    expected = {unit_key(unit) for unit in iter_train_units()}
    expected.update(unit_key(unit) for unit in iter_eval_units(confirm, pack))
    seen: set[tuple[object, ...]] = set()
    for row in rows:
        key = row_key(row)
        if key in seen:
            raise ValueError(f"duplicate progress row {key}")
        seen.add(key)
    if seen != expected:
        raise ValueError("progress is not the full frozen grid")
    grouped = _index_eval(rows)
    curves: list[dict[str, object]] = []
    for split, tasks in (("confirm", confirm), ("pack", pack)):
        for arm in ARMS:
            for checkpoint_step in checkpoints_for(arm):
                if split == "pack" and checkpoint_step != final_checkpoint(arm):
                    continue
                point: dict[str, object] = {
                    "arm": arm,
                    "checkpoint_step": checkpoint_step,
                    "split": split,
                }
                for k in CURVE_KS:
                    point[f"pass_at_{k}"] = _pass_at(
                        grouped,
                        tasks,
                        arm=arm,
                        checkpoint_step=checkpoint_step,
                        split=split,
                        k=k,
                    )
                curves.append(point)
    left = [
        unbiased_pass_at_k(
            EVAL_SAMPLES * len(RUN_SEEDS),
            sum(
                int(_regraded(task, sample)["strict_correct"])
                for seed in RUN_SEEDS
                for sample in grouped[( "format_sft", OPTIMIZER_STEPS, "confirm", task["task_id"], seed)]
            ),
            PRIMARY_K,
        )
        for task in confirm
    ]
    right = [
        unbiased_pass_at_k(
            EVAL_SAMPLES * len(RUN_SEEDS),
            sum(
                int(_regraded(task, sample)["strict_correct"])
                for seed in RUN_SEEDS
                for sample in grouped[
                    ("correctness_reward_rl", OPTIMIZER_STEPS, "confirm", task["task_id"], seed)
                ]
            ),
            PRIMARY_K,
        )
        for task in confirm
    ]
    interval = paired_bootstrap_interval(
        left,
        right,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
        alpha=0.05,
    )
    successes: dict[str, list[bool]] = {"format_sft": [], "correctness_reward_rl": []}
    two_tier_rows: list[dict[str, bool]] = []
    for task in confirm:
        for arm in ("format_sft", "correctness_reward_rl"):
            samples = [
                sample
                for seed in RUN_SEEDS
                for sample in grouped[(arm, OPTIMIZER_STEPS, "confirm", task["task_id"], seed)]
            ]
            flags = [_regraded(task, sample) for sample in samples]
            successes[arm].append(any(flag["strict_correct"] for flag in flags))
            if arm == "correctness_reward_rl":
                two_tier_rows.extend(flags)
    discordant_b = sum(
        (not left_ok) and right_ok for left_ok, right_ok in zip(successes["format_sft"], successes["correctness_reward_rl"], strict=True)
    )
    discordant_c = sum(
        left_ok and (not right_ok) for left_ok, right_ok in zip(successes["format_sft"], successes["correctness_reward_rl"], strict=True)
    )
    by_seed = []
    for seed in RUN_SEEDS:
        by_seed.append(
            {
                "format_sft_pass_at_1": _pass_at(
                    grouped,
                    confirm,
                    arm="format_sft",
                    checkpoint_step=OPTIMIZER_STEPS,
                    split="confirm",
                    k=1,
                    run_seed=seed,
                ),
                "correctness_reward_rl_pass_at_1": _pass_at(
                    grouped,
                    confirm,
                    arm="correctness_reward_rl",
                    checkpoint_step=OPTIMIZER_STEPS,
                    split="confirm",
                    k=1,
                    run_seed=seed,
                ),
                "run_seed": seed,
            }
        )
    return {
        "by_seed_confirm_final": by_seed,
        "curves": curves,
        "decision": contrast_decision(float(interval["low"]), float(interval["high"])),
        "decision_split": "confirm",
        "descriptive_two_tier_correctness_arm": summarize_two_tier(two_tier_rows),
        "mcnemar": mcnemar_exact(discordant_b, discordant_c),
        "paired_bootstrap": interval,
        "promotes_weights": False,
        "protocol_id": PROTOCOL_ID,
    }
