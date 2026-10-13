from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from vare.format_vs_correctness import PROTOCOL_ID as REFUSAL_ID
from vare.format_vs_correctness_v2 import (
    GROUP_SIZE,
    OPTIMIZER_STEPS,
    PLANNING_OPTIMIZER_SECONDS,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    RL_GENERATIONS,
    RUN_SEEDS,
    canonical_protocol_bytes,
    comparison_train_examples,
    confirm_eval_tasks,
    contrast_decision,
    format_reward,
    group_advantages,
    iter_eval_units,
    iter_train_units,
    pack_eval_tasks,
    resume_checkpoint,
    rl_rewards,
    rollout_seed,
    summarize_comparison,
    validate_protocol,
)
from vare.heldout_exact_integer import evaluate_expression

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/format_vs_correctness_comparison_v2.json"
PACK = ROOT / "benchmarks/heldout/exact_integer_v1/tasks.jsonl"
SCRIPT = ROOT / "scripts/run_format_vs_correctness_comparison_v2.py"
REFUSAL = ROOT / "scripts/format_vs_correctness_comparison_v1.py"


def _output(expression: str, correct: bool) -> str:
    value = evaluate_expression(expression)
    if correct:
        return f"#### {value}\n"
    return "nope"


def _grid(confirm_correct: set[str], pack_correct: set[str]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, object]]]:
    confirm = confirm_eval_tasks()
    pack = pack_eval_tasks(PACK.read_text(encoding="utf-8"))
    tasks = {row["task_id"]: row for row in [*confirm, *pack]}
    rows: list[dict[str, object]] = []
    for unit in iter_train_units():
        rows.append(
            {
                "arm": unit["arm"],
                "kind": "train",
                "loss": 1.0,
                "run_seed": unit["run_seed"],
                "step": unit["step"],
            }
        )
    for unit in iter_eval_units(confirm, pack):
        task_id = str(unit["task_id"])
        wanted = confirm_correct if unit["split"] == "confirm" else pack_correct
        rows.append(
            {
                "arm": unit["arm"],
                "checkpoint_step": unit["checkpoint_step"],
                "kind": "eval",
                "output": _output(tasks[task_id]["expression"], task_id in wanted or unit["arm"] in wanted),
                "run_seed": unit["run_seed"],
                "sample_index": unit["sample_index"],
                "split": unit["split"],
                "task_id": task_id,
            }
        )
    return confirm, pack, rows


def test_protocol_bytes_budget_and_disjoint_train_rows():
    blob = PROTOCOL.read_bytes()
    assert blob == canonical_protocol_bytes()
    assert hashlib.sha256(blob).hexdigest() == PROTOCOL_SHA256
    protocol = json.loads(blob.decode("utf-8"))
    validate_protocol(protocol)
    assert "3.004" not in blob.decode("utf-8")
    assert protocol["optimizer_steps"] == 16
    assert protocol["decision_split"] == "confirm"
    assert protocol["uses_pack_for_training"] is False
    assert protocol["planning_optimizer_seconds"] == 2880 == PLANNING_OPTIMIZER_SECONDS
    assert protocol["rl_generations"] == 480 == RL_GENERATIONS
    assert protocol["format_sft_target"] == "correct strict line #### <value>"
    examples = comparison_train_examples()
    assert [row["seed"] for row in examples] == protocol["train_seeds"]
    assert 1029953 not in protocol["train_seeds"]
    assert examples[0]["seed"] == 1032499
    assert len(examples) == OPTIMIZER_STEPS
    confirm = {row["expression"] for row in confirm_eval_tasks()}
    pack = {row["expression"] for row in pack_eval_tasks(PACK.read_text(encoding="utf-8"))}
    assert {row["expression"] for row in examples}.isdisjoint(confirm)
    assert {row["expression"] for row in examples}.isdisjoint(pack)
    assert len(iter_train_units()) == 320
    assert len(iter_eval_units(confirm_eval_tasks(), pack_eval_tasks(PACK.read_text(encoding="utf-8")))) == 17920


def test_rewards_keep_the_group_and_do_not_read_text_for_random():
    assert format_reward("#### 3") == 1.0
    assert format_reward("The result is 3.") == 0.0
    assert rl_rewards("format_reward_rl", "1 + 1", ["#### 2", "nope"], 20261016, 1) == [1.0, 0.0]
    assert rl_rewards("correctness_reward_rl", "1 + 1", ["#### 2", "#### 3"], 20261016, 1) == [1.0, 0.0]
    first = rl_rewards("random_reward_rl", "1 + 1", ["#### 2", "nope"], 20261016, 1)
    second = rl_rewards("random_reward_rl", "9 - 9", ["other", "text"], 20261016, 1)
    assert first == second
    assert len(first) == GROUP_SIZE
    assert group_advantages([1.0, 1.0]) == [0.0, 0.0]
    assert group_advantages([0.0, 0.0]) == [0.0, 0.0]
    scored = group_advantages([1.0, 0.0])
    assert len(scored) == 2
    assert scored[0] > 0 > scored[1]
    try:
        group_advantages([1.0])
    except ValueError as exc:
        assert "intact" in str(exc)
    else:
        raise AssertionError("a short group was accepted")


def test_resume_rejects_a_gap_and_sample_seeds_do_not_collide():
    assert resume_checkpoint([]) == 0
    assert resume_checkpoint([1, 2, 3]) == 3
    try:
        resume_checkpoint([1, 3])
    except ValueError as exc:
        assert "gap" in str(exc)
    else:
        raise AssertionError("a train gap was accepted")
    confirm = confirm_eval_tasks()
    pack = pack_eval_tasks(PACK.read_text(encoding="utf-8"))
    rollouts = {
        rollout_seed(arm, seed, step, index)
        for arm in ("format_reward_rl", "correctness_reward_rl", "random_reward_rl")
        for seed in RUN_SEEDS
        for step in range(1, OPTIMIZER_STEPS + 1)
        for index in range(GROUP_SIZE)
    }
    evals = {int(unit["sample_seed"]) for unit in iter_eval_units(confirm, pack)}
    assert len(rollouts) == RL_GENERATIONS
    assert rollouts.isdisjoint(evals)


def test_decision_uses_confirm_and_ignores_the_pack_curve():
    confirm, pack, tied = _grid(set(), {"correctness_reward_rl"})
    tied_summary = summarize_comparison(confirm, pack, tied)
    assert tied_summary["decision"] == "interval_includes_zero"
    assert tied_summary["decision_split"] == "confirm"
    assert tied_summary["promotes_weights"] is False
    pack_curve = {
        (row["arm"], row["pass_at_1"])
        for row in tied_summary["curves"]
        if row["split"] == "pack" and row["checkpoint_step"] == OPTIMIZER_STEPS
    }
    assert ("correctness_reward_rl", 1.0) in pack_curve
    assert ("format_sft", 0.0) in pack_curve
    confirm, pack, separated = _grid({"correctness_reward_rl"}, set())
    separated_summary = summarize_comparison(confirm, pack, separated)
    assert separated_summary["decision"] == "positive_contrast"
    assert contrast_decision(0.1, 0.2) == "positive_contrast"
    assert contrast_decision(-0.2, -0.1) == "negative_contrast"
    descriptive = separated_summary["descriptive_two_tier_correctness_arm"]
    assert descriptive["gate_result"] is False
    assert descriptive["posthoc"] is True


def test_check_does_not_train_and_v1_still_exits_2():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "mlx", "--output", "/tmp/vare-fvc-check", "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    body = json.loads(proc.stdout)
    assert body["status"] == "checked"
    assert body["train_units"] == 320
    assert body["eval_units"] == 17920
    assert body["protocol_id"] == PROTOCOL_ID
    assert not Path("/tmp/vare-fvc-check/progress.jsonl").exists()
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--steps" not in text
    assert "--limit" not in text
    assert "3.004" not in text
    refusal = subprocess.run([sys.executable, str(REFUSAL)], check=False, capture_output=True, text=True)
    assert refusal.returncode == 2
    assert json.loads(refusal.stdout)["protocol_id"] == REFUSAL_ID
