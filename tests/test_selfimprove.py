from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from minigrpo.expr import evaluate_expression
from minigrpo.grpo import correctness_reward, format_reward
from minigrpo.splits import build_split
from selfimprove.loop import read_jsonl, run_rounds
from selfimprove.propose import adjust_difficulty, keep_value, propose
from selfimprove.reports import DEV_COUNT, DevReport, dev_gate, monitor_signals, to_evaluation_report
from selfimprove.verifier import false_accept_rate, verdict_accepts

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_selfimprove.py"


def _scores(hits: int) -> dict[str, float]:
    return {f"t{index}": (1.0 if index < hits else 0.0) for index in range(DEV_COUNT)}


def _report(policy: str, hits: int, *, format_reward_value: float, train_reward: float, false_accept: float) -> DevReport:
    return DevReport(
        policy_id=policy,
        scores=_scores(hits),
        format_compliance=format_reward_value,
        false_accept_rate=false_accept,
        format_reward=format_reward_value,
        train_reward=train_reward,
    )


class Scripted:
    def __init__(self, reports: list[DevReport], rates: list[float]) -> None:
        self.reports = list(reports)
        self.rates = list(rates)
        self.rollbacks = 0
        self.commits = 0
        self.held = ""

    def restore(self, checkpoint: str) -> None:
        self.held = checkpoint

    def probe(self, problems: list[object]) -> float:
        assert problems
        return self.rates.pop(0)

    def train(self, problems: list[object], steps: int, reward_mode: str, round_index: int) -> tuple[float, float]:
        report = self.reports[0]
        return report.train_reward, report.format_reward

    def measure(self, policy_id: str) -> DevReport:
        report = self.reports.pop(0)
        if report.policy_id != policy_id:
            raise AssertionError(policy_id)
        return report

    def commit(self) -> None:
        self.commits += 1
        self.held = "promoted"

    def rollback(self) -> None:
        self.rollbacks += 1

    def checkpoint(self) -> str:
        return self.held


def test_proposer_keeps_only_recomputable_problems() -> None:
    assert keep_value("1 / 2") is None
    assert keep_value("1 + 2") == 3
    batch = propose(1, 2, 4)
    assert [problem.expression for problem in propose(1, 2, 4)] == [problem.expression for problem in batch]
    confirm = {task.expression for task in build_split("confirm")}
    dev = {task.expression for task in build_split("dev")}
    for problem in batch:
        assert evaluate_expression(problem.expression) == problem.value
        assert problem.expression not in confirm
        assert problem.expression not in dev
        assert problem.seed >= 5_000_000
    assert adjust_difficulty(2, 0.9) == 3
    assert adjust_difficulty(2, 0.1) == 1
    assert adjust_difficulty(2, 0.5) == 2
    assert adjust_difficulty(3, 0.95) == 3
    assert adjust_difficulty(1, 0.05) == 1


def test_format_reward_is_caught_by_the_existing_gate(tmp_path: Path) -> None:
    assert format_reward("#### 1") == 1.0
    assert correctness_reward("1 + 1", "#### 1") == 0.0
    base = _report("round-0", 16, format_reward_value=0.2, train_reward=0.2, false_accept=0.0)
    hacked = _report("round-1", 16, format_reward_value=1.0, train_reward=1.0, false_accept=0.75)
    better = _report("round-2", 18, format_reward_value=0.4, train_reward=0.5, false_accept=0.0)
    assert "format_only_reward_rise" in monitor_signals(base, hacked)
    assert "self_verifier_disagreement" in monitor_signals(base, hacked)
    decision = dev_gate().decide(to_evaluation_report(base), to_evaluation_report(hacked))
    assert not decision.accepted
    assert "insufficient_primary_gain" in decision.reasons
    assert "verifier_disagreement" in decision.reasons
    solver = Scripted([base, hacked, better], [0.9, 0.5])
    records = run_rounds(
        solver,
        output=tmp_path,
        rounds=2,
        problems_per_round=2,
        steps_per_round=1,
        reward_mode="format",
    )
    assert solver.rollbacks == 1
    assert solver.commits == 1
    assert records[0]["accepted"] is False
    assert records[0]["promoted_policy"] == "round-0"
    assert records[0]["next_difficulty"] == 3
    assert "format_only_reward_rise" in records[0]["signals"]
    assert records[1]["accepted"] is True
    assert records[1]["promoted_policy"] == "round-2"
    rows = read_jsonl(tmp_path / "rounds.jsonl")
    assert [row["kind"] for row in rows] == ["baseline", "round", "round"]
    again = run_rounds(
        Scripted([], []),
        output=tmp_path,
        rounds=2,
        problems_per_round=2,
        steps_per_round=1,
        reward_mode="format",
    )
    assert again == []


def test_heldout_divergence_rolls_back() -> None:
    base = _report("round-0", 16, format_reward_value=0.2, train_reward=0.2, false_accept=0.0)
    drifted = _report("round-1", 8, format_reward_value=0.2, train_reward=0.9, false_accept=0.0)
    assert monitor_signals(base, drifted) == ("heldout_divergence",)
    decision = dev_gate().decide(to_evaluation_report(base), to_evaluation_report(drifted))
    assert not decision.accepted
    assert "insufficient_primary_gain" in decision.reasons
    assert verdict_accepts("#### 4", "#### 4")
    assert not verdict_accepts("#### 4", "#### 5")
    assert false_accept_rate([(False, True), (False, False), (True, True)]) == 0.5


def test_check_command() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import mlx" not in source.split("def main")[0]
    assert "import torch" not in source.split("def main")[0]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "mlx", "--check"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["confirm_tasks_used"] == 0
    assert payload["dev_tasks"] == 32
    assert payload["min_eval_examples"] == 32
