"""N rounds of proposed problems, a dev measurement, and a promotion decision.

A rejected candidate is rolled back. The candidate directory stays on disk.
Confirm tasks are not scored.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from minigrpo.expr import evaluate_expression
from selfimprove.propose import Problem, adjust_difficulty, propose
from selfimprove.reports import DEV_COUNT, DevReport, dev_gate, monitor_signals, to_evaluation_report


class Solver(Protocol):
    def restore(self, checkpoint: str) -> None: ...

    def probe(self, problems: list[Problem]) -> float: ...

    def train(self, problems: list[Problem], steps: int, reward_mode: str, round_index: int) -> tuple[float, float]: ...

    def measure(self, policy_id: str) -> DevReport: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def checkpoint(self) -> str: ...


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise SystemExit(f"{path} contains a non-object row")
            rows.append(payload)
    return rows


def _scores(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict) or len(raw) != DEV_COUNT:
        raise SystemExit("round record does not contain the dev scores")
    return {str(key): float(value) for key, value in raw.items()}


def report_from_mapping(row: dict[str, object], prefix: str) -> DevReport:
    return DevReport(
        policy_id=str(row[f"{prefix}policy"]),
        scores=_scores(row[f"{prefix}scores"]),
        format_compliance=float(row[f"{prefix}format_compliance"]),
        false_accept_rate=float(row[f"{prefix}false_accept_rate"]),
        format_reward=float(row[f"{prefix}format_reward"]),
        train_reward=float(row[f"{prefix}train_reward"]),
    )


def _row_for(report: DevReport, prefix: str) -> dict[str, object]:
    return {
        f"{prefix}false_accept_rate": report.false_accept_rate,
        f"{prefix}format_compliance": report.format_compliance,
        f"{prefix}format_reward": report.format_reward,
        f"{prefix}policy": report.policy_id,
        f"{prefix}scores": report.scores,
        f"{prefix}train_reward": report.train_reward,
    }


def _require_recomputable(problems: list[Problem]) -> None:
    for problem in problems:
        if evaluate_expression(problem.expression) != problem.value:
            raise RuntimeError(f"proposer kept an expression that does not recompute: {problem.expression}")
        if problem.seed < 5_000_000:
            raise RuntimeError("proposer seed collided with the seed-split ranges")


def run_rounds(
    solver: Solver,
    *,
    output: Path,
    rounds: int,
    problems_per_round: int,
    steps_per_round: int,
    reward_mode: str,
) -> list[dict[str, object]]:
    if rounds < 1 or problems_per_round < 1 or steps_per_round < 1:
        raise ValueError("rounds, problems, and steps must be positive")
    if reward_mode not in {"correctness", "format"}:
        raise ValueError("reward_mode must be correctness or format")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "rounds.jsonl"
    existing = read_jsonl(path)
    baseline_rows = [row for row in existing if row.get("kind") == "baseline"]
    round_rows = [row for row in existing if row.get("kind") == "round"]
    if round_rows and not baseline_rows:
        raise SystemExit("rounds.jsonl is missing the baseline")
    if [int(row["round"]) for row in round_rows] != list(range(1, len(round_rows) + 1)):
        raise SystemExit("rounds.jsonl rounds are missing or duplicated")
    if baseline_rows:
        incumbent = report_from_mapping(baseline_rows[-1], "")
        solver.restore(str(baseline_rows[-1].get("checkpoint", "")))
    else:
        incumbent = solver.measure("round-0")
        solver.restore("")
        append_jsonl(path, {"checkpoint": "", "kind": "baseline", **_row_for(incumbent, "")})
    difficulty = 2
    if round_rows:
        last = round_rows[-1]
        difficulty = int(last["next_difficulty"])
        incumbent = report_from_mapping(last, "promoted_")
        solver.restore(str(last.get("promoted_checkpoint", "")))
    gate = dev_gate()
    written: list[dict[str, object]] = []
    for index in range(len(round_rows) + 1, rounds + 1):
        problems = propose(index, difficulty, problems_per_round)
        _require_recomputable(problems)
        solve_rate = solver.probe(problems)
        train_reward, format_reward = solver.train(problems, steps_per_round, reward_mode, index)
        candidate = solver.measure(f"round-{index}").with_rewards(train_reward, format_reward)
        signals = monitor_signals(incumbent, candidate)
        decision = gate.decide(to_evaluation_report(incumbent), to_evaluation_report(candidate))
        if decision.accepted:
            solver.commit()
            promoted = candidate
        else:
            solver.rollback()
            promoted = incumbent
        next_difficulty = adjust_difficulty(difficulty, solve_rate)
        record: dict[str, object] = {
            "accepted": decision.accepted,
            "difficulty": difficulty,
            "kind": "round",
            "next_difficulty": next_difficulty,
            "promoted_checkpoint": solver.checkpoint(),
            "reasons": list(decision.reasons),
            "reward_mode": reward_mode,
            "round": index,
            "signals": list(signals),
            "solve_rate": solve_rate,
        }
        record.update(_row_for(candidate, "candidate_"))
        record.update(_row_for(promoted, "promoted_"))
        append_jsonl(path, record)
        written.append(record)
        incumbent = promoted
        difficulty = next_difficulty
    return written
