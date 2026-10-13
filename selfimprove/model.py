"""Drive minigrpo from the proposer. Model imports stay inside the builder."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from minigrpo.expr import parse_final_integer
from minigrpo.grpo import correctness_reward
from minigrpo.loop import RunConfig, checkpoint_dir, dev_sample_seed, execute, read_jsonl
from minigrpo.splits import dev_tasks
from selfimprove.propose import Problem
from selfimprove.reports import DEV_COUNT, DevReport
from selfimprove.verifier import false_accept_rate, verdict_accepts, verifier_prompt


class ModelSolver:
    def __init__(self, trainer: object, config: RunConfig, output: Path) -> None:
        self.trainer = trainer
        self.config = config
        self.output = output
        self.dev = dev_tasks()
        if len(self.dev) != DEV_COUNT:
            raise RuntimeError("dev split does not have 32 tasks")
        self.incumbent: Path | None = None
        self.candidate: Path | None = None

    def restore(self, checkpoint: str) -> None:
        self.incumbent = Path(checkpoint) if checkpoint else None
        self.candidate = None

    def checkpoint(self) -> str:
        return "" if self.incumbent is None else str(self.incumbent)

    def _activate(self, adapter: Path | None) -> None:
        activate = getattr(self.trainer, "activate")
        activate(adapter)

    def _generate(self, prompt: str, seed: int) -> str:
        generate = getattr(self.trainer, "generate")
        text = generate(prompt, seed)
        if not isinstance(text, str):
            raise RuntimeError("generator did not return text")
        return text

    def probe(self, problems: list[Problem]) -> float:
        self._activate(self.incumbent)
        hits = 0
        for index, problem in enumerate(problems):
            text = self._generate(problem.prompt, self.config.seed + 10_000 + index)
            hits += int(correctness_reward(problem.expression, text) == 1.0)
        return hits / len(problems)

    def train(self, problems: list[Problem], steps: int, reward_mode: str, round_index: int) -> tuple[float, float]:
        round_dir = self.output / f"round-{round_index:04d}"
        config = replace(
            self.config,
            output=str(round_dir),
            steps=steps,
            reward_mode=reward_mode,
            eval_dev=False,
            init_adapter="" if self.incumbent is None else str(self.incumbent),
        )
        execute(self.trainer, config, [problem.as_task() for problem in problems], [])
        rows = [row for row in read_jsonl(round_dir / "metrics.jsonl") if row.get("kind") == "train"]
        if len(rows) != steps:
            raise RuntimeError("round did not finish every optimizer step")
        self.candidate = checkpoint_dir(round_dir, steps)
        reward = sum(float(row["reward_mean"]) for row in rows) / len(rows)
        formatted = sum(float(row["format_compliance_rate"]) for row in rows) / len(rows)
        return reward, formatted

    def measure(self, policy_id: str) -> DevReport:
        adapter = None if policy_id == "round-0" else self.candidate
        if policy_id == "round-0" and self.incumbent is not None:
            adapter = self.incumbent
        self._activate(adapter)
        scores: dict[str, float] = {}
        judgements: list[tuple[bool, bool]] = []
        compliant = 0
        for index, task in enumerate(self.dev):
            text = self._generate(task.prompt, dev_sample_seed(self.config.seed, index))
            correct = correctness_reward(task.expression, text) == 1.0
            compliant += int(parse_final_integer(text) is not None)
            scores[str(task.seed)] = 1.0 if correct else 0.0
            verdict = self._generate(
                verifier_prompt(task.expression, text),
                dev_sample_seed(self.config.seed, index) + 100_000,
            )
            judgements.append((correct, verdict_accepts(text, verdict)))
        return DevReport(
            policy_id=policy_id,
            scores=scores,
            format_compliance=compliant / len(self.dev),
            false_accept_rate=false_accept_rate(judgements),
            format_reward=0.0,
            train_reward=0.0,
        )

    def commit(self) -> None:
        if self.candidate is None:
            raise RuntimeError("there is no candidate to promote")
        self.incumbent = self.candidate

    def rollback(self) -> None:
        self.candidate = None


def build_trainer(backend: str, config: RunConfig) -> object:
    if backend == "mlx":
        from minigrpo.mlx_train import MlxTrainer

        return MlxTrainer(config)
    if backend == "torch":
        from minigrpo.torch_train import TorchTrainer

        return TorchTrainer(config)
    raise SystemExit(f"unknown backend {backend}")
