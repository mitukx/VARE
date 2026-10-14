"""A100 reward-signal ablation and a short promotion-gate loop.

FULL is planned to finish within 2.5 hours on one A100. QUICK is a small grid
that measures generation time and writes ``throughput.json``. A later FULL run
reads that file and lowers the step count, in multiples of the eval interval,
until the projected time fits. The locked ``run.json`` is not revised after
the run starts.

The random arm draws Bernoulli rewards at the probe accuracy. Those draws do
not depend on the completion text. Reference KL is logged with a coefficient
of 0. The base arm does not take an optimizer step. Confirm tasks are not
loaded. GSM8K is the Part A held-out set; the CPU smoke uses a local fixture
instead of downloading it.

Part B reuses ``PromotionGate`` with ``min_eval_examples`` 32 and the paired
bootstrap turned off. The scripted self-verifier case is the deterministic
rejection check. A GPU round records whether that short run actually diverged.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from minigrpo.expr import evaluate_expression, parse_final_integer
from minigrpo.grpo import (
    TokenRow,
    correctness_reward,
    format_reward,
    group_advantages,
    incorrect_reward,
    reference_kl,
    sequence_loss,
)
from minigrpo.loop import MAX_STEPS, Rollout, RunConfig, append_jsonl, read_jsonl
from minigrpo.splits import SplitTask, build_split, derive_seeds, dev_tasks
from selfimprove.propose import adjust_difficulty, propose
from selfimprove.reports import DEV_COUNT, DevReport, dev_gate, monitor_signals, to_evaluation_report
from selfimprove.verifier import false_accept_rate, verdict_accepts, verifier_prompt

Z95 = 1.959963984540054
PRIOR_SEC_TRAIN = 0.35
PRIOR_SEC_EVAL = 0.20
FULL_BUDGET_SECONDS = 9000.0
PART_B_RESERVE_SECONDS = 1500.0
FULL_STEP_FLOOR = 40
HELDOUT_FLOOR = 40
PROBE_N = 16
EMPTY_MARKER = "(empty)"
HELDOUT_SEED = 20261031
ARMS = ("correctness", "random", "incorrect", "format", "base")
PLAN_VERSION = 1

QWEN_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LLAMA_REPO = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
SMOL_REPO = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
SMOL_REVISION = "31b70e2e869a7173562077fd711b654946d38674"

GSM_INSTRUCTION = (
    "Solve the grade-school math problem.\n"
    "{question}\n"
    "Write the result on the final line in exactly this form: #### <integer>\n"
    "The final line must contain no other text."
)
_GSM_LINE = re.compile(r"####\s*(-?\d+)")
_ARM_COLOR = {
    "correctness": "#1f4e79",
    "random": "#b36b00",
    "incorrect": "#9b2226",
    "format": "#2d6a4f",
    "base": "#6c757d",
}


@dataclass(frozen=True, slots=True)
class HeldoutItem:
    prompt: str
    gold: int
    source: str


@dataclass
class Meter:
    train_seconds: float = 0.0
    train_samples: int = 0
    eval_seconds: float = 0.0
    eval_samples: int = 0

    def add_train(self, seconds: float, samples: int) -> None:
        self.train_seconds += seconds
        self.train_samples += samples

    def add_eval(self, seconds: float, samples: int) -> None:
        self.eval_seconds += seconds
        self.eval_samples += samples

    def rates(self) -> tuple[float | None, float | None]:
        train = None if self.train_samples < 1 else self.train_seconds / self.train_samples
        eval_rate = None if self.eval_samples < 1 else self.eval_seconds / self.eval_samples
        return train, eval_rate


def wilson_interval(correct: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n < 1 or correct < 0 or correct > n:
        raise ValueError("wilson inputs are outside the sample")
    phat = correct / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def pick_difficulty(rows: list[tuple[int, float]] | tuple[tuple[int, float], ...]) -> tuple[int, bool]:
    if not rows:
        raise ValueError("difficulty probe is empty")
    band = [(difficulty, accuracy) for difficulty, accuracy in rows if 0.2 <= accuracy <= 0.5]
    pool = band or list(rows)
    chosen = min(pool, key=lambda item: (abs(item[1] - 0.35), item[0]))
    return chosen[0], bool(band)


def assign_random_rewards(model: str, seed: int, step: int, group_size: int, rate: float) -> list[float]:
    if group_size < 1 or not 0.0 <= rate <= 1.0:
        raise ValueError("random reward arguments are outside the supported range")
    rng = random.Random(f"flagship-random:{model}:{seed}:{step}")
    return [1.0 if rng.random() < rate else 0.0 for _ in range(group_size)]


def assign_rewards(
    arm: str,
    expression: str,
    texts: list[str],
    *,
    random_rewards: list[float] | None = None,
) -> list[float]:
    if arm in {"correctness", "base"}:
        return [correctness_reward(expression, text) for text in texts]
    if arm == "incorrect":
        return [incorrect_reward(expression, text) for text in texts]
    if arm == "format":
        return [format_reward(text) for text in texts]
    if arm == "random":
        if random_rewards is None or len(random_rewards) != len(texts):
            raise ValueError("random arm requires one reward per group member")
        return list(random_rewards)
    raise ValueError(f"unknown arm: {arm}")


def normalize_completions(texts: list[str]) -> list[str]:
    if not texts:
        raise RuntimeError("empty batch")
    if all(not text.strip() for text in texts):
        raise RuntimeError("entire batch empty; refusing to drop the group")
    return [text if text.strip() else EMPTY_MARKER for text in texts]


def batch_seed(base: int, step: int, stream: int) -> int:
    """One seed covers a whole generate batch. Members are not seeded apart."""
    if not 0 <= step <= MAX_STEPS or not 0 <= stream < 20:
        raise ValueError("batch seed is outside the supported grid")
    return base + stream * 100_000 + step


def lenient_last_integer(text: str) -> int | None:
    matches = re.findall(r"-?\d+", text)
    if not matches:
        return None
    return int(matches[-1])


def parse_gsm_gold(answer: str) -> int | None:
    matches = _GSM_LINE.findall(answer.replace(",", ""))
    if not matches:
        return None
    return int(matches[-1])


def _model(slug: str, repo: str, revision: str, fallback: dict[str, str] | None = None) -> dict[str, str]:
    row = {"slug": slug, "repo": repo, "revision": revision, "fallback_slug": "", "fallback_repo": "", "fallback_revision": ""}
    if fallback:
        row.update(fallback)
    return row


def _qwen() -> dict[str, str]:
    return _model("qwen25-1.5b", QWEN_REPO, QWEN_REVISION)


def _llama() -> dict[str, str]:
    return _model(
        "llama32-1b",
        LLAMA_REPO,
        LLAMA_REVISION,
        {
            "fallback_slug": "smollm2-1.7b",
            "fallback_repo": SMOL_REPO,
            "fallback_revision": SMOL_REVISION,
        },
    )


def requested_grid(mode: str) -> dict[str, object]:
    if mode == "full":
        return {
            "mode": "full",
            "steps": 100,
            "heldout_n": 200,
            "dev_n": 32,
            "eval_every": 20,
            "group_size": 8,
            "arms": list(ARMS),
            "seeds": [20261031, 20261101],
            "n_models": 2,
            "models": [_qwen(), _llama()],
            "part_b_rounds": 3,
            "part_b_steps": 8,
            "part_b_group": 8,
            "part_b_problems": 4,
            "max_new_tokens": 128,
        }
    if mode == "quick":
        return {
            "mode": "quick",
            "steps": 30,
            "heldout_n": 64,
            "dev_n": 16,
            "eval_every": 6,
            "group_size": 8,
            "arms": list(ARMS),
            "seeds": [20261031],
            "n_models": 1,
            "models": [_qwen()],
            "part_b_rounds": 2,
            "part_b_steps": 4,
            "part_b_group": 4,
            "part_b_problems": 4,
            "max_new_tokens": 128,
        }
    raise ValueError("mode must be quick or full")


def sample_counts(plan: dict[str, object], steps: int, heldout_n: int) -> tuple[int, int]:
    arms = len(plan["arms"])  # type: ignore[arg-type]
    seeds = len(plan["seeds"])  # type: ignore[arg-type]
    models = int(plan["n_models"])
    group = int(plan["group_size"])
    every = int(plan["eval_every"])
    dev_n = int(plan["dev_n"])
    if every < 1 or steps < 1 or steps % every != 0:
        raise ValueError("steps must be a positive multiple of eval_every")
    checkpoints = steps // every + 1
    train = arms * models * seeds * steps * group
    evaluated = arms * models * seeds * checkpoints * (dev_n + heldout_n)
    return train, evaluated


def part_b_samples(plan: dict[str, object]) -> int:
    rounds = int(plan["part_b_rounds"])
    steps = int(plan["part_b_steps"])
    group = int(plan["part_b_group"])
    problems = int(plan["part_b_problems"])
    sources = 2
    measure = sources * (1 + rounds) * DEV_COUNT * 2
    train = sources * rounds * steps * group
    verifier_train = rounds * steps * group
    probe = sources * rounds * problems
    return measure + train + verifier_train + probe


def eta_seconds(plan: dict[str, object], steps: int, heldout_n: int, sec_train: float, sec_eval: float) -> float:
    train, evaluated = sample_counts(plan, steps, heldout_n)
    if plan["mode"] == "full":
        reserve = PART_B_RESERVE_SECONDS
        extra = 0.0
    else:
        reserve = 0.0
        extra = part_b_samples(plan) * sec_eval
    return train * sec_train + evaluated * sec_eval + reserve + extra


def fit_plan(mode: str, sec_train: float, sec_eval: float, rate_source: str) -> dict[str, object]:
    if sec_train <= 0.0 or sec_eval <= 0.0:
        raise ValueError("timing rates must be positive")
    plan = requested_grid(mode)
    steps = int(plan["steps"])
    heldout_n = int(plan["heldout_n"])
    every = int(plan["eval_every"])
    notes: list[str] = []
    if rate_source == "prior":
        notes.append("sec_train and sec_eval are unmeasured priors, not a timing measurement")
    if mode == "full":
        while eta_seconds(plan, steps, heldout_n, sec_train, sec_eval) > FULL_BUDGET_SECONDS and steps - every >= FULL_STEP_FLOOR:
            steps -= every
            notes.append(f"reduced steps to {steps}")
        while eta_seconds(plan, steps, heldout_n, sec_train, sec_eval) > FULL_BUDGET_SECONDS and heldout_n > HELDOUT_FLOOR:
            heldout_n = max(HELDOUT_FLOOR, heldout_n - 40)
            notes.append(f"reduced held-out to {heldout_n}")
        if eta_seconds(plan, steps, heldout_n, sec_train, sec_eval) > FULL_BUDGET_SECONDS:
            notes.append("eta_exceeds_budget")
    train, evaluated = sample_counts(plan, steps, heldout_n)
    fitted = dict(plan)
    fitted.update(
        {
            "budget_seconds": FULL_BUDGET_SECONDS,
            "eta_seconds": eta_seconds(plan, steps, heldout_n, sec_train, sec_eval),
            "eval_samples": evaluated,
            "heldout_n": heldout_n,
            "heldout_seed": HELDOUT_SEED,
            "kl_coef": 0.0,
            "lora": True,
            "notes": notes,
            "part_b_reserve_seconds": PART_B_RESERVE_SECONDS if mode == "full" else 0.0,
            "plan_version": PLAN_VERSION,
            "probe_n": PROBE_N,
            "rate_source": rate_source,
            "sec_eval": sec_eval,
            "sec_train": sec_train,
            "steps": steps,
            "train_samples": train,
        }
    )
    return fitted


def read_rates(path: Path | None) -> tuple[float, float, str]:
    if path is None or not path.exists():
        return PRIOR_SEC_TRAIN, PRIOR_SEC_EVAL, "prior"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("device") != "cuda":
        return PRIOR_SEC_TRAIN, PRIOR_SEC_EVAL, "prior"
    train = payload.get("sec_train")
    eval_rate = payload.get("sec_eval")
    if not isinstance(train, (int, float)) or not isinstance(eval_rate, (int, float)):
        return PRIOR_SEC_TRAIN, PRIOR_SEC_EVAL, "prior"
    if isinstance(train, bool) or isinstance(eval_rate, bool) or train <= 0 or eval_rate <= 0:
        return PRIOR_SEC_TRAIN, PRIOR_SEC_EVAL, "prior"
    return float(train), float(eval_rate), "measured"


def write_rates(path: Path, meter: Meter, *, announce: bool = True) -> None:
    train, eval_rate = meter.rates()
    if train is None or eval_rate is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": "cuda",
        "eval_samples": meter.eval_samples,
        "sec_eval": eval_rate,
        "sec_train": train,
        "train_samples": meter.train_samples,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not announce:
        return
    fitted = fit_plan("full", train, eval_rate, "measured")
    print(
        f"projected_full_seconds {fitted['eta_seconds']:.1f} "
        f"steps {fitted['steps']} heldout {fitted['heldout_n']} rate_source measured"
    )


def plan_report(mode: str, throughput: Path | None) -> dict[str, object]:
    sec_train, sec_eval, source = read_rates(throughput)
    return fit_plan(mode, sec_train, sec_eval, source)


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("mean of an empty series")
    return sum(values) / len(values)


def _kl_mean(rollouts: list[Rollout]) -> float:
    total = 0.0
    count = 0
    for rollout in rollouts:
        if rollout.ref_logp is None:
            continue
        for logp, ref, bit in zip(rollout.old_logp, rollout.ref_logp, rollout.mask, strict=True):
            if bit:
                total += reference_kl(logp, ref)
                count += 1
    return 0.0 if count == 0 else total / count


def _masked_mean(values: tuple[float, ...], mask: tuple[int, ...]) -> float:
    chosen = [value for value, bit in zip(values, mask, strict=True) if bit]
    if not chosen:
        raise RuntimeError("completion mask is empty")
    return sum(chosen) / len(chosen)


def _rows_for(rows: list[dict[str, object]], **wanted: object) -> list[dict[str, object]]:
    chosen = []
    for row in rows:
        if all(row.get(key) == value for key, value in wanted.items()):
            chosen.append(row)
    return chosen


def _train_steps(rows: list[dict[str, object]], slug: str, arm: str, seed: int) -> list[int]:
    steps = [int(row["step"]) for row in _rows_for(rows, kind="train", model=slug, arm=arm, seed=seed)]
    if steps != list(range(1, len(steps) + 1)):
        raise SystemExit(f"train steps for {slug}/{arm}/{seed} are missing or duplicated")
    return steps


def _has_eval(rows: list[dict[str, object]], slug: str, arm: str, seed: int, step: int, split: str) -> bool:
    return bool(_rows_for(rows, kind="eval", model=slug, arm=arm, seed=seed, step=step, split=split))


def _checkpoint(root: Path, step: int) -> Path:
    return root / "checkpoints" / f"step-{step:04d}"


def _require_checkpoint(root: Path, step: int, *, base: bool) -> None:
    path = _checkpoint(root, step)
    if base:
        if not (path / "marker.json").exists():
            raise SystemExit(f"missing base marker for step {step}")
        return
    if not path.exists():
        raise SystemExit(f"missing checkpoint for step {step}")


def _batched(trainer: object, prompts: list[str], seed: int, batch_size: int, *, sample: bool) -> tuple[list[str], float]:
    generate = getattr(trainer, "generate_batch")
    texts: list[str] = []
    started = time.perf_counter()
    for offset in range(0, len(prompts), batch_size):
        chunk = prompts[offset : offset + batch_size]
        raw = generate(chunk, seed + offset, sample=sample)
        if not isinstance(raw, list) or len(raw) != len(chunk):
            raise RuntimeError("generate_batch changed the group size")
        texts.extend(normalize_completions([str(text) for text in raw]))
    elapsed = time.perf_counter() - started
    if len(texts) != len(prompts):
        raise RuntimeError("eval batch changed size")
    return texts, elapsed


def _grade_eval(
    texts: list[str],
    golds: list[int | None],
    expressions: list[str | None],
) -> dict[str, object]:
    if len(texts) != len(golds) or len(texts) != len(expressions):
        raise RuntimeError("eval rows were truncated")
    correct = 0
    compliant = 0
    lenient_hits = 0
    for text, gold, expression in zip(texts, golds, expressions, strict=True):
        parsed = parse_final_integer(text)
        compliant += int(parsed is not None)
        if expression is not None:
            hit = correctness_reward(expression, text) == 1.0
        else:
            hit = parsed is not None and gold is not None and parsed == gold
        correct += int(hit)
        lenient = lenient_last_integer(text)
        lenient_hits += int(lenient is not None and gold is not None and lenient == gold)
    low, high = wilson_interval(correct, len(texts))
    return {
        "accuracy": correct / len(texts),
        "correct": correct,
        "format_compliance_rate": compliant / len(texts),
        "lenient_accuracy": lenient_hits / len(texts),
        "n": len(texts),
        "wilson_high": high,
        "wilson_low": low,
    }


def _shape_check(rollouts: list[Rollout], advantages: list[float]) -> None:
    sequence_loss(
        [
            TokenRow(
                logp=rollout.old_logp,
                logp_old=rollout.old_logp,
                mask=rollout.mask,
                advantage=advantage,
                logp_ref=rollout.ref_logp,
                entropy=rollout.entropy,
            )
            for rollout, advantage in zip(rollouts, advantages, strict=True)
        ],
        kl_coef=0.0,
    )


def run_part_a(
    output: Path,
    *,
    slug: str,
    arms: tuple[str, ...],
    seeds: tuple[int, ...],
    steps: int,
    group_size: int,
    eval_every: int,
    train: list[SplitTask],
    dev: list[SplitTask],
    heldout: list[HeldoutItem],
    random_rate: float,
    trainer_for: Callable[[str, int, Path | None], object],
    meter: Meter | None,
    difficulty: int,
    throughput: Path | None = None,
) -> None:
    if any(task.split == "confirm" for task in train) or any(task.split == "confirm" for task in dev):
        raise RuntimeError("confirm tasks cannot enter the flagship loop")
    confirm = set(derive_seeds("confirm"))
    if {task.seed for task in train} & confirm or {task.seed for task in dev} & confirm:
        raise RuntimeError("confirm seed leaked into the flagship split")
    if steps % eval_every != 0:
        raise ValueError("steps must be a multiple of eval_every")
    metrics = output / "metrics.jsonl"
    eval_steps = [0] + list(range(eval_every, steps + 1, eval_every))
    for arm in arms:
        for seed in seeds:
            root = output / "part_a" / slug / arm / f"seed-{seed}"
            root.mkdir(parents=True, exist_ok=True)
            rows = read_jsonl(metrics)
            done = _train_steps(rows, slug, arm, seed)
            if done and done[-1] > steps:
                raise SystemExit("metrics.jsonl has more steps than the locked plan")
            if 0 not in eval_steps:
                raise RuntimeError("step 0 eval is required")
            if done:
                for step in eval_steps:
                    if step >= done[-1]:
                        continue
                    if not _has_eval(rows, slug, arm, seed, step, "dev") or not _has_eval(rows, slug, arm, seed, step, "heldout"):
                        raise SystemExit(f"missing eval at step {step} after later training")
            frontier = bool(done) and done[-1] % eval_every == 0 and (
                not _has_eval(rows, slug, arm, seed, done[-1], "dev") or not _has_eval(rows, slug, arm, seed, done[-1], "heldout")
            )
            pending_eval = [done[-1]] if frontier else []
            need_train = not done or done[-1] < steps
            if not need_train and not pending_eval:
                continue
            resume = _checkpoint(root, done[-1]) if done else None
            if done:
                _require_checkpoint(root, done[-1], base=arm == "base")
            trainer = trainer_for(arm, seed, resume if done else None)
            try:
                _finish_arm(
                    trainer,
                    metrics,
                    root,
                    slug=slug,
                    arm=arm,
                    seed=seed,
                    steps=steps,
                    group_size=group_size,
                    eval_every=eval_every,
                    train=train,
                    dev=dev,
                    heldout=heldout,
                    random_rate=random_rate,
                    meter=meter,
                    difficulty=difficulty,
                    done=done,
                    throughput=throughput,
                )
            finally:
                release(trainer)


def _finish_arm(
    trainer: object,
    metrics: Path,
    root: Path,
    *,
    slug: str,
    arm: str,
    seed: int,
    steps: int,
    group_size: int,
    eval_every: int,
    train: list[SplitTask],
    dev: list[SplitTask],
    heldout: list[HeldoutItem],
    random_rate: float,
    meter: Meter | None,
    difficulty: int,
    done: list[int],
    throughput: Path | None = None,
) -> None:
    start = (done[-1] + 1) if done else 1
    if not done:
        _eval_checkpoint(
            trainer, metrics, slug, arm, seed, 0, group_size, dev, heldout, meter, throughput, batch_base=seed
        )
    elif done[-1] % eval_every == 0:
        _eval_checkpoint(
            trainer, metrics, slug, arm, seed, done[-1], group_size, dev, heldout, meter, throughput, batch_base=seed
        )
    for step in range(start, steps + 1):
        task = train[(step - 1) % len(train)]
        prompts = [task.prompt] * group_size
        generate = getattr(trainer, "generate_batch")
        encode = getattr(trainer, "encode_batch")
        started = time.perf_counter()
        raw = generate(prompts, batch_seed(seed, step, 1), sample=True)
        if not isinstance(raw, list) or len(raw) != group_size:
            raise RuntimeError("generate_batch changed the group size")
        texts = normalize_completions([str(text) for text in raw])
        random_rewards = None
        if arm == "random":
            random_rewards = assign_random_rewards(slug, seed, step, group_size, random_rate)
        rewards = assign_rewards(arm, task.expression, texts, random_rewards=random_rewards)
        rollouts = encode(prompts, texts)
        if not isinstance(rollouts, list) or len(rollouts) != group_size or len(rewards) != group_size:
            raise RuntimeError("group was truncated")
        advantages = group_advantages(rewards, group_size)
        _shape_check(rollouts, advantages)
        if arm == "base":
            loss = 0.0
            marker = _checkpoint(root, step)
            marker.mkdir(parents=True, exist_ok=True)
            (marker / "marker.json").write_text(json.dumps({"step": step}) + "\n", encoding="utf-8")
        else:
            update = getattr(trainer, "update")
            loss = float(update(rollouts, advantages, epochs=1))
            if not math.isfinite(loss):
                raise RuntimeError("non-finite loss")
            save = getattr(trainer, "save")
            save(_checkpoint(root, step))
        if meter is not None:
            meter.add_train(time.perf_counter() - started, group_size)
            if throughput is not None:
                write_rates(throughput, meter, announce=False)
        formats = [1.0 if parse_final_integer(text) is not None else 0.0 for text in texts]
        lengths = [float(sum(rollout.mask)) for rollout in rollouts]
        entropies = [_masked_mean(rollout.entropy, rollout.mask) for rollout in rollouts]
        append_jsonl(
            metrics,
            {
                "apply_update": arm != "base",
                "arm": arm,
                "difficulty": difficulty,
                "entropy_mean": _mean(entropies),
                "format_compliance_rate": _mean(formats),
                "group_size": group_size,
                "kind": "train",
                "kl_mean": 0.0 if arm == "base" else _kl_mean(rollouts),
                "loss": loss,
                "model": slug,
                "response_length_mean": _mean(lengths),
                "reward_mean": _mean(rewards),
                "seed": seed,
                "step": step,
                "task_seed": task.seed,
            },
        )
        if step % eval_every == 0:
            _eval_checkpoint(
                trainer, metrics, slug, arm, seed, step, group_size, dev, heldout, meter, throughput, batch_base=seed
            )


def _eval_checkpoint(
    trainer: object,
    metrics: Path,
    slug: str,
    arm: str,
    seed: int,
    step: int,
    batch_size: int,
    dev: list[SplitTask],
    heldout: list[HeldoutItem],
    meter: Meter | None,
    throughput: Path | None,
    *,
    batch_base: int,
) -> None:
    rows = read_jsonl(metrics)
    specs = (
        (
            "dev",
            [task.prompt for task in dev],
            [task.value for task in dev],
            [task.expression for task in dev],
            4,
        ),
        (
            "heldout",
            [item.prompt for item in heldout],
            [item.gold for item in heldout],
            [None for _ in heldout],
            5,
        ),
    )
    for split, prompts, golds, expressions, stream in specs:
        if _has_eval(rows, slug, arm, seed, step, split):
            continue
        texts, elapsed = _batched(
            trainer,
            prompts,
            batch_seed(batch_base, step, stream),
            batch_size,
            sample=True,
        )
        if meter is not None:
            meter.add_eval(elapsed, len(texts))
            if throughput is not None:
                write_rates(throughput, meter, announce=True)
        graded = _grade_eval(texts, golds, expressions)
        append_jsonl(
            metrics,
            {
                "arm": arm,
                "kind": "eval",
                "model": slug,
                "seed": seed,
                "split": split,
                "step": step,
                **graded,
            },
        )


def load_gsm8k(n: int, seed: int, cache: Path) -> list[HeldoutItem]:
    if cache.exists():
        rows = read_jsonl(cache)
        if len(rows) < n:
            raise SystemExit("held-out cache is shorter than the locked plan")
        return [
            HeldoutItem(prompt=str(row["prompt"]), gold=int(row["gold"]), source="gsm8k")
            for row in rows[:n]
        ]
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("GSM8K download requires the datasets package") from exc
    try:
        dataset = load_dataset("gsm8k", "main", split="test")
    except Exception as exc:
        raise RuntimeError("GSM8K test split could not be downloaded") from exc
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    chosen: list[dict[str, object]] = []
    for index in order:
        row = dataset[int(index)]
        gold = parse_gsm_gold(str(row["answer"]))
        if gold is None:
            continue
        question = str(row["question"]).strip()
        chosen.append(
            {
                "gold": gold,
                "index": int(index),
                "prompt": GSM_INSTRUCTION.format(question=question),
                "source": "gsm8k",
            }
        )
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise RuntimeError("GSM8K test did not yield enough integer answers")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in chosen), encoding="utf-8")
    return [HeldoutItem(prompt=str(row["prompt"]), gold=int(row["gold"]), source="gsm8k") for row in chosen]


def fixture_heldout(n: int) -> list[HeldoutItem]:
    items = []
    for index in range(n):
        expression = f"{20 + index} + {3 + index}"
        value = evaluate_expression(expression)
        items.append(
            HeldoutItem(
                prompt=GSM_INSTRUCTION.format(question=expression),
                gold=value,
                source="fixture",
            )
        )
    return items


def _scores(tasks: list[SplitTask], texts: list[str]) -> dict[str, float]:
    if len(tasks) != DEV_COUNT or len(texts) != DEV_COUNT:
        raise RuntimeError("the promotion gate report must cover all 32 dev tasks")
    return {
        str(task.seed): 1.0 if correctness_reward(task.expression, text) == 1.0 else 0.0
        for task, text in zip(tasks, texts, strict=True)
    }


def _measure_dev(trainer: object, tasks: list[SplitTask], seed: int, step: int, batch_size: int) -> tuple[DevReport, float]:
    prompts = [task.prompt for task in tasks]
    texts, _elapsed = _batched(trainer, prompts, batch_seed(seed, step, 6), batch_size, sample=True)
    verdict_prompts = [verifier_prompt(task.expression, text) for task, text in zip(tasks, texts, strict=True)]
    verdicts, _elapsed = _batched(trainer, verdict_prompts, batch_seed(seed, step, 7), batch_size, sample=True)
    scores = _scores(tasks, texts)
    judgements = []
    compliant = 0
    for task, text, verdict in zip(tasks, texts, verdicts, strict=True):
        correct = scores[str(task.seed)] == 1.0
        compliant += int(parse_final_integer(text) is not None)
        judgements.append((correct, verdict_accepts(text, verdict)))
    report = DevReport(
        policy_id="",
        scores=scores,
        format_compliance=compliant / len(tasks),
        false_accept_rate=false_accept_rate(judgements),
        format_reward=0.0,
        train_reward=0.0,
    )
    return report, report.primary()


def run_part_b(
    output: Path,
    *,
    slug: str,
    rounds: int,
    steps: int,
    group_size: int,
    problems_per_round: int,
    seed: int,
    trainer_for: Callable[[Path | None], object],
) -> None:
    dev = dev_tasks()
    if len(dev) != DEV_COUNT:
        raise RuntimeError("dev split does not have 32 tasks")
    metrics = output / "metrics.jsonl"
    for source in ("ground_truth", "self_verifier"):
        dest = output / "part_b" / source
        dest.mkdir(parents=True, exist_ok=True)
        log_path = dest / "rounds.jsonl"
        existing = read_jsonl(log_path)
        baselines = [row for row in existing if row.get("kind") == "baseline"]
        finished = [row for row in existing if row.get("kind") == "round"]
        if [int(row["round"]) for row in finished] != list(range(1, len(finished) + 1)):
            raise SystemExit(f"{source} rounds are missing or duplicated")
        if finished and not baselines:
            raise SystemExit(f"{source} is missing the baseline")
        difficulty = 2 if not finished else int(finished[-1]["next_difficulty"])
        if baselines:
            incumbent_report = _stored_report(baselines[-1], f"{source}-round-0")
            incumbent = Path(str(baselines[-1]["checkpoint"]))
            for row in finished:
                incumbent_report = _stored_report(row, f"{source}-promoted-{int(row['round'])}", prefix="promoted_")
                incumbent = Path(str(row["promoted_checkpoint"]))
        else:
            trainer = trainer_for(None)
            try:
                incumbent = dest / "incumbent-0000"
                save = getattr(trainer, "save")
                save(incumbent)
                measured_base, _primary = _measure_dev(trainer, dev, seed, 0, group_size)
                incumbent_report = DevReport(
                    policy_id=f"{source}-round-0",
                    scores=measured_base.scores,
                    format_compliance=measured_base.format_compliance,
                    false_accept_rate=measured_base.false_accept_rate,
                    format_reward=0.0,
                    train_reward=0.0,
                )
            finally:
                release(trainer)
            append_jsonl(log_path, {"checkpoint": str(incumbent), "kind": "baseline", **_report_payload(incumbent_report)})
            append_jsonl(
                metrics,
                {
                    "accepted": True,
                    "false_accept_rate": incumbent_report.false_accept_rate,
                    "kind": "rsi_baseline",
                    "model": slug,
                    "reward_source": source,
                    "round": 0,
                    "self_reward": 0.0,
                    "true_accuracy": incumbent_report.primary(),
                },
            )
        for index in range(len(finished) + 1, rounds + 1):
            problems = propose(index, difficulty, problems_per_round)
            for problem in problems:
                if evaluate_expression(problem.expression) != problem.value or problem.seed < 5_000_000:
                    raise RuntimeError("proposer returned a task that does not recompute")
            round_dir = dest / f"round-{index:04d}"
            partial = [int(row["step"]) for row in read_jsonl(round_dir / "metrics.jsonl") if row.get("kind") == "train"]
            if partial != list(range(1, len(partial) + 1)):
                raise SystemExit("part B train steps are missing or duplicated")
            start_ckpt = _checkpoint(round_dir, partial[-1]) if partial else incumbent
            probe_path = round_dir / "probe.json"
            solve_rate: float | None = None
            if probe_path.exists():
                solve_rate = float(json.loads(probe_path.read_text(encoding="utf-8"))["solve_rate"])
            elif partial:
                prober = trainer_for(incumbent)
                try:
                    solve_rate = _solve_rate(prober, problems, seed, index, group_size)
                finally:
                    release(prober)
                probe_path.parent.mkdir(parents=True, exist_ok=True)
                probe_path.write_text(json.dumps({"solve_rate": solve_rate}) + "\n", encoding="utf-8")
            trainer = trainer_for(start_ckpt)
            try:
                if solve_rate is None:
                    solve_rate = _solve_rate(trainer, problems, seed, index, group_size)
                    probe_path.parent.mkdir(parents=True, exist_ok=True)
                    probe_path.write_text(json.dumps({"solve_rate": solve_rate}) + "\n", encoding="utf-8")
                train_reward, format_reward, candidate = _train_round(
                    trainer,
                    round_dir,
                    problems,
                    source=source,
                    steps=steps,
                    group_size=group_size,
                    seed=seed,
                    round_index=index,
                )
                measured, _primary = _measure_dev(trainer, dev, seed, index, group_size)
                measured = DevReport(
                    policy_id=f"{source}-round-{index}",
                    scores=measured.scores,
                    format_compliance=measured.format_compliance,
                    false_accept_rate=measured.false_accept_rate,
                    format_reward=format_reward,
                    train_reward=train_reward,
                )
            finally:
                release(trainer)
            signals = monitor_signals(incumbent_report, measured)
            decision = dev_gate().decide(to_evaluation_report(incumbent_report), to_evaluation_report(measured))
            if decision.accepted:
                promoted = candidate
                incumbent = candidate
                incumbent_report = measured
            else:
                promoted = incumbent
            next_difficulty = adjust_difficulty(difficulty, solve_rate)
            record: dict[str, object] = {
                "accepted": decision.accepted,
                "candidate_checkpoint": str(candidate),
                "difficulty": difficulty,
                "false_accept_rate": measured.false_accept_rate,
                "format_compliance": measured.format_compliance,
                "format_reward": measured.format_reward,
                "kind": "round",
                "next_difficulty": next_difficulty,
                "promoted_checkpoint": str(promoted),
                "reasons": list(decision.reasons),
                "reward_source": source,
                "round": index,
                "scores": measured.scores,
                "self_reward": train_reward,
                "signals": list(signals),
                "solve_rate": solve_rate,
                "train_reward": train_reward,
                "true_accuracy": measured.primary(),
            }
            record.update(_report_payload(incumbent_report, prefix="promoted_"))
            append_jsonl(log_path, record)
            append_jsonl(
                metrics,
                {
                    "accepted": decision.accepted,
                    "false_accept_rate": measured.false_accept_rate,
                    "kind": "rsi_round",
                    "model": slug,
                    "promoted_policy": str(promoted),
                    "reasons": list(decision.reasons),
                    "reward_source": source,
                    "round": index,
                    "self_reward": train_reward,
                    "signals": list(signals),
                    "true_accuracy": measured.primary(),
                },
            )
            difficulty = next_difficulty


def _solve_rate(trainer: object, problems: list, seed: int, index: int, group_size: int) -> float:
    texts, _elapsed = _batched(
        trainer,
        [problem.prompt for problem in problems],
        batch_seed(seed, index, 8),
        group_size,
        sample=False,
    )
    hits = [
        correctness_reward(problem.expression, text) == 1.0
        for problem, text in zip(problems, texts, strict=True)
    ]
    return sum(1.0 for hit in hits if hit) / len(hits)


def _report_payload(report: DevReport, prefix: str = "") -> dict[str, object]:
    return {
        f"{prefix}false_accept_rate": report.false_accept_rate,
        f"{prefix}format_compliance": report.format_compliance,
        f"{prefix}format_reward": report.format_reward,
        f"{prefix}scores": report.scores,
        f"{prefix}train_reward": report.train_reward,
        f"{prefix}true_accuracy": report.primary(),
    }


def _stored_report(row: dict[str, object], policy_id: str, prefix: str = "") -> DevReport:
    raw = row.get(f"{prefix}scores")
    if not isinstance(raw, dict):
        raise SystemExit("round record has no dev scores")
    return DevReport(
        policy_id=policy_id,
        scores={str(key): float(value) for key, value in raw.items()},
        format_compliance=float(row[f"{prefix}format_compliance"]),
        false_accept_rate=float(row[f"{prefix}false_accept_rate"]),
        format_reward=float(row[f"{prefix}format_reward"]),
        train_reward=float(row[f"{prefix}train_reward"]),
    )


def _train_round(
    trainer: object,
    round_dir: Path,
    problems: list,
    *,
    source: str,
    steps: int,
    group_size: int,
    seed: int,
    round_index: int,
) -> tuple[float, float, Path]:
    metrics = round_dir / "metrics.jsonl"
    rows = read_jsonl(metrics)
    done = [int(row["step"]) for row in rows if row.get("kind") == "train"]
    if done != list(range(1, len(done) + 1)):
        raise SystemExit("part B train steps are missing or duplicated")
    if done and done[-1] > steps:
        raise SystemExit("part B round has more steps than the plan")
    rewards_seen: list[float] = [float(row["reward_mean"]) for row in rows if row.get("kind") == "train"]
    formats_seen: list[float] = [float(row["format_compliance_rate"]) for row in rows if row.get("kind") == "train"]
    for step in range(done[-1] + 1 if done else 1, steps + 1):
        problem = problems[(step - 1) % len(problems)]
        prompts = [problem.prompt] * group_size
        generate = getattr(trainer, "generate_batch")
        raw = generate(prompts, batch_seed(seed + round_index * 50, step, 9), sample=True)
        if not isinstance(raw, list) or len(raw) != group_size:
            raise RuntimeError("part B generate changed the group size")
        texts = normalize_completions([str(text) for text in raw])
        if source == "ground_truth":
            rewards = [correctness_reward(problem.expression, text) for text in texts]
        elif source == "self_verifier":
            verdict_prompts = [verifier_prompt(problem.expression, text) for text in texts]
            verdicts, _elapsed = _batched(
                trainer,
                verdict_prompts,
                batch_seed(seed + round_index * 50, step, 10),
                group_size,
                sample=True,
            )
            rewards = [1.0 if verdict_accepts(text, verdict) else 0.0 for text, verdict in zip(texts, verdicts, strict=True)]
        else:
            raise ValueError(source)
        encode = getattr(trainer, "encode_batch")
        rollouts = encode(prompts, texts)
        if len(rollouts) != group_size:
            raise RuntimeError("part B group was truncated")
        advantages = group_advantages(rewards, group_size)
        _shape_check(rollouts, advantages)
        update = getattr(trainer, "update")
        loss = float(update(rollouts, advantages, epochs=1))
        if not math.isfinite(loss):
            raise RuntimeError("non-finite loss")
        save = getattr(trainer, "save")
        save(_checkpoint(round_dir, step))
        formats = [format_reward(text) for text in texts]
        append_jsonl(
            metrics,
            {
                "format_compliance_rate": _mean(formats),
                "group_size": group_size,
                "kind": "train",
                "loss": loss,
                "reward_mean": _mean(rewards),
                "step": step,
            },
        )
        rewards_seen.append(_mean(rewards))
        formats_seen.append(_mean(formats))
    if len(rewards_seen) != steps:
        raise RuntimeError("part B round did not finish every step")
    return _mean(rewards_seen), _mean(formats_seen), _checkpoint(round_dir, steps)


def scripted_self_verifier_hack() -> dict[str, object]:
    """Deterministic rejection. This is not a model measurement."""
    before_scores = {str(index): (1.0 if index < 16 else 0.0) for index in range(DEV_COUNT)}
    after_scores = {str(index): (1.0 if index < 8 else 0.0) for index in range(DEV_COUNT)}
    before = DevReport(
        policy_id="fixture-incumbent",
        scores=before_scores,
        format_compliance=1.0,
        false_accept_rate=0.0,
        format_reward=0.2,
        train_reward=0.2,
    )
    after = DevReport(
        policy_id="fixture-candidate",
        scores=after_scores,
        format_compliance=1.0,
        false_accept_rate=0.5,
        format_reward=0.2,
        train_reward=0.9,
    )
    signals = monitor_signals(before, after)
    decision = dev_gate().decide(to_evaluation_report(before), to_evaluation_report(after))
    return {
        "accepted": decision.accepted,
        "false_accept_rate": after.false_accept_rate,
        "reasons": list(decision.reasons),
        "signals": list(signals),
        "source": "cpu_fixture",
        "true_accuracy_after": after.primary(),
        "true_accuracy_before": before.primary(),
    }


def render_heldout_svg(rows: list[dict[str, object]]) -> str:
    evals = [row for row in rows if row.get("kind") == "eval" and row.get("split") == "heldout"]
    models = []
    for row in evals:
        slug = str(row.get("model"))
        if slug not in models:
            models.append(slug)
    height = max(220, 280 * max(1, len(models)))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" font-family="sans-serif">',
        f'<rect width="960" height="{height}" fill="white"/>',
        '<text x="16" y="24" font-size="16">held-out strict accuracy</text>',
    ]
    for slot, slug in enumerate(models or ["none"]):
        top = 40 + slot * 280
        model_rows = [row for row in evals if str(row.get("model")) == slug]
        parts.append(f'<text x="16" y="{top + 18}" font-size="14">{slug}</text>')
        parts.append(f'<rect x="48" y="{top + 28}" width="880" height="220" fill="none" stroke="#ccc"/>')
        steps = sorted({int(row["step"]) for row in model_rows})
        for arm in ARMS:
            band = []
            line = []
            for index, step in enumerate(steps):
                values = [
                    float(row["accuracy"])
                    for row in model_rows
                    if row.get("arm") == arm and int(row["step"]) == step
                ]
                if not values:
                    continue
                x = 48 if len(steps) == 1 else 48 + 880 * index / (len(steps) - 1)
                band.append((x, min(values), max(values)))
                line.append((x, sum(values) / len(values)))
            color = _ARM_COLOR.get(arm, "#333")
            if len(band) >= 2:
                forward = " ".join(f"{x:.1f},{28 + top + 220 * (1 - low):.1f}" for x, low, _high in band)
                backward = " ".join(f"{x:.1f},{28 + top + 220 * (1 - high):.1f}" for x, _low, high in reversed(band))
                parts.append(f'<polygon points="{forward} {backward}" fill="{color}" opacity="0.15"/>')
            if line:
                points = " ".join(f"{x:.1f},{28 + top + 220 * (1 - y):.1f}" for x, y in line)
                parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        for index, arm in enumerate(ARMS):
            parts.append(
                f'<text x="{48 + index * 150}" y="{top + 268}" font-size="12" fill="{_ARM_COLOR[arm]}">{arm}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_rsi_svg(rows: list[dict[str, object]]) -> str:
    series = [row for row in rows if row.get("kind") in {"rsi_baseline", "rsi_round"}]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420" font-family="sans-serif">',
        '<rect width="960" height="420" fill="white"/>',
        '<text x="16" y="24" font-size="16">self reward and true accuracy</text>',
    ]
    sources = []
    for row in series:
        source = str(row.get("reward_source"))
        if source not in sources:
            sources.append(source)
    for slot, source in enumerate(sources or ["none"]):
        left = 40 + slot * 460
        parts.append(f'<text x="{left}" y="48" font-size="14">{source}</text>')
        parts.append(f'<rect x="{left}" y="60" width="420" height="280" fill="none" stroke="#ccc"/>')
        owned = [row for row in series if str(row.get("reward_source")) == source]
        owned.sort(key=lambda row: int(row["round"]))
        if not owned:
            continue
        for name, color in (("true_accuracy", "#1f4e79"), ("self_reward", "#9b2226")):
            points = []
            for index, row in enumerate(owned):
                x = left if len(owned) == 1 else left + 420 * index / (len(owned) - 1)
                y = 60 + 280 * (1 - float(row[name]))
                points.append(f"{x:.1f},{y:.1f}")
                if name == "true_accuracy" and row.get("kind") == "rsi_round":
                    label = "accept" if row.get("accepted") else "reject"
                    parts.append(f'<text x="{x + 4:.1f}" y="{y - 6:.1f}" font-size="11">{label}</text>')
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>')
        parts.append(f'<text x="{left}" y="360" font-size="12" fill="#1f4e79">true accuracy</text>')
        parts.append(f'<text x="{left + 160}" y="360" font-size="12" fill="#9b2226">self reward</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_summary(output: Path) -> dict[str, object]:
    rows = read_jsonl(output / "metrics.jsonl")
    heldout = [row for row in rows if row.get("kind") == "eval" and row.get("split") == "heldout"]
    rsi = [row for row in rows if row.get("kind") in {"rsi_baseline", "rsi_round"}]
    summary = {
        "heldout": heldout,
        "note": "A completed GPU JSONL is the measurement. This summary does not by itself establish a held-out gain.",
        "rsi": rsi,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    (figure_dir / "heldout.svg").write_text(render_heldout_svg(rows), encoding="utf-8")
    (figure_dir / "rsi.svg").write_text(render_rsi_svg(rows), encoding="utf-8")
    print("heldout")
    for row in heldout:
        print(
            f"  {row['model']} {row['arm']} seed {row['seed']} step {row['step']} "
            f"acc {row['accuracy']} wilson [{row['wilson_low']}, {row['wilson_high']}] n {row['n']}"
        )
    print("rsi")
    for row in rsi:
        print(
            f"  {row['reward_source']} round {row['round']} self {row['self_reward']} "
            f"true {row['true_accuracy']} accept {row.get('accepted')}"
        )
    return summary


def release(trainer: object) -> None:
    for name in ("model", "ref_model", "optimizer"):
        if hasattr(trainer, name):
            setattr(trainer, name, None)
    torch = sys.modules.get("torch")
    if torch is not None and bool(torch.cuda.is_available()):
        torch.cuda.empty_cache()


class TinyTrainer:
    """Deterministic stand-in. No weights and no hub download."""

    def __init__(self) -> None:
        self.theta = 0.0

    def prepare(self, resume: Path | None) -> None:
        self.theta = 0.0
        if resume is not None and (resume / "theta.json").exists():
            self.theta = float(json.loads((resume / "theta.json").read_text(encoding="utf-8"))["theta"])

    def activate(self, _adapter: Path | None) -> None:
        self.prepare(None)

    def generate_batch(self, prompts: list[str], seed: int, *, sample: bool = True) -> list[str]:
        rng = random.Random(seed + (0 if sample else 17))
        texts = []
        for prompt in prompts:
            value = _value_in_prompt(prompt)
            rate = 1.0 if not sample else min(0.95, max(0.05, 0.25 + self.theta))
            if value is not None and rng.random() < rate:
                texts.append(f"#### {value}")
            elif rng.random() < 0.5:
                texts.append("#### 0")
            else:
                texts.append("no final answer")
        return texts

    def encode_batch(self, prompts: list[str], completions: list[str]) -> list[Rollout]:
        if len(prompts) != len(completions):
            raise RuntimeError("tiny encode changed the group size")
        return [
            Rollout(
                text=completion,
                old_logp=(self.theta,),
                mask=(1,),
                entropy=(0.4,),
                token_ids=(1, 2),
                prompt_length=1,
            )
            for completion in completions
        ]

    def update(self, rollouts: list[Rollout], advantages: list[float], *, epochs: int) -> float:
        if len(rollouts) != len(advantages):
            raise RuntimeError("tiny update changed the group size")
        self.theta += 0.05 * _mean([float(value) for value in advantages])
        loss = sequence_loss(
            [
                TokenRow(
                    logp=(self.theta,),
                    logp_old=rollout.old_logp,
                    mask=rollout.mask,
                    advantage=advantage,
                    entropy=rollout.entropy,
                )
                for rollout, advantage in zip(rollouts, advantages, strict=True)
            ]
        )["loss"]
        return float(loss)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "theta.json").write_text(json.dumps({"theta": self.theta}) + "\n", encoding="utf-8")


def _value_in_prompt(prompt: str) -> int | None:
    match = re.search(r"Expression: (.+)", prompt)
    if match:
        expression = match.group(1).splitlines()[0].strip()
    else:
        expression = prompt.splitlines()[0].split(":", 1)[-1].strip()
    try:
        return evaluate_expression(expression)
    except ValueError:
        return None


def _lock_plan(output: Path, mode: str, throughput: Path | None) -> dict[str, object]:
    path = output / "run.json"
    output.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if int(saved.get("plan_version", -1)) != PLAN_VERSION or saved.get("mode") != mode:
            raise SystemExit("output directory was locked with a different flagship plan")
        print(
            f"locked steps {saved['steps']} heldout {saved['heldout_n']} "
            f"eta_seconds {saved['eta_seconds']} rate_source {saved['rate_source']}"
        )
        return saved
    sec_train, sec_eval, source = read_rates(throughput)
    plan = fit_plan(mode, sec_train, sec_eval, source)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"planned steps {plan['steps']} heldout {plan['heldout_n']} "
        f"eta_seconds {plan['eta_seconds']:.1f} rate_source {source}"
    )
    for note in plan["notes"]:
        print(note)
    return plan


def _probe(trainer: object, slug: str, metrics: Path) -> tuple[int, float, bool]:
    rows = _rows_for(read_jsonl(metrics), kind="probe", model=slug)
    if rows:
        chosen = [row for row in rows if row.get("chosen") is True]
        if len(chosen) != 1:
            raise SystemExit("probe rows do not name one difficulty")
        return int(chosen[0]["difficulty"]), float(chosen[0]["accuracy"]), bool(chosen[0]["calibration_inside_band"])
    table = []
    print("difficulty probe")
    for difficulty in (2, 3, 4):
        tasks = build_split("dev", difficulty=difficulty)[:PROBE_N]
        prompts = [task.prompt for task in tasks]
        texts, _elapsed = _batched(trainer, prompts, batch_seed(20261099, difficulty, 3), PROBE_N, sample=False)
        hits = sum(correctness_reward(task.expression, text) == 1.0 for task, text in zip(tasks, texts, strict=True))
        accuracy = hits / len(tasks)
        table.append((difficulty, accuracy))
        print(f"  difficulty {difficulty} accuracy {accuracy:.4f} n {len(tasks)}")
    chosen, inside = pick_difficulty(table)
    print(f"chosen {chosen} calibration_inside_band: {str(inside).lower()}")
    for difficulty, accuracy in table:
        append_jsonl(
            metrics,
            {
                "accuracy": accuracy,
                "calibration_inside_band": inside,
                "chosen": difficulty == chosen,
                "difficulty": difficulty,
                "kind": "probe",
                "model": slug,
                "n": PROBE_N,
            },
        )
    rate = next(accuracy for difficulty, accuracy in table if difficulty == chosen)
    return chosen, rate, inside


def _resolution_path(output: Path, requested_slug: str) -> Path:
    return output / "resolved" / f"{requested_slug}.json"


def _read_resolution(output: Path, requested_slug: str) -> dict[str, str] | None:
    path = _resolution_path(output, requested_slug)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"slug": str(payload["slug"]), "repo": str(payload["repo"]), "revision": str(payload["revision"])}


def _write_resolution(output: Path, requested_slug: str, used: dict[str, str], reason: str) -> None:
    path = _resolution_path(output, requested_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"reason": reason, "requested_slug": requested_slug, **used}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_gpu(mode: str, output: Path, throughput: Path | None) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. This runner does not record a fake GPU result.")
    name = torch.cuda.get_device_name(0)
    print(f"gpu {name}")
    if "A100" not in name:
        print("warning: the 2.5h grid targets one A100; continuing on this GPU")
    plan = _lock_plan(output, mode, throughput)
    hack = scripted_self_verifier_hack()
    (output / "scripted_hack.json").write_text(json.dumps(hack, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    meter = Meter()
    heldout = load_gsm8k(int(plan["heldout_n"]), int(plan["heldout_seed"]), output / "heldout_gsm8k.jsonl")
    for spec in plan["models"]:
        if not isinstance(spec, dict):
            raise SystemExit("locked plan has a malformed model list")
        used = _open_spec(output, spec, int(plan["max_new_tokens"]))
        metrics = output / "metrics.jsonl"
        probe_trainer = _make_torch(used, seed=20261099, output=output, log_kl=False, apply_update=False)
        probe_trainer.activate(None)
        try:
            difficulty, rate, _inside = _probe(probe_trainer, used["slug"], metrics)
        finally:
            release(probe_trainer)
        print(f"random_rate {used['slug']} {rate:.4f}")
        train = build_split("train", difficulty=difficulty)
        dev = build_split("dev", difficulty=difficulty)[: int(plan["dev_n"])]
        run_part_a(
            output,
            slug=used["slug"],
            arms=tuple(str(arm) for arm in plan["arms"]),  # type: ignore[union-attr]
            seeds=tuple(int(seed) for seed in plan["seeds"]),  # type: ignore[union-attr]
            steps=int(plan["steps"]),
            group_size=int(plan["group_size"]),
            eval_every=int(plan["eval_every"]),
            train=train,
            dev=dev,
            heldout=heldout,
            random_rate=rate,
            trainer_for=_torch_factory(used, output, int(plan["max_new_tokens"]), int(plan["group_size"])),
            meter=meter,
            difficulty=difficulty,
            throughput=throughput,
        )
        if throughput is not None:
            write_rates(throughput, meter)
    qwen = _qwen()
    run_part_b(
        output,
        slug=qwen["slug"],
        rounds=int(plan["part_b_rounds"]),
        steps=int(plan["part_b_steps"]),
        group_size=int(plan["part_b_group"]),
        problems_per_round=int(plan["part_b_problems"]),
        seed=20261041,
        trainer_for=_part_b_factory(qwen, output, int(plan["max_new_tokens"]), int(plan["part_b_group"])),
    )
    write_summary(output)


def _make_torch(spec: dict[str, str], *, seed: int, output: Path, log_kl: bool, apply_update: bool, group_size: int = 8, max_new_tokens: int = 128) -> object:
    from minigrpo.torch_train import TorchTrainer

    config = RunConfig(
        backend="torch",
        model=spec["repo"],
        revision=spec["revision"],
        steps=1,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
        learning_rate=1e-5,
        kl_coef=0.0,
        clip_eps=0.2,
        dr_grpo=False,
        epochs=1,
        lora_rank=8,
        lora_scale=20.0,
        lora_dropout=0.0,
        num_layers=1,
        seed=seed,
        temperature=0.7,
        output=str(output),
        max_seq_length=1024,
        torch_dtype="bfloat16",
        apply_update=apply_update,
        log_kl=log_kl,
    )
    return TorchTrainer(config)


def _open_spec(output: Path, spec: dict[str, object], max_new_tokens: int) -> dict[str, str]:
    requested = str(spec["slug"])
    existing = _read_resolution(output, requested)
    if existing is not None:
        print(f"resolved {requested} -> {existing['slug']}")
        return existing
    primary = {"slug": requested, "repo": str(spec["repo"]), "revision": str(spec["revision"])}
    try:
        trainer = _make_torch(primary, seed=1, output=output, log_kl=False, apply_update=False, max_new_tokens=max_new_tokens)
        trainer.activate(None)
        release(trainer)
    except Exception as exc:
        fallback_repo = str(spec.get("fallback_repo", ""))
        if not fallback_repo:
            raise
        print(f"fallback from {primary['repo']}: {type(exc).__name__}: {exc}")
        used = {
            "slug": str(spec["fallback_slug"]),
            "repo": fallback_repo,
            "revision": str(spec["fallback_revision"]),
        }
        trainer = _make_torch(used, seed=1, output=output, log_kl=False, apply_update=False, max_new_tokens=max_new_tokens)
        trainer.activate(None)
        release(trainer)
        _write_resolution(output, requested, used, f"{type(exc).__name__}: {exc}")
        return used
    _write_resolution(output, requested, primary, "")
    return primary


def _torch_factory(spec: dict[str, str], output: Path, max_new_tokens: int, group_size: int) -> Callable[[str, int, Path | None], object]:
    def factory(arm: str, seed: int, resume: Path | None) -> object:
        trainer = _make_torch(
            spec,
            seed=seed,
            output=output,
            log_kl=arm != "base",
            apply_update=arm != "base",
            group_size=group_size,
            max_new_tokens=max_new_tokens,
        )
        if arm == "base":
            trainer.activate(None)
        else:
            trainer.prepare(resume)
        return trainer

    return factory


def _part_b_factory(spec: dict[str, str], output: Path, max_new_tokens: int, group_size: int) -> Callable[[Path | None], object]:
    def factory(resume: Path | None) -> object:
        trainer = _make_torch(
            spec,
            seed=20261041,
            output=output,
            log_kl=True,
            apply_update=True,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
        )
        trainer.prepare(resume)
        return trainer

    return factory


def run_cpu_smoke(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    hack = scripted_self_verifier_hack()
    if hack["accepted"] or "verifier_disagreement" not in hack["reasons"] or "heldout_divergence" not in hack["signals"]:
        raise RuntimeError("scripted self-verifier case was not rejected")
    (output / "scripted_hack.json").write_text(json.dumps(hack, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan_path = output / "run.json"
    if not plan_path.exists():
        plan = {
            "arms": ["correctness", "random", "incorrect", "format", "base"],
            "dev_n": 2,
            "eval_every": 2,
            "group_size": 2,
            "heldout_n": 2,
            "heldout_source": "fixture",
            "mode": "cpu-smoke",
            "plan_version": PLAN_VERSION,
            "seeds": [20261031],
            "steps": 2,
        }
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("mode") != "cpu-smoke":
        raise SystemExit("output directory is not a cpu smoke")
    metrics = output / "metrics.jsonl"
    slug = "tiny"
    if not _rows_for(read_jsonl(metrics), kind="probe", model=slug):
        chosen, inside = pick_difficulty(((2, 0.8), (3, 0.4), (4, 0.1)))
        for difficulty, accuracy in ((2, 0.8), (3, 0.4), (4, 0.1)):
            append_jsonl(
                metrics,
                {
                    "accuracy": accuracy,
                    "calibration_inside_band": inside,
                    "chosen": difficulty == chosen,
                    "difficulty": difficulty,
                    "kind": "probe",
                    "model": slug,
                    "n": 1,
                    "scripted": True,
                },
            )
    _difficulty, rate, _inside = _probe(TinyTrainer(), slug, metrics)
    train = build_split("train", difficulty=3)[:2]
    dev = build_split("dev", difficulty=3)[:2]
    heldout = fixture_heldout(2)

    def trainer_for(arm: str, seed: int, resume: Path | None) -> TinyTrainer:
        trainer = TinyTrainer()
        if arm == "base":
            trainer.prepare(None)
        else:
            trainer.prepare(resume)
        return trainer

    run_part_a(
        output,
        slug=slug,
        arms=tuple(plan["arms"]),
        seeds=tuple(int(seed) for seed in plan["seeds"]),
        steps=int(plan["steps"]),
        group_size=int(plan["group_size"]),
        eval_every=int(plan["eval_every"]),
        train=train,
        dev=dev,
        heldout=heldout,
        random_rate=rate,
        trainer_for=trainer_for,
        meter=None,
        difficulty=3,
    )

    def part_b_trainer(resume: Path | None) -> TinyTrainer:
        trainer = TinyTrainer()
        trainer.prepare(resume)
        return trainer

    run_part_b(
        output,
        slug=slug,
        rounds=1,
        steps=1,
        group_size=2,
        problems_per_round=2,
        seed=20261041,
        trainer_for=part_b_trainer,
    )
    return write_summary(output)
