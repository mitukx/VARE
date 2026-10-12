"""Fresh held-out exact-integer tasks and a base-model pass@k gate.

The task pack is generated from a frozen seed. It is an evaluation set only.
A non-degenerate result does not authorize a policy update.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

PROTOCOL_ID = "heldout-exact-integer-base-gate-v1"
GENERATION_SEED = 20261013
N_TASKS = 64
FAMILIES = ("add_sub", "mul_add", "div_exact", "paren")
SAMPLES_PER_TASK = 4
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 48
SAMPLING_SEED = 20261013
WILSON_Z = 1.959963984540054

FINAL_LINE = re.compile(r"^####\s+(-?\d+)\s*$")
_EXPR_TOKEN = re.compile(r"\d+|[()+\-*/]")
_ANSWER_TOKEN = re.compile(r"(?<!\d)-?\d+(?!\d)")


@dataclass(frozen=True, slots=True)
class ExactIntegerTask:
    task_id: str
    family: str
    prompt: str
    expression: str

    def to_dict(self) -> dict[str, str]:
        return {
            "expression": self.expression,
            "family": self.family,
            "prompt": self.prompt,
            "task_id": self.task_id,
            "value_sha256": hashlib.sha256(str(evaluate_expression(self.expression)).encode("utf-8")).hexdigest(),
        }


def evaluate_expression(expression: str) -> int:
    """Evaluate a generator-produced integer expression.

    Only non-negative integer literals and ``+ - * / ( )`` are accepted.
    ``/`` must divide evenly.
    """
    compact = expression.replace(" ", "")
    tokens = _EXPR_TOKEN.findall(compact)
    if "".join(tokens) != compact or not tokens:
        raise ValueError(f"unsupported expression: {expression}")
    value, index = _parse_sum(tokens, 0)
    if index != len(tokens):
        raise ValueError(f"trailing expression tokens: {expression}")
    return value


def _parse_sum(tokens: list[str], index: int) -> tuple[int, int]:
    value, index = _parse_product(tokens, index)
    while index < len(tokens) and tokens[index] in {"+", "-"}:
        op = tokens[index]
        rhs, index = _parse_product(tokens, index + 1)
        value = value + rhs if op == "+" else value - rhs
    return value, index


def _parse_product(tokens: list[str], index: int) -> tuple[int, int]:
    value, index = _parse_unary(tokens, index)
    while index < len(tokens) and tokens[index] in {"*", "/"}:
        op = tokens[index]
        rhs, index = _parse_unary(tokens, index + 1)
        if op == "*":
            value *= rhs
        else:
            if rhs == 0 or value % rhs != 0:
                raise ValueError("division is not exact")
            value //= rhs
    return value, index


def _parse_unary(tokens: list[str], index: int) -> tuple[int, int]:
    if index >= len(tokens):
        raise ValueError("expression ended early")
    if tokens[index] == "(":
        value, index = _parse_sum(tokens, index + 1)
        if index >= len(tokens) or tokens[index] != ")":
            raise ValueError("missing closing parenthesis")
        return value, index + 1
    if not tokens[index].isdigit():
        raise ValueError(f"expected integer, found {tokens[index]}")
    return int(tokens[index]), index + 1


def _prompt(expression: str) -> str:
    return (
        "Compute the integer value of this expression: "
        f"{expression}\n"
        "Write the result on the final line in exactly this form: #### <integer>\n"
        "The final line must contain no other text."
    )


def _answer_leaks(prompt: str, value: int) -> bool:
    return str(value) in _ANSWER_TOKEN.findall(prompt)


def _draw(family: str, rng: random.Random) -> tuple[str, int]:
    if family == "add_sub":
        left = rng.randint(20, 80)
        right = rng.randint(20, 80)
        sub = rng.randint(1, 9)
        expression = f"{left} + {right} - {sub}"
    elif family == "mul_add":
        left = rng.randint(3, 12)
        right = rng.randint(3, 12)
        extra = rng.randint(2, 9)
        expression = f"{left} * {right} + {extra}"
    elif family == "div_exact":
        factor = rng.randint(2, 9)
        multiplier = rng.randint(3, 12)
        divisor = rng.randint(2, 6)
        expression = f"({factor * divisor} * {multiplier}) / {divisor}"
    elif family == "paren":
        left = rng.randint(4, 18)
        right = rng.randint(4, 18)
        scale = rng.randint(2, 5)
        sub = rng.randint(1, 9)
        expression = f"({left} + {right}) * {scale} - {sub}"
    else:
        raise ValueError(family)
    return expression, evaluate_expression(expression)


def generate_tasks(
    *,
    seed: int = GENERATION_SEED,
    per_family: int = N_TASKS // len(FAMILIES),
) -> list[ExactIntegerTask]:
    if per_family < 1:
        raise ValueError("per_family must be positive")
    rng = random.Random(seed)
    tasks: list[ExactIntegerTask] = []
    for family in FAMILIES:
        produced = 0
        attempts = 0
        while produced < per_family:
            attempts += 1
            if attempts > per_family * 50:
                raise RuntimeError(f"could not draw non-leaking tasks for {family}")
            expression, value = _draw(family, rng)
            prompt = _prompt(expression)
            if _answer_leaks(prompt, value):
                continue
            tasks.append(
                ExactIntegerTask(
                    task_id=f"exact-integer-v1-{len(tasks):03d}",
                    family=family,
                    prompt=prompt,
                    expression=expression,
                )
            )
            produced += 1
    return tasks


def canonical_task_pack(tasks: Sequence[ExactIntegerTask]) -> str:
    lines = [json.dumps(task.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False) for task in tasks]
    return "\n".join(lines) + "\n"


def task_pack_sha256(tasks: Sequence[ExactIntegerTask]) -> str:
    return hashlib.sha256(canonical_task_pack(tasks).encode("utf-8")).hexdigest()


def load_task_pack(text: str) -> list[dict[str, str]]:
    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"task pack line {line_no} is not an object")
        rows.append({key: str(row[key]) for key in ("expression", "family", "prompt", "task_id", "value_sha256")})
    return rows


def verify_task_pack(rows: Sequence[Mapping[str, str]]) -> None:
    if len(rows) < 50:
        raise ValueError(f"held-out pack has {len(rows)} tasks; at least 50 are required")
    ids = [row["task_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("held-out task ids are not unique")
    for row in rows:
        value = evaluate_expression(row["expression"])
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        if digest != row["value_sha256"]:
            raise ValueError(f"{row['task_id']} value hash does not match its expression")
        if row["expression"] not in row["prompt"]:
            raise ValueError(f"{row['task_id']} prompt does not show its expression")
        if _answer_leaks(row["prompt"], value):
            raise ValueError(f"{row['task_id']} prompt contains its answer")
        if not row["prompt"].endswith("The final line must contain no other text."):
            raise ValueError(f"{row['task_id']} prompt does not use the frozen instruction")


def parse_final_integer(text: str) -> int | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    match = FINAL_LINE.fullmatch(lines[-1])
    if match is None:
        return None
    return int(match.group(1))


def grade_output(expression: str, output: str) -> bool:
    parsed = parse_final_integer(output)
    if parsed is None:
        return False
    return parsed == evaluate_expression(expression)


def wilson_interval(successes: int, total: int, *, z: float = WILSON_Z) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one task")
    if not 0 <= successes <= total:
        raise ValueError("successes must lie between 0 and total")
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    margin = z * math.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def pass_at_k(correct_by_task: Mapping[str, Sequence[bool]]) -> dict[str, Any]:
    if not correct_by_task:
        raise ValueError("pass@k requires tasks")
    widths = {len(flags) for flags in correct_by_task.values()}
    if widths != {SAMPLES_PER_TASK}:
        raise ValueError(f"each task must have exactly {SAMPLES_PER_TASK} graded samples")
    passed = sum(1 for flags in correct_by_task.values() if any(flags))
    total = len(correct_by_task)
    low, high = wilson_interval(passed, total)
    rate = passed / total
    if rate == 0.0:
        decision = "stop_zero"
    elif rate == 1.0:
        decision = "stop_saturated"
    else:
        decision = "nondegenerate_base_signal"
    return {
        "decision": decision,
        "k": SAMPLES_PER_TASK,
        "passed_tasks": passed,
        "pass_rate": rate,
        "tasks": total,
        "wilson95_high": high,
        "wilson95_low": low,
    }


def decision_text(decision: str) -> str:
    if decision == "stop_zero":
        return (
            "pass@k is zero. Stop. Do not change the prompt, parser, sample count, "
            "or threshold, and do not start an update."
        )
    if decision == "stop_saturated":
        return (
            "pass@k is one. The pack is saturated for this model and decoder. "
            "Stop. Do not retune it into a training set."
        )
    if decision == "nondegenerate_base_signal":
        return (
            "pass@k is strictly between zero and one. This is a base-model feasibility "
            "result for this pack only. It authorizes a separate update-cost smoke, "
            "not a policy-update comparison and not a claim of capability."
        )
    raise ValueError(decision)


def comparison_refusal() -> dict[str, str]:
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "refused",
        "reason": (
            "A multi-seed best-of-n versus GRPO or RFT comparison is not started. "
            "This pack is held-out evaluation. A non-degenerate base gate would "
            "authorize only a separate update-cost smoke after save/reload and "
            "resource checks. Those records are not inputs to this entry point."
        ),
    }


def sample_seed(task_index: int, sample_index: int) -> int:
    if task_index < 0 or not 0 <= sample_index < SAMPLES_PER_TASK:
        raise ValueError("sample index is outside the frozen grid")
    return SAMPLING_SEED + task_index * SAMPLES_PER_TASK + sample_index


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path} line {line_no} is not valid JSON. A disconnect can leave a partial "
                "final line. Inspect that line before deleting only the partial line; completed lines stay."
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path} line {line_no} is not an object")
        rows.append(row)
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def completed_samples(
    rows: Iterable[Mapping[str, Any]],
    *,
    pack_sha256: str,
    backend: str,
    model_id: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("pack_sha256") != pack_sha256 or row.get("backend") != backend:
            raise ValueError("resume file was written for a different pack or backend")
        if row.get("model_id") != model_id:
            raise ValueError("resume file was written for a different model")
        key = (str(row["task_id"]), int(row["sample_index"]))
        if key in completed:
            raise ValueError(f"duplicate sample {key[0]} #{key[1]}")
        completed[key] = dict(row)
    return completed


def summarize_samples(tasks: Sequence[Mapping[str, str]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[bool | None]] = {task["task_id"]: [None] * SAMPLES_PER_TASK for task in tasks}
    for row in rows:
        task_id = str(row["task_id"])
        index = int(row["sample_index"])
        if task_id not in by_task or not 0 <= index < SAMPLES_PER_TASK:
            raise ValueError(f"sample {(task_id, index)} is outside the frozen grid")
        if by_task[task_id][index] is not None:
            raise ValueError(f"duplicate sample {task_id} #{index}")
        by_task[task_id][index] = bool(row["correct"])
    if any(flag is None for flags in by_task.values() for flag in flags):
        done = sum(flag is not None for flags in by_task.values() for flag in flags)
        return {"complete": False, "graded_samples": done, "expected_samples": len(tasks) * SAMPLES_PER_TASK}
    correct = {task_id: [bool(flag) for flag in flags] for task_id, flags in by_task.items()}
    summary = pass_at_k(correct)
    summary["complete"] = True
    summary["statement"] = decision_text(str(summary["decision"]))
    return summary


Sampler = Callable[[str, int], str]


def run_gate(
    tasks: Sequence[Mapping[str, str]],
    sampler: Sampler,
    output_dir: Path,
    *,
    pack_sha256: str,
    backend: str,
    model_id: str,
) -> dict[str, Any]:
    verify_task_pack(tasks)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    existing = completed_samples(
        read_jsonl(samples_path),
        pack_sha256=pack_sha256,
        backend=backend,
        model_id=model_id,
    )
    for task_index, task in enumerate(tasks):
        for sample_index in range(SAMPLES_PER_TASK):
            key = (task["task_id"], sample_index)
            if key in existing:
                continue
            seed = sample_seed(task_index, sample_index)
            output = sampler(task["prompt"], seed)
            correct = grade_output(task["expression"], output)
            row = {
                "backend": backend,
                "correct": correct,
                "model_id": model_id,
                "output": output,
                "pack_sha256": pack_sha256,
                "sample_index": sample_index,
                "seed": seed,
                "task_id": task["task_id"],
            }
            append_jsonl(samples_path, row)
            existing[key] = row
    summary = summarize_samples(tasks, list(existing.values()))
    summary["backend"] = backend
    summary["model_id"] = model_id
    summary["pack_sha256"] = pack_sha256
    summary["protocol_id"] = PROTOCOL_ID
    if summary["complete"]:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary
