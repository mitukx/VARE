"""Observation-conditioned proposers.

``DiagnosticProposer`` is a deterministic controller policy. It is not a
trained model. Its first repair for each failure class is a plausible wrong
edit; a later edit is tried only after that measurement is rejected.

``OracleProposer`` copies a known-good control. It exists to calibrate the
grader and must not be reported as agent success.
"""

from __future__ import annotations

import math
from typing import Any

from .spec import ResearchTask


def _num(metrics: dict[str, Any], key: str, default: float = 1.0) -> float:
    if key not in metrics or metrics[key] is None:
        return default
    try:
        number = float(metrics[key])
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


class HypothesisNotApplicable(RuntimeError):
    """The current source does not match a repair operator."""


def _require_once(source: str, old: str, label: str) -> None:
    if source.count(old) != 1:
        raise HypothesisNotApplicable(f"{label} pattern count is {source.count(old)}")


def extra_accumulation_scale(source: str) -> str:
    """Wrong hypothesis: scale the accumulator again after the loop."""
    target = "acc = [value * inv for value in acc]\n"
    _require_once(source, target, "accumulation scale")
    lines = source.splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if line.strip() == target.strip())
    indent = len(lines[index]) - len(lines[index].lstrip(" "))
    for_at = None
    for_indent = None
    for cursor in range(index, -1, -1):
        raw = lines[cursor]
        if not raw.strip():
            continue
        current = len(raw) - len(raw.lstrip(" "))
        if current < indent and raw.lstrip().startswith("for "):
            for_at = cursor
            for_indent = current
            break
    if for_at is None or for_indent is None:
        raise HypothesisNotApplicable("scale is not inside a for-loop")
    end = for_at + 1
    while end < len(lines):
        if lines[end].strip() and (len(lines[end]) - len(lines[end].lstrip(" "))) <= for_indent:
            break
        end += 1
    lines.insert(end, " " * for_indent + target.strip() + "\n")
    return "".join(lines)


def hoist_accumulation_scale(source: str) -> str:
    """Move the in-loop accumulator scale to a single scale after the loop."""
    target = "acc = [value * inv for value in acc]\n"
    _require_once(source, target, "accumulation scale")
    lines = source.splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if line.strip() == target.strip())
    indent = len(lines[index]) - len(lines[index].lstrip(" "))
    for_at = None
    for_indent = None
    for cursor in range(index, -1, -1):
        raw = lines[cursor]
        if not raw.strip():
            continue
        current = len(raw) - len(raw.lstrip(" "))
        if current < indent and raw.lstrip().startswith("for "):
            for_at = cursor
            for_indent = current
            break
    if for_at is None or for_indent is None:
        raise HypothesisNotApplicable("scale is not inside a for-loop")
    line = lines.pop(index)
    end = for_at + 1
    while end < len(lines):
        if lines[end].strip() and (len(lines[end]) - len(lines[end].lstrip(" "))) <= for_indent:
            break
        end += 1
    lines.insert(end, " " * for_indent + line.strip() + "\n")
    return "".join(lines)


def off_by_one_average(source: str) -> str:
    old = "inv = 1.0 / passes\n"
    _require_once(source, old, "pass average")
    return source.replace(old, "inv = 1.0 / (passes + 1)\n", 1)


def hoist_identical_passes(source: str) -> str:
    start = "    for _ in range(passes):\n"
    _require_once(source, start, "identical passes")
    lines = source.splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if line == start)
    end = index + 1
    while end < len(lines) and (lines[end].startswith("        ") or lines[end].strip() == ""):
        end += 1
    block: list[str] = []
    for line in lines[index + 1:end]:
        block.append(line[4:] if line.startswith("        ") else line)
    text = "".join(lines[:index] + block + lines[end:])
    old_inv = "    inv = 1.0 / passes\n"
    old_ret = "    return [[value * inv for value in row] for row in acc]\n"
    _require_once(text, old_inv, "hoisted average")
    _require_once(text, old_ret, "hoisted return")
    text = text.replace(old_inv, "", 1)
    return text.replace(old_ret, "    return acc\n", 1)


def shrink_learning_rate(source: str) -> str:
    old = "LEARNING_RATE = 0.25\n"
    _require_once(source, old, "learning rate")
    return source.replace(old, "LEARNING_RATE = 0.025\n", 1)


def restore_descent_sign(source: str) -> str:
    old = "        resid = y - pred\n"
    _require_once(source, old, "residual sign")
    return source.replace(old, "        resid = pred - y\n", 1)


def _tried(observation: dict[str, Any]) -> set[str]:
    tried = set()
    for row in observation.get("memory") or []:
        if isinstance(row, dict) and row.get("hypothesis_id"):
            tried.add(str(row["hypothesis_id"]))
    return tried


def _baseline_files(observation: dict[str, Any]) -> dict[str, str]:
    files = observation.get("baseline_files")
    if not isinstance(files, dict):
        return {}
    return {str(key): str(value) for key, value in files.items()}


def _metrics(observation: dict[str, Any]) -> dict[str, Any]:
    public = observation.get("last_public")
    if not isinstance(public, dict):
        return {}
    metrics = public.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _file_edit(hypothesis_id: str, filename: str, content: str) -> dict[str, Any]:
    return {
        "type": "edit",
        "hypothesis_id": hypothesis_id,
        "files": {filename: content},
        "measure": True,
    }


class DiagnosticProposer:
    name = "diagnostic_feedback_v1"
    kind = "diagnostic_policy"

    def propose(self, observation: dict[str, Any]) -> dict[str, Any]:
        tried = _tried(observation)
        if "baseline" not in tried:
            return {"type": "run_public", "hypothesis_id": "baseline"}
        files = _baseline_files(observation)
        public = observation.get("last_public")
        if not isinstance(public, dict) or public.get("classification") != "public_measurement":
            return {
                "type": "stop",
                "hypothesis_id": "stop_unmeasured",
                "reason": "the last public experiment did not return measurements",
            }
        if "accum.py" in files:
            return self._grad(files["accum.py"], _metrics(observation), tried)
        if "project.py" in files:
            return self._score(files["project.py"], _metrics(observation), tried)
        if "experiment.py" in files:
            return self._affine(files["experiment.py"], _metrics(observation), tried)
        return {"type": "stop", "hypothesis_id": "stop_unknown_task", "reason": "no repair family"}

    def _grad(self, source: str, metrics: dict[str, Any], tried: set[str]) -> dict[str, Any]:
        error = _num(metrics, "max_abs_error", default=1.0)
        if error <= 1e-6 and "submit" not in tried:
            return {"type": "submit", "hypothesis_id": "submit"}
        if "extra_normalization" not in tried:
            return _file_edit("extra_normalization", "accum.py", extra_accumulation_scale(source))
        if "hoist_accumulation_scale" not in tried:
            return _file_edit(
                "hoist_accumulation_scale",
                "accum.py",
                hoist_accumulation_scale(source),
            )
        return {"type": "stop", "hypothesis_id": "stop_exhausted", "reason": "grad hypotheses exhausted"}

    def _score(self, source: str, metrics: dict[str, Any], tried: set[str]) -> dict[str, Any]:
        error = _num(metrics, "max_abs_error", default=1.0)
        if "off_by_one_average" not in tried:
            return _file_edit("off_by_one_average", "project.py", off_by_one_average(source))
        if "hoist_identical_passes" not in tried:
            return _file_edit(
                "hoist_identical_passes",
                "project.py",
                hoist_identical_passes(source),
            )
        if error <= 1e-8 and "submit" not in tried:
            return {"type": "submit", "hypothesis_id": "submit"}
        return {"type": "stop", "hypothesis_id": "stop_exhausted", "reason": "score hypotheses exhausted"}

    def _affine(self, source: str, metrics: dict[str, Any], tried: set[str]) -> dict[str, Any]:
        mse = _num(metrics, "recomputed_mse", default=1.0)
        step = _num(metrics, "gradient_step_abs_error", default=1.0)
        diverged = bool(metrics.get("diverged"))
        solved = mse <= 1e-3 and step <= 1e-6 and not diverged
        if "shrink_learning_rate" not in tried:
            return _file_edit("shrink_learning_rate", "experiment.py", shrink_learning_rate(source))
        if "restore_descent_sign" not in tried:
            return _file_edit("restore_descent_sign", "experiment.py", restore_descent_sign(source))
        if solved and "submit" not in tried:
            return {"type": "submit", "hypothesis_id": "submit"}
        return {"type": "stop", "hypothesis_id": "stop_exhausted", "reason": "debug hypotheses exhausted"}


class FirstHypothesisProposer(DiagnosticProposer):
    """Submit immediately after the first wrong repair. A negative control."""

    name = "first_hypothesis_v1"
    kind = "single_attempt_control"

    def propose(self, observation: dict[str, Any]) -> dict[str, Any]:
        tried = _tried(observation)
        action = super().propose(observation)
        if action["type"] == "edit" and "submit" not in tried and _edit_already_measured(tried):
            return {"type": "submit", "hypothesis_id": "submit"}
        return action


def _edit_already_measured(tried: set[str]) -> bool:
    return bool(tried - {"baseline"})


class OracleProposer:
    name = "oracle_control_v1"
    kind = "oracle_control"

    def __init__(self, task: ResearchTask) -> None:
        self.task = task

    def propose(self, observation: dict[str, Any]) -> dict[str, Any]:
        tried = _tried(observation)
        if "baseline" not in tried:
            return {"type": "run_public", "hypothesis_id": "baseline"}
        if "oracle_known_good" not in tried:
            files = {}
            control = self.task.control_dir("known_good")
            for name in self.task.editable_files:
                files[name] = (control / name).read_text(encoding="utf-8")
            return {
                "type": "edit",
                "hypothesis_id": "oracle_known_good",
                "files": files,
                "measure": True,
            }
        return {"type": "submit", "hypothesis_id": "submit"}


def proposer_for(name: str, task: ResearchTask):
    if name == "diagnostic":
        return DiagnosticProposer()
    if name == "first-hypothesis":
        return FirstHypothesisProposer()
    if name == "oracle":
        return OracleProposer(task)
    if name == "external":
        return None
    raise ValueError(f"unknown proposer {name}")
