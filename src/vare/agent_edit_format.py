"""Edit formats for coding-agent evaluations, separated from task scoring.

``edit_format`` may be ``unified_diff``, ``whole_file``, or
``structured_replace``. The latter is executed only through the runner's strict
``replace_text`` tool, not this free-form parser. A parsed edit is not a passing
grade. ``format_parse_failure`` is recorded even when ``task_passed`` is false.
``outcome`` is a separate label:
``no_tool_call``, ``malformed_edit``, ``edit_applied_but_wrong``, ``timeout``,
or ``pass``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping, Sequence

EDIT_FORMATS = ("unified_diff", "whole_file", "structured_replace")
OUTCOMES = ("no_tool_call", "malformed_edit", "edit_applied_but_wrong", "timeout", "pass")
_TIMEOUT_STOPS = frozenset({"wall_time_exhausted", "token_budget_exhausted"})
_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_HEADING = re.compile(r"(?m)^(?:FILE|path):\s*(\S+)\s*$")
_DIFF_HEADER = re.compile(r"(?m)^--- \S")
_FORBIDDEN = (
    "GIT binary patch",
    "new file mode",
    "deleted file mode",
    "rename from ",
    "rename to ",
    "Subproject commit",
)


class EditFormatError(ValueError):
    """The requested edit format is not one of the two supported settings."""


@dataclass(frozen=True)
class FileUpdate:
    path: str
    content: str | None = None


@dataclass(frozen=True)
class ParsedEdit:
    edit_format: str
    paths: tuple[str, ...]
    diff_text: str | None
    files: tuple[FileUpdate, ...]


@dataclass(frozen=True)
class ParseFailure:
    edit_format: str
    reason: str


@dataclass(frozen=True)
class ApplyResult:
    applied: bool
    paths: tuple[str, ...]
    error: str | None = None


def require_edit_format(value: str) -> str:
    if value not in EDIT_FORMATS:
        raise EditFormatError("edit_format must be unified_diff, whole_file, or structured_replace")
    return value


def parse_submission(text: str, *, edit_format: str) -> ParsedEdit | ParseFailure:
    """Parse one configured format. The other format is a format failure."""
    require_edit_format(edit_format)
    if not isinstance(text, str) or not text.strip():
        return ParseFailure(edit_format, "no_edit")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    has_diff = _DIFF_HEADER.search(normalized) is not None or "\ndiff --git " in "\n" + normalized
    has_file = _has_whole_file_marker(normalized)
    if edit_format == "unified_diff":
        if has_file and not has_diff:
            return ParseFailure(edit_format, "wrong_format")
        diff = _extract_diff(normalized)
        if diff is None:
            if not _looks_like_edit(normalized):
                return ParseFailure(edit_format, "no_edit")
            return ParseFailure(edit_format, "malformed_diff")
        if any(feature in diff for feature in _FORBIDDEN):
            return ParseFailure(edit_format, "malformed_diff")
        paths = _diff_paths(diff)
        if not paths:
            return ParseFailure(edit_format, "malformed_diff")
        return ParsedEdit(edit_format, tuple(paths), diff, tuple(FileUpdate(path) for path in paths))
    if edit_format == "structured_replace":
        return ParseFailure(edit_format, "wrong_format" if has_diff or has_file else "no_edit")
    if has_diff and not has_file:
        return ParseFailure(edit_format, "wrong_format")
    files = _whole_files(normalized)
    if not files:
        if not _looks_like_edit(normalized):
            return ParseFailure(edit_format, "no_edit")
        return ParseFailure(edit_format, "malformed_file")
    file_paths = tuple(path for path, _content in files)
    if len(set(file_paths)) != len(file_paths):
        return ParseFailure(edit_format, "malformed_file")
    updates = tuple(FileUpdate(path, content) for path, content in files)
    return ParsedEdit(edit_format, file_paths, None, updates)


def apply_parsed_edit(workspace: Path, edit: ParsedEdit, allowed: set[str]) -> ApplyResult:
    """Write an allowlisted existing file, or apply a unified diff with git."""
    rejected = _reject_paths(edit.paths, allowed, workspace)
    if rejected is not None:
        return ApplyResult(False, tuple(edit.paths), rejected)
    if edit.edit_format == "whole_file":
        for update in edit.files:
            if update.content is None or "\x00" in update.content:
                return ApplyResult(False, edit.paths, "malformed_file")
            target = workspace / update.path
            target.write_text(update.content, encoding="utf-8")
        return ApplyResult(True, edit.paths, None)
    if not edit.diff_text:
        return ApplyResult(False, edit.paths, "malformed_diff")
    checked = subprocess.run(
        ["git", "apply", "--check", "--"],
        cwd=workspace, input=edit.diff_text, text=True, capture_output=True, check=False,
    )
    if checked.returncode != 0:
        detail = (checked.stderr or checked.stdout or "patch failed validation").strip()
        return ApplyResult(False, edit.paths, detail[:1000])
    applied = subprocess.run(
        ["git", "apply", "--"],
        cwd=workspace, input=edit.diff_text, text=True, capture_output=True, check=False,
    )
    if applied.returncode != 0:
        detail = (applied.stderr or "patch failed").strip()
        return ApplyResult(False, edit.paths, detail[:1000])
    return ApplyResult(True, edit.paths, None)


def classify_outcome(
    *,
    task_passed: bool,
    stop_reason: str | None,
    source_changed: bool,
    edit_attempted: bool,
    edit_applied: bool,
    format_parse_failure: str | None,
    timed_out: bool,
) -> str:
    """Label the attempt. A parse failure is not the same event as a wrong task."""
    if task_passed:
        return "pass"
    if timed_out or stop_reason in _TIMEOUT_STOPS:
        return "timeout"
    if source_changed or edit_applied:
        return "edit_applied_but_wrong"
    if format_parse_failure or edit_attempted:
        return "malformed_edit"
    return "no_tool_call"


def per_task_reward_variance(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    """Sample variance of reward within each task. One sample has no variance."""
    grouped: dict[str, list[float]] = {}
    missing: dict[str, int] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        reward = row.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            missing[task_id] = missing.get(task_id, 0) + 1
            continue
        grouped.setdefault(task_id, []).append(float(reward))
    report: dict[str, dict[str, object]] = {}
    for task_id in sorted(set(grouped) | set(missing)):
        rewards = grouped.get(task_id, [])
        count = len(rewards)
        mean = sum(rewards) / count if count else None
        if count < 2 or mean is None:
            variance = None
        else:
            variance = sum((value - mean) ** 2 for value in rewards) / (count - 1)
        report[task_id] = {
            "n": count,
            "mean": mean,
            "sample_variance": variance,
            "missing_reward": missing.get(task_id, 0),
        }
    return report


def pass_at_k(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Fraction of tasks with at least one ``task_passed`` sample."""
    by_task: dict[str, list[bool]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        by_task.setdefault(task_id, []).append(row.get("task_passed") is True)
    tasks = len(by_task)
    passed = sum(1 for values in by_task.values() if any(values))
    return {
        "tasks": tasks,
        "tasks_with_any_success": passed,
        "pass_at_k": (passed / tasks) if tasks else None,
    }


def inference_arms(protocol: Mapping[str, object]) -> list[dict[str, object]]:
    """Arms from ``inference.arms``, or the single v1 primary inference block."""
    inference = protocol.get("inference")
    if not isinstance(inference, dict):
        raise EditFormatError("protocol inference block is missing")
    if "arms" in inference:
        arms = inference["arms"]
        if not isinstance(arms, list) or not arms:
            raise EditFormatError("inference.arms must be a non-empty list")
        normalized = []
        for arm in arms:
            if not isinstance(arm, dict):
                raise EditFormatError("each inference arm must be an object")
            normalized.append(_normalize_arm(arm, inference))
        return normalized
    models = protocol.get("models")
    seeds = [101]
    if isinstance(models, dict):
        primary = models.get("primary")
        if isinstance(primary, dict) and isinstance(primary.get("seeds"), list):
            seeds = list(primary["seeds"])
    return [_normalize_arm({
        "name": "primary",
        "temperature": inference.get("temperature", 0),
        "do_sample": inference.get("do_sample", False),
        "top_p": inference.get("top_p", 0.95),
        "top_k": inference.get("top_k", 20),
        "seeds": seeds,
    }, inference)]


def _normalize_arm(arm: Mapping[str, object], inference: Mapping[str, object]) -> dict[str, object]:
    name = arm.get("name")
    seeds = arm.get("seeds")
    if not isinstance(name, str) or not name:
        raise EditFormatError("inference arm name is missing")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        raise EditFormatError("inference arm seeds must be a non-empty integer list")
    do_sample = bool(arm.get("do_sample", False))
    temperature = arm.get("temperature", 0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise EditFormatError("inference arm temperature must be a number")
    top_p = _optional_number(arm.get("top_p", 0.95 if do_sample else None), "top_p")
    top_k = _optional_int(arm.get("top_k", 20 if do_sample else None), "top_k")
    edit_format = inference.get("edit_format", "unified_diff")
    if not isinstance(edit_format, str):
        raise EditFormatError("edit_format must be a string")
    seed_list = [seed for seed in seeds if isinstance(seed, int) and not isinstance(seed, bool)]
    return {
        "name": name,
        "temperature": float(temperature),
        "do_sample": do_sample,
        "top_p": top_p,
        "top_k": top_k,
        "seeds": seed_list,
        "edit_format": require_edit_format(edit_format),
        "samples_per_task": len(seed_list),
    }


def _optional_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EditFormatError(f"{label} must be a number")
    return float(value)


def _optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise EditFormatError(f"{label} must be an integer")
    return value


def _looks_like_edit(text: str) -> bool:
    if "```" in text:
        return True
    if _DIFF_HEADER.search(text):
        return True
    if _HEADING.search(text):
        return True
    return False


def _has_whole_file_marker(text: str) -> bool:
    if _HEADING.search(text):
        return True
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "content" in payload and "path" in payload:
            return True
    for match in _FENCE.finditer(text):
        if _path_from_info(match.group(1)):
            return True
    return False


def _extract_diff(text: str) -> str | None:
    lines = text.split("\n")
    start = None
    for index, line in enumerate(lines):
        if line.startswith("diff --git "):
            start = index
            break
        if line.startswith("--- ") and index + 1 < len(lines) and lines[index + 1].startswith("+++ "):
            start = index
            break
    if start is None:
        return None
    kept: list[str] = []
    seen_hunk = False
    for line in lines[start:]:
        if line.startswith("```"):
            break
        if kept and seen_hunk and line and not _diff_continuation(line):
            break
        kept.append(line)
        if line.startswith("@@"):
            seen_hunk = True
    while kept and kept[-1] == "":
        kept.pop()
    if not seen_hunk:
        return None
    return "\n".join(kept) + "\n"


def _diff_continuation(line: str) -> bool:
    if line.startswith((
        "diff --git ", "index ", "--- ", "+++ ", "@@",
        "old mode ", "new mode ", "similarity index ", "dissimilarity index ",
    )):
        return True
    return line[:1] in {" ", "+", "-", "\\"} or line == ""


def _diff_paths(diff: str) -> list[str]:
    paths: list[str] = []
    for line in diff.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        token = line[4:].strip()
        if token == "/dev/null":
            return []
        if token.startswith(("a/", "b/")):
            token = token[2:]
        if token.startswith("/") or ".." in Path(token).parts:
            return []
        if token not in paths:
            paths.append(token)
    return paths


def _whole_files(text: str) -> list[tuple[str, str]]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("path"), str) and isinstance(payload.get("content"), str):
            path = _clean_path(payload["path"])
            if path is None:
                return []
            return [(path, payload["content"])]
    found: list[tuple[str, str]] = []
    headings = list(_HEADING.finditer(text))
    fences = list(_FENCE.finditer(text))
    if headings and fences:
        for heading, fence in zip(headings, fences, strict=False):
            path = _clean_path(heading.group(1))
            info_path = _path_from_info(fence.group(1))
            if path is None:
                return []
            if info_path is not None and info_path != path:
                return []
            found.append((path, fence.group(2)))
        return found
    for fence in fences:
        path = _path_from_info(fence.group(1))
        if path is None:
            return []
        found.append((path, fence.group(2)))
    return found


def _path_from_info(info: str) -> str | None:
    info = info.strip()
    if not info or info in {"diff", "patch"}:
        return None
    for part in info.split():
        if part.startswith("path="):
            return _clean_path(part.split("=", 1)[1])
        if part.startswith("python") or part in {"py", "text"}:
            continue
        cleaned = _clean_path(part)
        if cleaned is not None:
            return cleaned
    return None


def _clean_path(raw: str) -> str | None:
    token = raw.strip().strip("`").strip('"').strip("'")
    if not token or token.startswith("/") or "\\" in token:
        return None
    parts = Path(token).parts
    if ".." in parts or not parts:
        return None
    return token


def _reject_paths(paths: Sequence[str], allowed: set[str], workspace: Path) -> str | None:
    if not paths or any(path not in allowed for path in paths):
        return "patch paths must be existing allowlisted source files"
    for path in paths:
        target = workspace / path
        if target.is_symlink() or not target.is_file():
            return "patch paths must be existing allowlisted source files"
        if os.path.commonpath([str(workspace.resolve()), str(target.resolve())]) != str(workspace.resolve()):
            return "patch paths must be existing allowlisted source files"
    return None
