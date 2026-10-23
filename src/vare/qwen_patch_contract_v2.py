"""Strict unified-diff parsing and phase-aware grading outcomes for coding agents."""

from __future__ import annotations

import re
import subprocess
from typing import Any, Callable


class PatchFormatError(ValueError):
    """Candidate output is not exactly one complete allowlisted unified diff."""


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")
_INDEX = re.compile(r"^index [0-9a-fA-F]+\.\.[0-9a-fA-F]+(?: [0-7]{6})?$")


def validate_unified_diff(raw: str, target_path: str) -> str:
    """Validate one fully consumed Git diff and return it unchanged."""
    if not raw or "\x00" in raw or "\r" in raw:
        raise PatchFormatError("empty or unsupported candidate output")
    text = raw[:-1] if raw.endswith("\n") else raw
    lines = text.split("\n")
    expected_header = f"diff --git a/{target_path} b/{target_path}"
    if not lines or lines[0] != expected_header:
        raise PatchFormatError("output must start with the exact target diff header")
    i = 1
    if i < len(lines) and _INDEX.fullmatch(lines[i]):
        i += 1
    if len(lines) < i + 4 or lines[i] != f"--- a/{target_path}":
        raise PatchFormatError("missing exact old-file header")
    if lines[i + 1] != f"+++ b/{target_path}":
        raise PatchFormatError("missing exact new-file header")
    i += 2
    old_hunks = new_hunks = 0
    while i < len(lines):
        match = _HUNK.fullmatch(lines[i])
        if match is None:
            raise PatchFormatError("unexpected text outside a unified-diff hunk")
        old_remaining = int(match.group(2) or "1")
        new_remaining = int(match.group(4) or "1")
        i += 1
        while old_remaining or new_remaining:
            if i >= len(lines):
                raise PatchFormatError("truncated unified-diff hunk")
            row = lines[i]
            if row == r"\ No newline at end of file":
                i += 1
                continue
            if not row or row[0] not in " +-":
                raise PatchFormatError("invalid unified-diff hunk line")
            if row[0] in " -":
                if old_remaining == 0:
                    raise PatchFormatError("hunk contains excess old-side lines")
                old_remaining -= 1
            if row[0] in " +":
                if new_remaining == 0:
                    raise PatchFormatError("hunk contains excess new-side lines")
                new_remaining -= 1
            i += 1
        old_hunks += 1
        new_hunks += 1
    if not old_hunks or old_hunks != new_hunks:
        raise PatchFormatError("diff must contain at least one complete hunk")
    return text + ("\n" if raw.endswith("\n") else "")


def grade_with_phase_taxonomy(
    raw: str,
    target_path: str,
    *,
    apply_patch: Callable[[str], Any],
    grade_candidate: Callable[[Any], dict[str, Any]],
    validate_patch: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Separate format, patch-application, grader, and infrastructure failures."""
    try:
        patch = validate_unified_diff(raw, target_path)
    except PatchFormatError as exc:
        return {"passed": False, "failure": "format_failure", "detail": str(exc)}

    if validate_patch is not None:
        try:
            validate_patch(patch)
        except Exception as exc:
            return {"passed": False, "failure": "format_failure", "detail": str(exc)[:500]}

    try:
        candidate = apply_patch(patch)
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        return {"passed": False, "failure": "patch_application_timeout", "detail": str(exc)[:500]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"passed": False, "failure": "application_infrastructure", "detail": str(exc)[:500]}
    except Exception as exc:  # task adapters expose typed ValueErrors for apply failures
        return {"passed": False, "failure": "patch_application_failure", "detail": str(exc)[:500]}

    try:
        result = grade_candidate(candidate)
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        return {"passed": False, "failure": "grader_timeout", "detail": str(exc)[:500]}
    except (OSError, RuntimeError) as exc:
        return {"passed": False, "failure": "grader_infrastructure", "detail": str(exc)[:500]}
    except Exception as exc:
        return {"passed": False, "failure": "grader_failure", "detail": str(exc)[:500]}
    if result.get("passed") is True:
        return {"passed": True, "failure": None, "grading": result}
    failures = [case.get("failure") for case in result.get("cases", [])]
    if "timeout" in failures:
        failure = "grader_timeout"
    elif failures and all(reason in {"wrong_output", "wrong_exception"} for reason in failures):
        failure = "grader_wrong_answer"
    elif failures and any(reason in {"missing_result", "malformed_result", "runner_error"} for reason in failures):
        failure = "grader_infrastructure"
    else:
        failure = "grader_failure"
    return {"passed": False, "failure": failure, "grading": result}
