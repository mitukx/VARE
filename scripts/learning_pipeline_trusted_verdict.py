"""Verdicts decided by the trusted parent from authenticated per-test frames (v2).

``learning_pipeline_isolated_grading.grade_isolated`` (v1 of the sandboxed
grader) parses pytest's ``PASSED <id>`` lines from the sandbox's standard
output. Those lines are produced by the same process that imports the
candidate, so the verdict channel is candidate-writable; the audit reproduced
three false passes against the native grader along exactly this channel
(argv echo + ``os._exit(0)``, a canary-aware ``os._exit(1)`` variant, and a
restore of the reference from the clone's git history).

This module separates *test execution* from *judging*:

* execution: ``learning_pipeline_trusted_runner`` runs inside the sandbox and
  decides each item's outcome itself, by calling the test function directly
  (pytest's reporting hooks, report objects and terminal output are not used);
* channel: each outcome is written as a frame authenticated with HMAC-SHA256
  under a fresh 256-bit key that is delivered on standard input only and read
  before any candidate code is imported. Text that candidate code writes to
  any descriptor cannot form a valid frame without that key;
* judging: the parent (outside the sandbox) accepts a pass only if the frame
  stream is complete and consistent (``start`` → ``collected`` → one
  ``result`` per collected item → ``finish``, contiguous sequence numbers, the
  per-call nonce, no repeated ids), the ``finish`` exit code equals both the
  process exit code and the code implied by the results, the two per-call
  canaries have their expected outcomes, and every requested id is PASSED.

Anything else is ``passed: False`` with a ``verdict_integrity_*`` reason.
The isolation preconditions (attestation of the same sandbox configuration,
static scan of added lines) are those of the v1 sandboxed grader.
"""
from __future__ import annotations

import difflib
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from scripts import learning_pipeline_isolated_grading as ig

SCHEMA = "learning_pipeline_isolated_grade_v2"
VERDICT_CHANNEL = "hmac_frames_v1"
RUNNER_PATH = Path(__file__).with_name("learning_pipeline_trusted_runner.py")
FRAME_PREFIX = "VARE_FRAME "
PASSING = "PASSED"


# v1 patterns (unchanged, so the recorded run-001 scan stays reproducible) plus
# patterns aimed at the v2 channel: in-process introspection that could reach
# the runner's key or closures, raw descriptor writes, frame text, and
# comparison overrides that make assertions vacuous.
BLOCKING_PATTERNS_V2: tuple[tuple[str, str], ...] = (*ig.BLOCKING_PATTERNS, (
    "introspection",
    r"\bgc\b|\binspect\.(?:currentframe|stack|getclosurevars|getmembers)\b|__closure__|cell_contents"
    r"|__code__|__globals__|\bsys\._getframe\b|\bf_(?:locals|globals|back)\b|/proc\b",
), (
    "fd_access", r"\bos\.(?:write|dup2?|fdopen|close|closerange|pipe|read)\b|\bfileno\b",
), (
    "frame_forgery", r"VARE_FRAME|\bhmac\b",
), (
    "comparison_override", r"\bdef\s+__(?:eq|ne|bool|hash)__\b|__(?:eq|ne)__\s*=",
))


def static_flags_v2(buggy_text: str, candidate_text: str) -> list[str]:
    """Blocking pattern names (v2 list) found in lines the candidate added."""
    added = [ln[1:] for ln in difflib.unified_diff(buggy_text.splitlines(), candidate_text.splitlines(),
                                                    lineterm="", n=0)
             if ln.startswith("+") and not ln.startswith("+++")]
    text = "\n".join(added)
    return [name for name, pat in BLOCKING_PATTERNS_V2 if re.search(pat, text)]


def runner_source() -> str:
    return RUNNER_PATH.read_text(encoding="utf-8")


def runner_sha256() -> str:
    return hashlib.sha256(runner_source().encode()).hexdigest()


class FrameError(ValueError):
    """The frame stream is not a complete, authentic record of one run."""


def frame_line(key: bytes, body: Mapping[str, Any]) -> str:
    """Encode one frame exactly as the runner does (used by tests and audits)."""
    payload = json.dumps(dict(body), sort_keys=True, separators=(",", ":"))
    return FRAME_PREFIX + hmac.new(key, payload.encode(), hashlib.sha256).hexdigest() + " " + payload


def parse_frames(stdout: str, *, key: bytes, nonce: str) -> list[dict[str, Any]]:
    """Authenticated frames in order; raise FrameError on any inconsistency.

    Lines that do not start with the frame prefix are ignored (they carry no
    authority); a line that starts with it must authenticate.
    """
    frames: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.startswith(FRAME_PREFIX):
            continue
        parts = line[len(FRAME_PREFIX):].split(" ", 1)
        if len(parts) != 2:
            raise FrameError("verdict_integrity_bad_frame")
        mac, payload = parts
        want = hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, want):
            raise FrameError("verdict_integrity_bad_frame")
        try:
            body = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise FrameError("verdict_integrity_bad_frame") from exc
        if not isinstance(body, dict) or body.get("nonce") != nonce or body.get("seq") != len(frames):
            raise FrameError("verdict_integrity_sequence")
        frames.append(body)
    return frames


def results_from_frames(frames: Sequence[Mapping[str, Any]], *, returncode: int) -> dict[str, str]:
    """{nodeid: status} from a complete frame stream, or FrameError."""
    if not frames or frames[0].get("kind") != "start":
        raise FrameError("verdict_integrity_no_start")
    if frames[-1].get("kind") != "finish":
        raise FrameError("verdict_integrity_no_finish")
    middle = frames[1:-1]
    if not middle or middle[0].get("kind") != "collected" or not isinstance(middle[0].get("ids"), list):
        raise FrameError("verdict_integrity_no_collection")
    collected = list(middle[0]["ids"])
    status: dict[str, str] = {}
    for f in middle[1:]:
        nodeid, st = f.get("nodeid"), f.get("status")
        if f.get("kind") != "result" or not isinstance(nodeid, str) or not isinstance(st, str):
            raise FrameError("verdict_integrity_sequence")
        if nodeid in status:
            raise FrameError("verdict_integrity_duplicate_result")
        status[nodeid] = st
    if list(status) != collected or len(set(collected)) != len(collected):
        raise FrameError("verdict_integrity_incomplete")
    finish = frames[-1]
    implied = 0 if status and all(v == PASSING for v in status.values()) else 1
    if finish.get("exit") != implied or finish.get("n_results") != len(status):
        raise FrameError("verdict_integrity_finish")
    if returncode != implied:
        raise FrameError("verdict_integrity_exit_code")
    return status


def judge_frames(stdout: str, *, key: bytes, nonce: str, returncode: int,
                 fail_to_pass: Sequence[str], pass_to_pass: Sequence[str],
                 canary_ok: str, canary_fail: str) -> tuple[bool, str, dict[str, str]]:
    """(passed, reason, {requested id: status}) decided only from authentic frames."""
    ids = list(dict.fromkeys([*fail_to_pass, *pass_to_pass]))
    try:
        status = results_from_frames(parse_frames(stdout, key=key, nonce=nonce), returncode=returncode)
    except FrameError as exc:
        return False, str(exc), {t: "UNVERIFIED" for t in ids}
    tests = {t: status.get(t, "MISSING") for t in ids}
    if canary_ok not in status and canary_fail not in status:
        return False, "collection_error", tests
    if status.get(canary_ok) != PASSING or status.get(canary_fail) != "FAILED":
        return False, "verdict_integrity_canary", tests
    if not ids:
        return False, "no_tests", tests
    if all(v == PASSING for v in tests.values()):
        return True, "pass", tests
    return False, "tests_failed", tests


def grade_isolated_v2(
    backend: ig.SandboxBackend,
    attestation: ig.Attestation | None,
    *,
    workspace_root: Path,
    source_tree: Path,
    target_file: str,
    buggy_text: str,
    candidate_text: str,
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    import_roots: Sequence[str] | None = None,
    timeout: float = 180,
    static_scan: bool = True,
    on_workspace: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Grade one candidate in a fresh sandbox; the parent judges authenticated frames.

    ``import_roots`` are sandbox paths prepended to ``sys.path`` (default: the
    backend's). ``static_scan=False`` exists only for containment and
    verdict-integrity audits; grading for rewards keeps the default.
    """
    t0 = time.time()
    source = runner_source()
    rec: dict[str, Any] = {
        "schema": SCHEMA, "verdict_channel": VERDICT_CHANNEL, "backend": backend.name,
        "config_digest": backend.config_digest(),
        "runner_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_text.encode()).hexdigest(),
        "passed": False, "executed": False,
    }

    def done(reason: str, **extra: Any) -> dict[str, Any]:
        rec.update(reason=reason, seconds=round(time.time() - t0, 3), **extra)
        return rec

    if attestation is None or not attestation.ok:
        return done("isolation_unverified")
    if attestation.config_digest != rec["config_digest"] or attestation.backend != backend.name:
        return done("isolation_unverified_config_changed")
    flags = static_flags_v2(buggy_text, candidate_text)
    rec["static_flags"] = flags
    rec["static_scan"] = bool(static_scan)
    if flags and static_scan:
        return done("quarantined_static_flags")
    rel = Path(target_file)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        return done("unsafe_target_path")
    roots = list(import_roots if import_roots is not None else getattr(backend, "import_roots", ()))
    workspace = Path(tempfile.mkdtemp(prefix="grade-", dir=workspace_root))
    try:
        ig.stage_tree(source_tree, workspace)
        (workspace / rel).write_text(candidate_text, encoding="utf-8")
        nonce = secrets.token_hex(8)
        canary_file = f"test_zz_{nonce}.py"
        canary_ok, canary_fail = f"test_a_{nonce}", f"test_b_{nonce}"
        (workspace / canary_file).write_text(ig._canary_source(canary_ok, canary_fail))
        if on_workspace is not None:
            on_workspace(workspace)
        ids = list(dict.fromkeys([*fail_to_pass, *pass_to_pass]))
        ok_id, fail_id = f"{canary_file}::{canary_ok}", f"{canary_file}::{canary_fail}"
        spec = {"nonce": nonce, "import_roots": roots, "test_ids": [*ids, ok_id, fail_id]}
        key = secrets.token_bytes(32)
        cmd = [backend.python(), "-I", "-c", source, json.dumps(spec)]
        rec["executed"] = True
        try:
            r = backend.run(workspace, cmd, timeout=timeout, stdin=(key.hex() + "\n").encode())
        except subprocess.TimeoutExpired:
            return done("test_timeout", tests={t: "TIMEOUT" for t in ids})
        except ig.IsolationUnavailable as exc:
            return done("isolation_unverified", detail=str(exc)[:300])
        passed, reason, tests = judge_frames(
            r.stdout, key=key, nonce=nonce, returncode=r.returncode, fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass, canary_ok=ok_id, canary_fail=fail_id)
        rec["passed"] = passed
        return done(reason, process_exit=r.returncode, tests=tests,
                    stdout_sha256=hashlib.sha256(r.stdout.encode()).hexdigest(),
                    stderr_tail=r.stderr[-400:] if not passed else "")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
