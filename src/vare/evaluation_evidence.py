"""Durable, content-addressed records for completed model evaluations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid


EVALUATION_SCHEMA = "vare.evaluation-evidence.v1"
START_SCHEMA = "vare.evaluation-start.v1"
FAILURE_SCHEMA = "vare.evaluation-failure.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROLES = {"baseline", "incumbent", "candidate", "final"}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _validate_evaluation(record: Mapping[str, object]) -> None:
    _require(record.get("schema") == EVALUATION_SCHEMA, "unknown evaluation evidence schema")
    _require(isinstance(record.get("invocation_id"), str)
             and bool(record["invocation_id"]), "missing evaluation invocation id")
    _require(record.get("role") in _ROLES, "invalid evaluation role")
    _require(_valid_timestamp(record.get("started_at_utc"))
             and _valid_timestamp(record.get("completed_at_utc")),
             "evaluation timestamps must be timezone-aware UTC")
    for name in ("run_id", "scenario_id", "arm"):
        _require(isinstance(record.get(name), str) and bool(record[name]),
                 f"missing evaluation {name}")
    for section in ("model", "checkpoint", "verifier", "evaluator", "source_hashes"):
        _require(isinstance(record.get(section), Mapping), f"missing evaluation {section} identity")
    checkpoint = record["checkpoint"]
    _require(isinstance(checkpoint, Mapping)
             and isinstance(checkpoint.get("policy_id"), str)
             and isinstance(checkpoint.get("file_sha256"), str)
             and _SHA256.fullmatch(checkpoint["file_sha256"]) is not None,
             "invalid evaluation checkpoint identity")
    verifier = record["verifier"]
    _require(isinstance(verifier, Mapping)
             and isinstance(verifier.get("name"), str)
             and type(verifier.get("version")) is int
             and isinstance(verifier.get("identity_sha256"), str)
             and _SHA256.fullmatch(verifier["identity_sha256"]) is not None
             and isinstance(verifier.get("definition"), Mapping)
             and type(verifier.get("pass_threshold")) in (int, float)
             and math.isfinite(float(verifier["pass_threshold"])),
             "invalid evaluation verifier identity or decision threshold")
    hashes = record["source_hashes"]
    _require(bool(hashes) and all(
        isinstance(name, str) and isinstance(digest, str) and _SHA256.fullmatch(digest)
        for name, digest in hashes.items()), "invalid evaluation source hashes")
    aggregate = record.get("aggregate")
    _require(isinstance(aggregate, Mapping)
             and type(aggregate.get("primary")) in (int, float)
             and type(aggregate.get("n")) is int
             and math.isfinite(float(aggregate["primary"])),
             "invalid aggregate evaluation score")
    tasks = record.get("tasks")
    _require(isinstance(tasks, list) and bool(tasks), "evaluation has no task results")
    task_ids: set[str] = set()
    for row in tasks:
        _require(isinstance(row, Mapping), "malformed task evaluation result")
        task_id = row.get("task_id")
        _require(isinstance(task_id, str) and task_id not in task_ids,
                 "missing or duplicate evaluated task id")
        task_ids.add(task_id)
        _require(isinstance(row.get("prompt"), str)
                 and isinstance(row.get("output_text"), str)
                 and isinstance(row.get("output_token_ids"), list)
                 and all(type(token) is int for token in row["output_token_ids"]),
                 "task result omits the exact prompt or generated output")
        _require(type(row.get("score")) in (int, float)
                 and math.isfinite(float(row["score"]))
                 and type(row.get("passed")) is bool,
                 "task result has invalid score or pass/fail decision")
        _require(row["passed"] == (float(row["score"])
                                    >= float(record["verifier"]["pass_threshold"])),
                 "task pass/fail decision does not match the frozen threshold")
    _require(aggregate["n"] == len(tasks), "aggregate task count differs from raw results")
    mean_score = math.fsum(float(row["score"]) for row in tasks) / len(tasks)
    _require(math.isclose(float(aggregate["primary"]), mean_score,
                          rel_tol=0.0, abs_tol=1e-12),
             "aggregate score does not reconstruct from task results")
    invocation = record.get("evaluation_invocation")
    _require(isinstance(invocation, Mapping)
             and invocation.get("id") == record.get("invocation_id")
             and invocation.get("started_at_utc") == record.get("started_at_utc")
             and invocation.get("completed_at_utc") == record.get("completed_at_utc"),
             "evaluation invocation metadata differs from record identity")
    _require(datetime.fromisoformat(str(record["completed_at_utc"]))
             >= datetime.fromisoformat(str(record["started_at_utc"])),
             "evaluation completion precedes evaluation start")


def _with_digest(record: Mapping[str, object]) -> dict[str, object]:
    payload = dict(record)
    payload.pop("record_sha256", None)
    payload["record_sha256"] = sha256_bytes(canonical_json(payload))
    return payload


def _verify_digest(record: Mapping[str, object]) -> None:
    digest = record.get("record_sha256")
    _require(isinstance(digest, str) and _SHA256.fullmatch(digest) is not None,
             "record has no valid self-hash")
    payload = dict(record)
    del payload["record_sha256"]
    _require(sha256_bytes(canonical_json(payload)) == digest,
             "record self-hash mismatch")


def _atomic_create(path: Path, payload: bytes) -> None:
    """Atomically create a file without replacing any existing directory entry."""
    _require(not path.exists() and not path.is_symlink(),
             f"evidence destination already exists: {path.name}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class EvaluationEvidenceJournal:
    """Persist each completed evaluation as a separate atomic JSON artifact."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if self.root.exists() or self.root.is_symlink():
            raise FileExistsError(f"evaluation evidence directory already exists: {self.root}")
        self.root.mkdir(parents=True, exist_ok=False)
        self._invocations: list[str] = []
        self._started: dict[str, dict[str, object]] = {}

    @property
    def invocation_ids(self) -> tuple[str, ...]:
        return tuple(self._invocations)

    def record_evaluation_started(self, record: Mapping[str, object]) -> Path:
        payload = dict(record)
        _require(payload.get("schema") == START_SCHEMA, "unknown evaluation start schema")
        invocation_id = payload.get("invocation_id")
        _require(isinstance(invocation_id, str) and bool(invocation_id),
                 "missing evaluation invocation id")
        _require(invocation_id not in self._started, "duplicate started evaluation id")
        _require(_valid_timestamp(payload.get("started_at_utc")),
                 "evaluation start timestamp must be timezone-aware UTC")
        _require(payload.get("role") in _ROLES, "invalid evaluation role")
        for name in ("run_id", "scenario_id", "arm"):
            _require(isinstance(payload.get(name), str) and bool(payload[name]),
                     f"missing evaluation {name}")
        for section in ("model", "checkpoint", "verifier", "evaluator", "source_hashes"):
            _require(isinstance(payload.get(section), Mapping),
                     f"missing evaluation {section} identity")
        prompts = payload.get("tasks")
        _require(isinstance(prompts, list) and bool(prompts)
                 and all(isinstance(row, Mapping)
                         and isinstance(row.get("task_id"), str)
                         and isinstance(row.get("prompt"), str) for row in prompts),
                 "evaluation start does not retain exact task prompts")
        task_ids = [row["task_id"] for row in prompts]
        _require(len(task_ids) == len(set(task_ids)),
                 "evaluation start contains duplicate task IDs")
        sealed = _with_digest(payload)
        encoded = json.dumps(sealed, indent=2, sort_keys=True, ensure_ascii=False,
                             allow_nan=False).encode("utf-8") + b"\n"
        path = self.root / f"started-{invocation_id}.json"
        try:
            _atomic_create(path, encoded)
        except BaseException:
            if path.is_file() and not path.is_symlink() and path.read_bytes() == encoded:
                self._started[invocation_id] = sealed
            raise
        self._started[invocation_id] = sealed
        return path

    def record_evaluation(self, record: Mapping[str, object]) -> Path:
        payload = dict(record)
        _validate_evaluation(payload)
        invocation_id = str(payload["invocation_id"])
        _require(invocation_id not in self._invocations, "duplicate in-memory invocation id")
        _require(invocation_id in self._started,
                 "completed evaluation has no durable start record")
        started = self._started[invocation_id]
        _require(payload["started_at_utc"] == started["started_at_utc"],
                 "evaluation start timestamp differs from durable start record")
        start_tasks = started["tasks"]
        _require(isinstance(start_tasks, list), "malformed evaluation start tasks")
        _require([(row["task_id"], row["prompt"]) for row in start_tasks]
                 == [(row["task_id"], row["prompt"]) for row in payload["tasks"]],
                 "completed evaluation task prompts differ from start record")
        for key in ("role", "run_id", "scenario_id", "arm", "model", "checkpoint",
                    "verifier", "evaluator", "source_hashes", "round"):
            _require(payload.get(key) == started.get(key),
                     f"completed evaluation {key} differs from durable start record")
        path = self.root / f"evaluation-{invocation_id}.json"
        sealed = _with_digest(payload)
        encoded = json.dumps(
            sealed, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
        ).encode("utf-8") + b"\n"
        try:
            _atomic_create(path, encoded)
        except BaseException:
            if path.is_file() and not path.is_symlink() and path.read_bytes() == encoded:
                self._invocations.append(invocation_id)
            raise
        self._invocations.append(invocation_id)
        return path

    def record_failure(self, *, failure_type: str, message: str,
                       occurred_at_utc: str | None = None,
                       stage: str | None = None,
                       run_id: str | None = None,
                       scenario_id: str | None = None,
                       arm: str | None = None) -> Path:
        _require(isinstance(failure_type, str) and bool(failure_type),
                 "failure type is required")
        _require(isinstance(message, str), "failure message must be text")
        timestamp = occurred_at_utc or utc_now()
        _require(_valid_timestamp(timestamp), "failure timestamp must be timezone-aware UTC")
        payload: dict[str, object] = {
            "schema": FAILURE_SCHEMA,
            "failure_id": uuid.uuid4().hex,
            "failure_type": failure_type,
            "message": message,
            "occurred_at_utc": timestamp,
            "completed_invocation_ids": list(self._invocations),
            "started_invocation_ids": sorted(self._started),
            "stage": stage,
            "run_id": run_id,
            "scenario_id": scenario_id,
            "arm": arm,
        }
        sealed = _with_digest(payload)
        encoded = json.dumps(
            sealed, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
        ).encode("utf-8") + b"\n"
        path = self.root / f"failure-{payload['failure_id']}.json"
        _atomic_create(path, encoded)
        return path


def audit_evaluation_evidence(
    root: Path,
    *,
    scorer: Callable[[Mapping[str, object], str, str], float] | None = None,
) -> dict[str, object]:
    """Verify hashes and optionally reconstruct every score with an independent scorer."""
    root = Path(root)
    _require(root.is_dir() and not root.is_symlink(), "unsafe evidence root")
    files = sorted(root.iterdir(), key=lambda item: (0 if item.name.startswith("started-") else 1,
                                                    item.name))
    _require(all(path.is_file() and not path.is_symlink()
                 and path.suffix == ".json" for path in files),
             "evidence root contains an unexpected or transient file")
    invocation_ids: set[str] = set()
    starts: dict[str, Mapping[str, object]] = {}
    failure_rows: list[Mapping[str, object]] = []
    evaluations = failures = 0
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        _require(isinstance(record, dict), "evidence record is not an object")
        _verify_digest(record)
        if record.get("schema") == START_SCHEMA:
            invocation_id = record.get("invocation_id")
            _require(isinstance(invocation_id, str) and invocation_id not in starts,
                     "missing or duplicate started evaluation invocation id")
            _require(path.name == f"started-{invocation_id}.json",
                     "start filename does not bind its invocation id")
            _require(_valid_timestamp(record.get("started_at_utc")),
                     "invalid evaluation start timestamp")
            _require(record.get("role") in _ROLES
                     and isinstance(record.get("tasks"), list),
                     "malformed evaluation start record")
            for name in ("run_id", "scenario_id", "arm"):
                _require(isinstance(record.get(name), str) and bool(record[name]),
                         f"start record missing {name}")
            for section in ("model", "checkpoint", "verifier", "evaluator", "source_hashes"):
                _require(isinstance(record.get(section), Mapping),
                         f"start record missing {section}")
            _require(bool(record["source_hashes"])
                     and all(isinstance(digest, str) and _SHA256.fullmatch(digest)
                             for digest in record["source_hashes"].values()),
                     "invalid source hashes in evaluation start record")
            prompt_rows = record["tasks"]
            _require(bool(prompt_rows) and all(
                isinstance(row, Mapping) and isinstance(row.get("task_id"), str)
                and isinstance(row.get("prompt"), str) for row in prompt_rows),
                "evaluation start prompt list is malformed")
            starts[invocation_id] = record
        elif record.get("schema") == EVALUATION_SCHEMA:
            _validate_evaluation(record)
            invocation_id = str(record["invocation_id"])
            _require(invocation_id not in invocation_ids,
                     "duplicate evaluation invocation id")
            invocation_ids.add(invocation_id)
            _require(path.name == f"evaluation-{invocation_id}.json",
                     "evaluation filename does not bind its invocation id")
            _require(invocation_id in starts,
                     "completed evaluation has no durable start record")
            start_tasks = starts[invocation_id]["tasks"]
            _require(isinstance(start_tasks, list)
                     and [(row.get("task_id"), row.get("prompt")) for row in start_tasks]
                     == [(row.get("task_id"), row.get("prompt")) for row in record["tasks"]],
                     "completed prompts differ from durable start record")
            for key in ("role", "run_id", "scenario_id", "arm", "model", "checkpoint",
                        "verifier", "evaluator", "source_hashes", "round",
                        "started_at_utc"):
                _require(record.get(key) == starts[invocation_id].get(key),
                         f"completed evaluation {key} differs from durable start record")
            if scorer is not None:
                verifier = record["verifier"]
                assert isinstance(verifier, Mapping)
                definition = verifier["definition"]
                threshold = float(verifier["pass_threshold"])
                for row in record["tasks"]:
                    assert isinstance(row, Mapping)
                    recomputed = float(scorer(definition, row["prompt"], row["output_text"]))
                    _require(math.isfinite(recomputed)
                             and math.isclose(recomputed, float(row["score"]),
                                              rel_tol=0.0, abs_tol=1e-12),
                             f"score reconstruction mismatch: {row['task_id']}")
                    _require(bool(row["passed"]) == (recomputed >= threshold),
                             f"pass/fail reconstruction mismatch: {row['task_id']}")
            evaluations += 1
        elif record.get("schema") == FAILURE_SCHEMA:
            _require(_valid_timestamp(record.get("occurred_at_utc")),
                     "invalid failure timestamp")
            _require(isinstance(record.get("failure_type"), str)
                     and isinstance(record.get("message"), str)
                     and isinstance(record.get("completed_invocation_ids"), list)
                     and isinstance(record.get("started_invocation_ids"), list)
                     and (record.get("stage") is None
                          or isinstance(record.get("stage"), str)),
                     "malformed failure record")
            failure_rows.append(record)
            failures += 1
        else:
            raise ValueError("unknown evaluation evidence record schema")
    for failure in failure_rows:
        _require(set(failure["completed_invocation_ids"]) <= invocation_ids,
                 "failure record references an unknown completed evaluation")
        _require(set(failure["started_invocation_ids"]) <= set(starts),
                 "failure record references an unknown started evaluation")
    return {
        "pass": True,
        "evaluation_records": evaluations,
        "started_evaluations": len(starts),
        "incomplete_evaluations": sorted(set(starts) - invocation_ids),
        "failure_records": failures,
        "invocation_ids": sorted(invocation_ids),
        "integrity": "self-hashes verified; scores independently reconstructed" if scorer
        else "self-hashes verified",
    }
