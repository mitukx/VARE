"""Durable, bounded mission journal and safe next-action planner.

This module records research state; it does not launch models, graders, cloud
jobs, or background schedulers. ``resume`` reconstructs the next admissible
action for an explicit subsequent invocation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator


PHASES = (
    "DISCOVER",
    "SCREEN",
    "QUALIFY",
    "FREEZE",
    "EXECUTE",
    "VERIFY",
    "ANALYZE",
    "DECIDE",
    "INTEGRATE",
    "NEXT_CYCLE",
)
PHASE_NEXT = dict(zip(PHASES, (*PHASES[1:], "DISCOVER"), strict=True))
FAILURE_CLASSES = {"A", "B", "C", "D", "E"}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
DEFAULT_BUDGET = {
    "memory_bytes": 8 * 1024**3,
    "disk_bytes": 20 * 1024**3,
    "model_tokens": 0,
    "wall_seconds": 3600,
    "candidate_attempts": 3,
}
RESOURCE_KEYS = tuple(DEFAULT_BUDGET)
MAX_ACTION_ATTEMPTS = 3
MAX_EVENTS = 100_000
MAX_TASKS = 10_000
MAX_ACTIONS = 10_000
MAX_DB_BYTES = 64 * 1024**2
MAX_EVENT_BYTES = 64 * 1024
MAX_EVIDENCE_BYTES = 512 * 1024**2
GENESIS = "0" * 64


class MissionError(ValueError):
    """The mission state or requested transition is invalid."""


class SourceMismatchError(MissionError):
    """The checkout no longer matches the mission's immutable source identity."""


class LeaseError(MissionError):
    """A stale or foreign action lease cannot publish a result."""


class IntegrityError(MissionError):
    """The append-only mission event chain or database is inconsistent."""


@dataclass(frozen=True, slots=True)
class FailureDecision:
    retry_same_operation: bool
    action_status: str
    next_action: str
    consumed_outcome: bool
    retry_after_seconds: int
    escalation_required: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_failure(
    failure_class: str,
    *,
    attempt_number: int,
    max_attempts: int = MAX_ACTION_ATTEMPTS,
    same_root_cause_streak: int = 1,
    model_exposure_started: bool = False,
    operation_idempotent: bool = False,
) -> FailureDecision:
    """Apply the bounded A–E recovery policy without altering experiment criteria."""
    if failure_class not in FAILURE_CLASSES:
        raise MissionError("failure class must be one of A, B, C, D, or E")
    if type(attempt_number) is not int or attempt_number < 1:
        raise MissionError("attempt_number must be a positive integer")
    if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_ACTION_ATTEMPTS:
        raise MissionError(f"max_attempts must be in 1..{MAX_ACTION_ATTEMPTS}")
    if type(same_root_cause_streak) is not int or same_root_cause_streak < 1:
        raise MissionError("same_root_cause_streak must be a positive integer")
    if type(model_exposure_started) is not bool:
        raise MissionError("model_exposure_started must be boolean")
    if type(operation_idempotent) is not bool:
        raise MissionError("operation_idempotent must be boolean")

    if failure_class == "A" and not model_exposure_started:
        if not operation_idempotent:
            return FailureDecision(
                False,
                "uncertain",
                "reconcile_external_state_before_retry",
                False,
                0,
                True,
            )
        if same_root_cause_streak >= 3:
            return FailureDecision(
                False, "blocked", "redesign_or_change_direction", False, 0, True
            )
        if attempt_number < max_attempts:
            delay = min(30 * (2 ** (attempt_number - 1)), 300)
            return FailureDecision(
                True, "queued", "retry_idempotent_operation", False, delay, False
            )
        return FailureDecision(
            False, "failed", "escalate_after_retry_exhaustion", False, 0, True
        )

    if failure_class == "A":
        return FailureDecision(
            False, "consumed", "continue_on_new_task_family", True, 0, False
        )
    if failure_class == "B":
        return FailureDecision(
            False,
            "superseded",
            "new_implementation_version_and_regressions",
            False,
            0,
            False,
        )
    if failure_class == "C":
        return FailureDecision(
            False, "consumed", "new_protocol_and_unconsumed_task", True, 0, False
        )
    if failure_class == "D":
        return FailureDecision(
            False,
            "consumed",
            "failure_analysis_then_fresh_development_cohort",
            True,
            0,
            False,
        )
    return FailureDecision(
        False, "blocked", "schedule_independent_admissible_work", False, 0, False
    )


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(raw: bytes | str) -> str:
    data = raw.encode("utf-8") if isinstance(raw, str) else raw
    return hashlib.sha256(data).hexdigest()


def _require_sha(value: str, field: str) -> None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise MissionError(f"{field} must be a lowercase SHA-256 digest")


def _identity(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise MissionError(
            f"{field} must be a nonempty identifier of at most 256 characters"
        )


def _resource_map(raw: dict[str, Any], *, field: str) -> dict[str, int]:
    if not isinstance(raw, dict) or set(raw) - set(RESOURCE_KEYS):
        raise MissionError(f"{field} contains unsupported resource keys")
    result = {key: raw.get(key, 0) for key in RESOURCE_KEYS}
    for key, value in result.items():
        if type(value) is not int or value < 0:
            raise MissionError(f"{field}.{key} must be a nonnegative integer")
    return result


def _source_identity(repo: Path) -> tuple[Path, str]:
    root = repo.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise MissionError("repository root must be a directory")
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise MissionError(f"could not establish Git source identity: {exc}") from exc
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise MissionError("Git returned an invalid source revision")
    if status:
        raise MissionError(
            "mission source checkout must be clean; preserve dirty work in a separate experiment"
        )
    return root, revision


def _verified_evidence_ref(
    reference: str, expected_sha256: str, *, source_root: str
) -> str:
    """Resolve an artifact outside the pinned source tree and verify its digest."""
    if not isinstance(reference, str) or not reference.strip() or len(reference) > 1024:
        raise MissionError(
            "evidence reference must be a nonempty path of at most 1024 characters"
        )
    _require_sha(expected_sha256, "evidence_sha256")
    supplied = Path(reference).expanduser()
    if supplied.is_symlink():
        raise MissionError("evidence artifact cannot be a symlink")
    try:
        path = supplied.resolve(strict=True)
        metadata = path.stat()
    except OSError as exc:
        raise MissionError(f"evidence artifact is unavailable: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise MissionError("evidence artifact must be a regular file")
    if metadata.st_size > MAX_EVIDENCE_BYTES:
        raise MissionError("evidence artifact exceeds the 512 MiB verification limit")
    root = Path(source_root).resolve(strict=True)
    if path == root or root in path.parents:
        raise MissionError(
            "evidence artifacts must be outside the pinned source checkout"
        )
    try:
        with path.open("rb") as handle:
            observed = hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        raise MissionError(f"could not hash evidence artifact: {exc}") from exc
    if observed != expected_sha256:
        raise IntegrityError(
            "evidence artifact bytes do not match the recorded SHA-256"
        )
    return str(path)


class MissionJournal:
    """Transactional SQLite journal with event hashing, source checks, and fenced leases."""

    def __init__(self, path: str | Path):
        original = Path(path).expanduser()
        if original.is_symlink():
            raise MissionError("linked mission database is not accepted")
        if not original.is_file():
            raise MissionError("mission database does not exist")
        self.path = original.resolve(strict=True)

    @staticmethod
    def create(
        path: str | Path,
        *,
        mission_id: str,
        repo_root: str | Path,
        budget: dict[str, Any] | None = None,
    ) -> "MissionJournal":
        _identity(mission_id, "mission_id")
        root, revision = _source_identity(Path(repo_root))
        target = Path(path).expanduser()
        if target.is_symlink():
            raise MissionError("linked mission database is not accepted")
        target.parent.mkdir(parents=True, exist_ok=True)
        target = target.parent.resolve() / target.name
        if target == root or root in target.parents:
            raise MissionError(
                "mission state database must be outside the trusted source checkout"
            )
        limits = _resource_map(
            budget if budget is not None else DEFAULT_BUDGET, field="budget"
        )
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.init-", suffix=".sqlite3", dir=target.parent
        )
        os.fchmod(fd, 0o600)
        os.close(fd)
        temporary = Path(temporary_name)
        published = False
        try:
            db = sqlite3.connect(str(temporary), isolation_level=None, timeout=5)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript("""
                CREATE TABLE missions (
                    mission_id TEXT PRIMARY KEY,
                    repo_root TEXT NOT NULL,
                    source_revision TEXT NOT NULL,
                    runtime_json TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    status TEXT NOT NULL,
                    budget_json TEXT NOT NULL,
                    spent_json TEXT NOT NULL,
                    event_seq INTEGER NOT NULL DEFAULT -1,
                    event_head TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE tasks (
                    mission_id TEXT NOT NULL REFERENCES missions(mission_id),
                    task_id TEXT NOT NULL,
                    family_id TEXT NOT NULL,
                    split TEXT NOT NULL,
                    eligibility TEXT NOT NULL,
                    exposure TEXT NOT NULL,
                    source_identity TEXT NOT NULL,
                    grader_sha256 TEXT,
                    certificate_sha256 TEXT,
                    exposure_audit_sha256 TEXT,
                    record_ref TEXT,
                    record_sha256 TEXT,
                    PRIMARY KEY (mission_id, task_id),
                    UNIQUE (mission_id, family_id)
                );
                CREATE TABLE actions (
                    mission_id TEXT NOT NULL REFERENCES missions(mission_id),
                    action_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    retry_safe INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    task_id TEXT,
                    protocol_sha256 TEXT,
                    provenance_sha256 TEXT,
                    identity_sha256 TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    retry_after REAL NOT NULL DEFAULT 0,
                    lease_owner TEXT,
                    lease_generation INTEGER NOT NULL DEFAULT 0,
                    lease_expires REAL,
                    model_started INTEGER NOT NULL DEFAULT 0,
                    result_sha256 TEXT,
                    result_ref TEXT,
                    recommendation TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (mission_id, action_id),
                    UNIQUE (mission_id, idempotency_key),
                    FOREIGN KEY (mission_id, task_id) REFERENCES tasks(mission_id, task_id)
                );
                CREATE UNIQUE INDEX one_model_attempt_per_task
                    ON actions(mission_id, task_id) WHERE kind = 'model_attempt';
                CREATE TABLE attempts (
                    mission_id TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    lease_generation INTEGER NOT NULL,
                    owner TEXT NOT NULL,
                    status TEXT NOT NULL,
                    failure_class TEXT,
                    root_cause TEXT,
                    detail_sha256 TEXT,
                    detail_text TEXT NOT NULL DEFAULT '',
                    detail_ref TEXT,
                    result_sha256 TEXT,
                    result_ref TEXT,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    PRIMARY KEY (mission_id, action_id, ordinal),
                    FOREIGN KEY (mission_id, action_id) REFERENCES actions(mission_id, action_id)
                );
                CREATE TABLE events (
                    mission_id TEXT NOT NULL REFERENCES missions(mission_id),
                    sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    event_sha256 TEXT NOT NULL,
                    PRIMARY KEY (mission_id, sequence)
                );
            """)
            now = time.time()
            runtime = {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "sqlite": sqlite3.sqlite_version,
            }
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO missions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    mission_id,
                    str(root),
                    revision,
                    _canonical(runtime),
                    "DISCOVER",
                    "active",
                    _canonical(limits),
                    _canonical({key: 0 for key in RESOURCE_KEYS}),
                    -1,
                    GENESIS,
                    now,
                    now,
                ),
            )
            MissionJournal._event(
                db,
                mission_id,
                "mission_created",
                {
                    "repo_root": str(root),
                    "source_revision": revision,
                    "runtime": runtime,
                    "budget": limits,
                },
            )
            db.execute("COMMIT")
            db.close()
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise MissionError(
                    "mission database already exists; refusing to overwrite frozen state"
                ) from exc
            published = True
            temporary.unlink()
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            try:
                db.close()
            except (UnboundLocalError, sqlite3.Error):
                pass
            if not published:
                for suffix in ("", "-wal", "-shm", "-journal"):
                    Path(str(temporary) + suffix).unlink(missing_ok=True)
            raise
        return MissionJournal(target)

    @contextmanager
    def _connection(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(str(self.path), isolation_level=None, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    @staticmethod
    def _mission(db: sqlite3.Connection, mission_id: str) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM missions WHERE mission_id=?", (mission_id,)
        ).fetchone()
        if row is None:
            raise MissionError("unknown mission id")
        return row

    @staticmethod
    def _check_source(mission: sqlite3.Row) -> None:
        root, revision = _source_identity(Path(mission["repo_root"]))
        if str(root) != mission["repo_root"] or revision != mission["source_revision"]:
            raise SourceMismatchError(
                "mission checkout changed; create a new mission for the new source identity"
            )

    @staticmethod
    def _assert_chain(db: sqlite3.Connection, mission: sqlite3.Row) -> None:
        previous = GENESIS
        rows = db.execute(
            "SELECT sequence,event_json,event_sha256 FROM events WHERE mission_id=? ORDER BY sequence",
            (mission["mission_id"],),
        ).fetchall()
        for index, row in enumerate(rows):
            try:
                event = json.loads(row["event_json"])
            except json.JSONDecodeError as exc:
                raise IntegrityError("malformed mission event JSON") from exc
            if (
                event.get("sequence") != index
                or event.get("previous_sha256") != previous
            ):
                raise IntegrityError("mission event sequence or predecessor mismatch")
            observed = _digest(_canonical(event))
            if observed != row["event_sha256"]:
                raise IntegrityError("mission event hash mismatch")
            previous = observed
        if len(rows) - 1 != mission["event_seq"] or previous != mission["event_head"]:
            raise IntegrityError("mission event head does not match the retained chain")
        if rows:
            final_event = json.loads(rows[-1]["event_json"])
            if final_event.get("state_sha256") != MissionJournal._state_sha256(
                db, mission["mission_id"]
            ):
                raise IntegrityError(
                    "current mission state does not match its last authenticated event"
                )

    @staticmethod
    def _assert_artifacts(db: sqlite3.Connection, mission: sqlite3.Row) -> None:
        task_records = db.execute(
            "SELECT record_ref,record_sha256 FROM tasks "
            "WHERE mission_id=? AND record_ref IS NOT NULL",
            (mission["mission_id"],),
        ).fetchall()
        for artifact in task_records:
            _verified_evidence_ref(
                artifact["record_ref"],
                artifact["record_sha256"],
                source_root=mission["repo_root"],
            )
        artifacts = db.execute(
            "SELECT result_ref,result_sha256 FROM actions "
            "WHERE mission_id=? AND status='completed'",
            (mission["mission_id"],),
        ).fetchall()
        for artifact in artifacts:
            if artifact["result_ref"] is None:
                raise IntegrityError("completed action has no retained result artifact")
            _verified_evidence_ref(
                artifact["result_ref"],
                artifact["result_sha256"],
                source_root=mission["repo_root"],
            )
        details = db.execute(
            "SELECT detail_ref,detail_sha256 FROM attempts "
            "WHERE mission_id=? AND detail_ref IS NOT NULL",
            (mission["mission_id"],),
        ).fetchall()
        for artifact in details:
            _verified_evidence_ref(
                artifact["detail_ref"],
                artifact["detail_sha256"],
                source_root=mission["repo_root"],
            )
        decisions = db.execute(
            "SELECT event_json FROM events WHERE mission_id=? ORDER BY sequence",
            (mission["mission_id"],),
        ).fetchall()
        for row in decisions:
            event = json.loads(row["event_json"])
            if event.get("kind") == "phase_advanced":
                payload = event.get("payload", {})
                _verified_evidence_ref(
                    payload.get("decision_ref"),
                    payload.get("decision_sha256"),
                    source_root=mission["repo_root"],
                )

    @staticmethod
    def _state_sha256(db: sqlite3.Connection, mission_id: str) -> str:
        mission = MissionJournal._mission(db, mission_id)
        tasks = db.execute(
            "SELECT task_id,family_id,split,eligibility,exposure,source_identity,grader_sha256,"
            "certificate_sha256,exposure_audit_sha256,record_ref,record_sha256 "
            "FROM tasks WHERE mission_id=? ORDER BY task_id",
            (mission_id,),
        ).fetchall()
        actions = db.execute(
            "SELECT action_id,idempotency_key,kind,retry_safe,phase,task_id,protocol_sha256,provenance_sha256,"
            "identity_sha256,request_json,status,attempts,max_attempts,retry_after,lease_owner,"
            "lease_generation,lease_expires,model_started,result_sha256,result_ref,recommendation "
            "FROM actions WHERE mission_id=? ORDER BY action_id",
            (mission_id,),
        ).fetchall()
        attempts = db.execute(
            "SELECT action_id,ordinal,lease_generation,owner,status,failure_class,root_cause,detail_sha256,detail_text,"
            "detail_ref,result_sha256,result_ref,started_at,ended_at FROM attempts WHERE mission_id=? ORDER BY action_id,ordinal",
            (mission_id,),
        ).fetchall()
        state = {
            "mission": {
                "mission_id": mission_id,
                "repo_root": mission["repo_root"],
                "source_revision": mission["source_revision"],
                "runtime_json": mission["runtime_json"],
                "phase": mission["phase"],
                "status": mission["status"],
                "budget_json": mission["budget_json"],
                "spent_json": mission["spent_json"],
            },
            "tasks": [dict(row) for row in tasks],
            "actions": [dict(row) for row in actions],
            "attempts": [dict(row) for row in attempts],
        }
        return _digest(_canonical(state))

    @staticmethod
    def _event(
        db: sqlite3.Connection, mission_id: str, kind: str, payload: dict[str, Any]
    ) -> None:
        mission = MissionJournal._mission(db, mission_id)
        sequence = mission["event_seq"] + 1
        if sequence >= MAX_EVENTS:
            raise MissionError(
                "mission event limit reached; preserve this journal and start a new cycle"
            )
        event = {
            "mission_id": mission_id,
            "sequence": sequence,
            "kind": kind,
            "created_at": time.time(),
            "payload": payload,
            "state_sha256": MissionJournal._state_sha256(db, mission_id),
            "previous_sha256": mission["event_head"],
        }
        event_json = _canonical(event)
        event_size = len(event_json.encode("utf-8"))
        if event_size > MAX_EVENT_BYTES:
            raise MissionError("mission event exceeds the immutable event-size limit")
        page_count = db.execute("PRAGMA page_count").fetchone()[0]
        page_size = db.execute("PRAGMA page_size").fetchone()[0]
        if page_count * page_size + event_size > MAX_DB_BYTES:
            raise MissionError(
                "mission journal storage limit reached; no records were pruned"
            )
        event_hash = _digest(event_json)
        db.execute(
            "INSERT INTO events VALUES(?,?,?,?)",
            (mission_id, sequence, event_json, event_hash),
        )
        db.execute(
            "UPDATE missions SET event_seq=?,event_head=?,updated_at=? WHERE mission_id=?",
            (sequence, event_hash, event["created_at"], mission_id),
        )

    def register_task(
        self,
        mission_id: str,
        *,
        task_id: str,
        family_id: str,
        split: str,
        eligibility: str,
        exposure: str,
        source_identity: str,
        grader_sha256: str | None = None,
        certificate_sha256: str | None = None,
        exposure_audit_sha256: str | None = None,
        record_ref: str | None = None,
        record_sha256: str | None = None,
    ) -> None:
        for value, field in (
            (task_id, "task_id"),
            (family_id, "family_id"),
            (source_identity, "source_identity"),
        ):
            _identity(value, field)
        if split not in {"train", "development", "heldout"}:
            raise MissionError("split must be train, development, or heldout")
        if eligibility not in {"candidate", "qualified", "eligible", "rejected"}:
            raise MissionError("unknown task eligibility status")
        if exposure not in {"unseen", "unknown", "exposed", "consumed"}:
            raise MissionError("unknown task exposure state")
        for value, field in (
            (grader_sha256, "grader_sha256"),
            (certificate_sha256, "certificate_sha256"),
            (exposure_audit_sha256, "exposure_audit_sha256"),
        ):
            if value is not None:
                _require_sha(value, field)
        if record_ref is None and record_sha256 is not None:
            raise MissionError("record_sha256 requires a task record_ref")
        if record_ref is not None:
            if not record_ref.strip() or len(record_ref) > 1024:
                raise MissionError(
                    "record_ref must be a nonempty evidence reference of at most 1024 characters"
                )
            _require_sha(record_sha256, "record_sha256")
        if eligibility in {"qualified", "eligible"} and (
            not grader_sha256 or not certificate_sha256
        ):
            raise MissionError(
                "qualified task requires pinned grader and qualification certificate digests"
            )
        if eligibility == "eligible" and (
            exposure != "unseen" or not exposure_audit_sha256
        ):
            raise MissionError("model-eligible task requires an unseen exposure audit")
        if exposure == "unseen" and not exposure_audit_sha256:
            raise MissionError(
                "unseen exposure status requires an exposure-audit digest"
            )
        record = (
            mission_id,
            task_id,
            family_id,
            split,
            eligibility,
            exposure,
            source_identity,
            grader_sha256,
            certificate_sha256,
            exposure_audit_sha256,
            record_ref,
            record_sha256,
        )
        with self._connection(write=True) as db:
            mission = self._mission(db, mission_id)
            self._check_source(mission)
            self._assert_chain(db, mission)
            self._assert_artifacts(db, mission)
            if record_ref is not None:
                record_ref = _verified_evidence_ref(
                    record_ref,
                    record_sha256,
                    source_root=mission["repo_root"],
                )
                record = (*record[:-2], record_ref, record_sha256)
            old = db.execute(
                "SELECT * FROM tasks WHERE mission_id=? AND task_id=?",
                (mission_id, task_id),
            ).fetchone()
            if old is not None:
                old_tuple = tuple(
                    old[key]
                    for key in (
                        "mission_id",
                        "task_id",
                        "family_id",
                        "split",
                        "eligibility",
                        "exposure",
                        "source_identity",
                        "grader_sha256",
                        "certificate_sha256",
                        "exposure_audit_sha256",
                        "record_ref",
                        "record_sha256",
                    )
                )
                if old_tuple != record:
                    raise MissionError(
                        "task identity is immutable; conflicting registration rejected"
                    )
                return
            if (
                db.execute(
                    "SELECT COUNT(*) FROM tasks WHERE mission_id=?", (mission_id,)
                ).fetchone()[0]
                >= MAX_TASKS
            ):
                raise MissionError(
                    "mission task limit reached; no prior records were pruned"
                )
            db.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", record)
            self._event(
                db,
                mission_id,
                "task_registered",
                {
                    "task_id": task_id,
                    "family_id": family_id,
                    "split": split,
                    "eligibility": eligibility,
                    "exposure": exposure,
                    "source_identity": source_identity,
                    "grader_sha256": grader_sha256,
                    "certificate_sha256": certificate_sha256,
                    "exposure_audit_sha256": exposure_audit_sha256,
                    "record_ref": record_ref,
                    "record_sha256": record_sha256,
                },
            )

    def add_action(
        self,
        mission_id: str,
        *,
        action_id: str,
        idempotency_key: str,
        kind: str,
        phase: str,
        retry_safe: bool = False,
        request: dict[str, Any] | None = None,
        task_id: str | None = None,
        protocol_sha256: str | None = None,
        provenance_sha256: str | None = None,
        max_attempts: int = MAX_ACTION_ATTEMPTS,
    ) -> dict[str, Any]:
        for value, field in (
            (action_id, "action_id"),
            (idempotency_key, "idempotency_key"),
            (kind, "kind"),
        ):
            _identity(value, field)
        if phase not in PHASES:
            raise MissionError("unknown research phase")
        if type(retry_safe) is not bool:
            raise MissionError("retry_safe must be boolean; retry is denied by default")
        if (
            type(max_attempts) is not int
            or not 1 <= max_attempts <= MAX_ACTION_ATTEMPTS
        ):
            raise MissionError(f"max_attempts must be in 1..{MAX_ACTION_ATTEMPTS}")
        if protocol_sha256 is not None:
            _require_sha(protocol_sha256, "protocol_sha256")
        if provenance_sha256 is not None:
            _require_sha(provenance_sha256, "provenance_sha256")
        requested = _resource_map(request or {}, field="request")
        if requested["candidate_attempts"]:
            raise MissionError(
                "candidate_attempts are charged at the model exposure boundary"
            )
        if kind in {"model_attempt", "training", "heldout_evaluation"} and (
            protocol_sha256 is None or provenance_sha256 is None
        ):
            raise MissionError(
                f"{kind} requires frozen protocol and model/verifier/environment provenance digests"
            )
        if kind == "training":
            raise MissionError(
                "training is disabled in mission-journal v1; no authenticated data-readiness adapter is configured"
            )
        if kind == "model_attempt" and task_id is None:
            raise MissionError("model_attempt requires one qualified Development task")
        if kind == "model_attempt" and phase != "EXECUTE":
            raise MissionError("model_attempt must be queued for the EXECUTE phase")
        if kind == "heldout_evaluation" and task_id is not None:
            raise MissionError(
                "heldout_evaluation is a cohort action and cannot name one task"
            )
        with self._connection(write=True) as db:
            mission = self._mission(db, mission_id)
            self._check_source(mission)
            self._assert_chain(db, mission)
            self._assert_artifacts(db, mission)
            existing = db.execute(
                "SELECT * FROM actions WHERE mission_id=? AND idempotency_key=?",
                (mission_id, idempotency_key),
            ).fetchone()
            action_identity = {
                "action_id": action_id,
                "kind": kind,
                "retry_safe": retry_safe,
                "phase": phase,
                "task_id": task_id,
                "protocol_sha256": protocol_sha256,
                "provenance_sha256": provenance_sha256,
                "request": requested,
                "max_attempts": max_attempts,
            }
            identity_hash = _digest(_canonical(action_identity))
            if existing is not None:
                if (
                    existing["action_id"] != action_id
                    or existing["identity_sha256"] != identity_hash
                ):
                    raise MissionError(
                        "idempotency key was reused for a different action"
                    )
                return self._action_dict(existing)

            if task_id is not None:
                task = db.execute(
                    "SELECT * FROM tasks WHERE mission_id=? AND task_id=?",
                    (mission_id, task_id),
                ).fetchone()
                if task is None:
                    raise MissionError("action references an unknown task")
                if (
                    kind == "model_attempt"
                    and db.execute(
                        "SELECT 1 FROM actions WHERE mission_id=? AND kind='model_attempt' AND task_id=?",
                        (mission_id, task_id),
                    ).fetchone()
                ):
                    raise MissionError(
                        "this task family already has a model attempt; select a fresh task family"
                    )
                if kind == "model_attempt" and not (
                    task["split"] == "development"
                    and task["eligibility"] == "eligible"
                    and task["exposure"] == "unseen"
                ):
                    raise MissionError(
                        "model inference requires an unseen, independently eligible Development task"
                    )
            if kind == "heldout_evaluation":
                rows = db.execute(
                    "SELECT COUNT(*) FROM tasks WHERE mission_id=? AND split='heldout' "
                    "AND eligibility='eligible' AND exposure='unseen'",
                    (mission_id,),
                ).fetchone()[0]
                if rows < 15:
                    raise MissionError(
                        "held-out evaluation is closed until 15 unseen eligible task families are registered"
                    )
                raise MissionError(
                    "heldout_evaluation is disabled in mission-journal v1; no single-use evaluation exposure adapter is configured"
                )
            limits = json.loads(mission["budget_json"])
            for key in RESOURCE_KEYS:
                if requested[key] > limits[key]:
                    raise MissionError(
                        f"action reservation exceeds mission budget for {key}"
                    )
            if (
                db.execute(
                    "SELECT COUNT(*) FROM actions WHERE mission_id=?", (mission_id,)
                ).fetchone()[0]
                >= MAX_ACTIONS
            ):
                raise MissionError(
                    "mission action limit reached; no prior records were pruned"
                )
            now = time.time()
            db.execute(
                "INSERT INTO actions(mission_id,action_id,idempotency_key,kind,phase,task_id,protocol_sha256,provenance_sha256,"
                "retry_safe,identity_sha256,request_json,status,attempts,max_attempts,retry_after,lease_owner,lease_generation,"
                "lease_expires,model_started,result_sha256,recommendation,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,'queued',0,?,0,NULL,0,NULL,0,NULL,NULL,?,?)",
                (
                    mission_id,
                    action_id,
                    idempotency_key,
                    kind,
                    phase,
                    task_id,
                    protocol_sha256,
                    provenance_sha256,
                    int(retry_safe),
                    identity_hash,
                    _canonical(requested),
                    max_attempts,
                    now,
                    now,
                ),
            )
            self._event(
                db,
                mission_id,
                "action_queued",
                {**action_identity, "identity_sha256": identity_hash},
            )
            return self._action_dict(
                db.execute(
                    "SELECT * FROM actions WHERE mission_id=? AND action_id=?",
                    (mission_id, action_id),
                ).fetchone()
            )

    @staticmethod
    def _action_dict(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["request"] = json.loads(result.pop("request_json"))
        result["model_started"] = bool(result["model_started"])
        return result

    def _expire_leases(
        self, db: sqlite3.Connection, mission_id: str, now: float
    ) -> None:
        expired = db.execute(
            "SELECT * FROM actions WHERE mission_id=? AND status='leased' AND lease_expires<=? ORDER BY action_id",
            (mission_id, now),
        ).fetchall()
        for action in expired:
            attempt = db.execute(
                "SELECT * FROM attempts WHERE mission_id=? AND action_id=? AND ordinal=?",
                (mission_id, action["action_id"], action["attempts"]),
            ).fetchone()
            if attempt is None or attempt["status"] != "started":
                raise IntegrityError("expired action lease lacks one live attempt")
            if action["model_started"]:
                decision = classify_failure(
                    "A",
                    attempt_number=action["attempts"],
                    max_attempts=action["max_attempts"],
                    model_exposure_started=True,
                    operation_idempotent=bool(action["retry_safe"]),
                )
                if action["task_id"]:
                    db.execute(
                        "UPDATE tasks SET exposure='consumed' WHERE mission_id=? AND task_id=?",
                        (mission_id, action["task_id"]),
                    )
            else:
                decision = classify_failure(
                    "A",
                    attempt_number=action["attempts"],
                    max_attempts=action["max_attempts"],
                    operation_idempotent=bool(action["retry_safe"]),
                )
            now_text = time.time()
            db.execute(
                "UPDATE attempts SET status='abandoned',failure_class='A',root_cause='lease_expired',ended_at=? "
                "WHERE mission_id=? AND action_id=? AND ordinal=?",
                (now_text, mission_id, action["action_id"], action["attempts"]),
            )
            retry_after = now + decision.retry_after_seconds
            recommendation = decision.next_action
            db.execute(
                "UPDATE actions SET status=?,retry_after=?,lease_owner=NULL,lease_expires=NULL,"
                "recommendation=?,updated_at=? WHERE mission_id=? AND action_id=?",
                (
                    decision.action_status,
                    retry_after,
                    recommendation,
                    now_text,
                    mission_id,
                    action["action_id"],
                ),
            )
            spent = json.loads(MissionJournal._mission(db, mission_id)["spent_json"])
            request = json.loads(action["request_json"])
            limits = json.loads(MissionJournal._mission(db, mission_id)["budget_json"])
            # A lost worker may have used all reserved resources; charge that upper bound.
            charge = {key: request[key] for key in RESOURCE_KEYS}
            charge["candidate_attempts"] = 0
            if any(spent[key] + charge[key] > limits[key] for key in RESOURCE_KEYS):
                raise MissionError(
                    "expired action reservation exceeds mission-wide budget"
                )
            for key in RESOURCE_KEYS:
                spent[key] += charge[key]
            db.execute(
                "UPDATE missions SET spent_json=? WHERE mission_id=?",
                (_canonical(spent), mission_id),
            )
            self._event(
                db,
                mission_id,
                "lease_expired",
                {
                    "action_id": action["action_id"],
                    "attempt": action["attempts"],
                    "model_exposure_started": bool(action["model_started"]),
                    "decision": decision.to_dict(),
                    "charged_resources": charge,
                },
            )

    def claim_next(
        self,
        mission_id: str,
        *,
        owner: str,
        now: float | None = None,
        lease_seconds: int = 120,
    ) -> dict[str, Any] | None:
        _identity(owner, "owner")
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 3600:
            raise MissionError("lease_seconds must be in 1..3600")
        current = time.time() if now is None else now
        with self._connection(write=True) as db:
            mission = self._mission(db, mission_id)
            self._check_source(mission)
            self._assert_chain(db, mission)
            self._assert_artifacts(db, mission)
            if mission["status"] != "active":
                return None
            self._expire_leases(db, mission_id, current)
            mission = self._mission(db, mission_id)
            active = db.execute(
                "SELECT request_json FROM actions WHERE mission_id=? AND status='leased' AND lease_expires>?",
                (mission_id, current),
            ).fetchall()
            used = {key: 0 for key in RESOURCE_KEYS}
            for row in active:
                request = json.loads(row["request_json"])
                for key in RESOURCE_KEYS:
                    used[key] += request[key]
            limits = json.loads(mission["budget_json"])
            spent = json.loads(mission["spent_json"])
            queued = db.execute(
                "SELECT * FROM actions WHERE mission_id=? AND phase=? AND status='queued' AND retry_after<=? "
                "ORDER BY action_id",
                (mission_id, mission["phase"], current),
            ).fetchall()
            for action in queued:
                request = json.loads(action["request_json"])
                if any(
                    spent[key] + used[key] + request[key] > limits[key]
                    for key in RESOURCE_KEYS
                ):
                    continue
                if action["attempts"] >= action["max_attempts"]:
                    db.execute(
                        "UPDATE actions SET status='failed',recommendation='escalate_after_retry_exhaustion' "
                        "WHERE mission_id=? AND action_id=?",
                        (mission_id, action["action_id"]),
                    )
                    self._event(
                        db,
                        mission_id,
                        "retry_exhausted",
                        {"action_id": action["action_id"]},
                    )
                    continue
                generation = action["lease_generation"] + 1
                ordinal = action["attempts"] + 1
                expires = current + lease_seconds
                db.execute(
                    "UPDATE actions SET status='leased',attempts=?,lease_owner=?,lease_generation=?,lease_expires=?,updated_at=? "
                    "WHERE mission_id=? AND action_id=?",
                    (
                        ordinal,
                        owner,
                        generation,
                        expires,
                        current,
                        mission_id,
                        action["action_id"],
                    ),
                )
                db.execute(
                    "INSERT INTO attempts(mission_id,action_id,ordinal,lease_generation,owner,status,started_at) "
                    "VALUES(?,?,?,?,?,'started',?)",
                    (
                        mission_id,
                        action["action_id"],
                        ordinal,
                        generation,
                        owner,
                        current,
                    ),
                )
                self._event(
                    db,
                    mission_id,
                    "action_claimed",
                    {
                        "action_id": action["action_id"],
                        "owner": owner,
                        "generation": generation,
                        "attempt": ordinal,
                        "lease_expires": expires,
                        "resource_reservation": request,
                    },
                )
                result = db.execute(
                    "SELECT * FROM actions WHERE mission_id=? AND action_id=?",
                    (mission_id, action["action_id"]),
                ).fetchone()
                return self._action_dict(result)
            return None

    def _fenced_action(
        self,
        db: sqlite3.Connection,
        mission_id: str,
        action_id: str,
        owner: str,
        generation: int,
        now: float,
    ) -> sqlite3.Row:
        action = db.execute(
            "SELECT * FROM actions WHERE mission_id=? AND action_id=?",
            (mission_id, action_id),
        ).fetchone()
        if action is None:
            raise MissionError("unknown action")
        if (
            action["status"] != "leased"
            or action["lease_owner"] != owner
            or action["lease_generation"] != generation
            or action["lease_expires"] is None
            or action["lease_expires"] <= now
        ):
            raise LeaseError("action lease is expired, foreign, or superseded")
        return action

    def start_model_exposure(
        self,
        mission_id: str,
        action_id: str,
        *,
        owner: str,
        generation: int,
        now: float | None = None,
    ) -> None:
        current = time.time() if now is None else now
        with self._connection(write=True) as db:
            mission = self._mission(db, mission_id)
            self._check_source(mission)
            self._assert_chain(db, mission)
            self._assert_artifacts(db, mission)
            action = self._fenced_action(
                db, mission_id, action_id, owner, generation, current
            )
            if action["kind"] != "model_attempt" or action["model_started"]:
                raise MissionError(
                    "only an unstarted model_attempt may cross the exposure boundary"
                )
            if mission["phase"] != "EXECUTE" or action["phase"] != "EXECUTE":
                raise MissionError(
                    "model exposure is admitted only in the EXECUTE phase"
                )
            task = db.execute(
                "SELECT * FROM tasks WHERE mission_id=? AND task_id=?",
                (mission_id, action["task_id"]),
            ).fetchone()
            if (
                task is None
                or task["split"] != "development"
                or task["eligibility"] != "eligible"
                or task["exposure"] != "unseen"
            ):
                raise MissionError("task eligibility changed before model exposure")
            spent = json.loads(mission["spent_json"])
            budget = json.loads(mission["budget_json"])
            if spent["candidate_attempts"] + 1 > budget["candidate_attempts"]:
                raise MissionError("mission candidate-attempt budget is exhausted")
            spent["candidate_attempts"] += 1
            now_text = time.time()
            db.execute(
                "UPDATE missions SET spent_json=? WHERE mission_id=?",
                (_canonical(spent), mission_id),
            )
            db.execute(
                "UPDATE tasks SET exposure='consumed' WHERE mission_id=? AND task_id=?",
                (mission_id, action["task_id"]),
            )
            db.execute(
                "UPDATE actions SET model_started=1,updated_at=? WHERE mission_id=? AND action_id=?",
                (now_text, mission_id, action_id),
            )
            self._event(
                db,
                mission_id,
                "model_exposure_started",
                {
                    "action_id": action_id,
                    "task_id": action["task_id"],
                    "generation": generation,
                    "protocol_sha256": action["protocol_sha256"],
                },
            )

    def complete(
        self,
        mission_id: str,
        action_id: str,
        *,
        owner: str,
        generation: int,
        result_sha256: str,
        actual_resources: dict[str, Any] | None = None,
        result_ref: str | None = None,
        now: float | None = None,
    ) -> None:
        _require_sha(result_sha256, "result_sha256")
        if actual_resources is None or set(actual_resources) != set(RESOURCE_KEYS):
            raise MissionError(
                "actual_resources must explicitly report every resource key"
            )
        actual = _resource_map(actual_resources, field="actual_resources")
        current = time.time() if now is None else now
        with self._connection(write=True) as db:
            mission = self._mission(db, mission_id)
            self._check_source(mission)
            self._assert_chain(db, mission)
            self._assert_artifacts(db, mission)
            if result_ref is None:
                raise MissionError(
                    "completed action requires a retained result_ref artifact"
                )
            result_ref = _verified_evidence_ref(
                result_ref, result_sha256, source_root=mission["repo_root"]
            )
            prior = db.execute(
                "SELECT * FROM actions WHERE mission_id=? AND action_id=?",
                (mission_id, action_id),
            ).fetchone()
            if prior is None:
                raise MissionError("unknown action")
            if prior["status"] == "completed":
                prior_attempt = db.execute(
                    "SELECT * FROM attempts WHERE mission_id=? AND action_id=? AND lease_generation=?",
                    (mission_id, action_id, generation),
                ).fetchone()
                if (
                    prior_attempt is not None
                    and prior_attempt["owner"] == owner
                    and prior_attempt["status"] == "completed"
                    and prior_attempt["result_sha256"] == result_sha256
                ):
                    completion_event = db.execute(
                        "SELECT event_json FROM events WHERE mission_id=? ORDER BY sequence DESC",
                        (mission_id,),
                    ).fetchall()
                    expected_resources = None
                    for event_row in completion_event:
                        event = json.loads(event_row["event_json"])
                        payload = event.get("payload", {})
                        if (
                            event.get("kind") == "action_completed"
                            and payload.get("action_id") == action_id
                            and payload.get("generation") == generation
                        ):
                            expected_resources = payload.get("actual_resources")
                            break
                    if (
                        result_ref != prior_attempt["result_ref"]
                        or expected_resources != actual
                    ):
                        raise LeaseError(
                            "completed action acknowledgment conflicts on evidence or resource report"
                        )
                    return
                raise LeaseError(
                    "completed action cannot accept a duplicate or conflicting result"
                )
            action = self._fenced_action(
                db, mission_id, action_id, owner, generation, current
            )
            if action["kind"] == "model_attempt" and not action["model_started"]:
                raise MissionError(
                    "model attempt cannot complete before its exposure boundary is journaled"
                )
            request = json.loads(action["request_json"])
            if actual["candidate_attempts"]:
                raise MissionError(
                    "candidate_attempts are charged at exposure admission, not as actual work"
                )
            if any(actual[key] > request[key] for key in RESOURCE_KEYS):
                raise MissionError(
                    "actual resource use exceeds the reserved action budget"
                )
            spent = json.loads(mission["spent_json"])
            budget = json.loads(mission["budget_json"])
            if any(spent[key] + actual[key] > budget[key] for key in RESOURCE_KEYS):
                raise MissionError(
                    "actual resource use exceeds the mission-wide budget"
                )
            for key in RESOURCE_KEYS:
                spent[key] += actual[key]
            now_text = time.time()
            db.execute(
                "UPDATE missions SET spent_json=? WHERE mission_id=?",
                (_canonical(spent), mission_id),
            )
            db.execute(
                "UPDATE actions SET status='completed',result_sha256=?,result_ref=?,lease_owner=NULL,lease_expires=NULL,updated_at=? "
                "WHERE mission_id=? AND action_id=?",
                (result_sha256, result_ref, now_text, mission_id, action_id),
            )
            db.execute(
                "UPDATE attempts SET status='completed',result_sha256=?,result_ref=?,ended_at=? WHERE mission_id=? AND action_id=? AND ordinal=?",
                (
                    result_sha256,
                    result_ref,
                    now_text,
                    mission_id,
                    action_id,
                    action["attempts"],
                ),
            )
            self._event(
                db,
                mission_id,
                "action_completed",
                {
                    "action_id": action_id,
                    "generation": generation,
                    "result_sha256": result_sha256,
                    "result_ref": result_ref,
                    "actual_resources": actual,
                },
            )

    def fail(
        self,
        mission_id: str,
        action_id: str,
        *,
        owner: str,
        generation: int,
        failure_class: str,
        root_cause: str,
        detail: str,
        detail_ref: str | None = None,
        actual_resources: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> FailureDecision:
        _identity(root_cause, "root_cause")
        if not isinstance(detail, str) or len(detail.encode("utf-8")) > 16384:
            raise MissionError("failure detail must be UTF-8 text of at most 16 KiB")
        if actual_resources is None or set(actual_resources) != set(RESOURCE_KEYS):
            raise MissionError(
                "failed action must explicitly report every resource key"
            )
        actual = _resource_map(actual_resources, field="actual_resources")
        detail_sha256 = _digest(detail)
        current = time.time() if now is None else now
        with self._connection(write=True) as db:
            mission = self._mission(db, mission_id)
            self._check_source(mission)
            self._assert_chain(db, mission)
            self._assert_artifacts(db, mission)
            prior = db.execute(
                "SELECT * FROM actions WHERE mission_id=? AND action_id=?",
                (mission_id, action_id),
            ).fetchone()
            if prior is None:
                raise MissionError("unknown action")
            if detail_ref is not None:
                detail_ref = _verified_evidence_ref(
                    detail_ref, detail_sha256, source_root=mission["repo_root"]
                )
            if prior["status"] != "leased":
                prior_attempt = db.execute(
                    "SELECT * FROM attempts WHERE mission_id=? AND action_id=? AND lease_generation=?",
                    (mission_id, action_id, generation),
                ).fetchone()
                if (
                    prior_attempt is not None
                    and prior_attempt["owner"] == owner
                    and prior_attempt["status"] == "failed"
                    and prior_attempt["failure_class"] == failure_class
                    and prior_attempt["root_cause"] == root_cause
                    and prior_attempt["detail_sha256"] == detail_sha256
                    and prior_attempt["detail_text"] == detail
                    and prior_attempt["detail_ref"] == detail_ref
                ):
                    rows = db.execute(
                        "SELECT event_json FROM events WHERE mission_id=? ORDER BY sequence DESC",
                        (mission_id,),
                    ).fetchall()
                    for row in rows:
                        event = json.loads(row["event_json"])
                        payload = event.get("payload", {})
                        if (
                            event.get("kind") == "action_failed"
                            and payload.get("action_id") == action_id
                            and payload.get("generation") == generation
                        ):
                            if payload.get("actual_resources") != actual:
                                raise LeaseError(
                                    "duplicate failure acknowledgment conflicts on resource report"
                                )
                            return FailureDecision(**payload["decision"])
                raise LeaseError("failure report is not for the current action lease")
            action = self._fenced_action(
                db, mission_id, action_id, owner, generation, current
            )
            request = json.loads(action["request_json"])
            if actual["candidate_attempts"] or any(
                actual[key] > request[key] for key in RESOURCE_KEYS
            ):
                raise MissionError(
                    "failed action resource report exceeds its reservation"
                )
            streak_rows = db.execute(
                "SELECT root_cause FROM attempts WHERE mission_id=? AND action_id=? "
                "AND status='failed' ORDER BY ordinal DESC",
                (mission_id, action_id),
            ).fetchall()
            streak = 1
            for row in streak_rows:
                if row["root_cause"] != root_cause:
                    break
                streak += 1
            decision = classify_failure(
                failure_class,
                attempt_number=action["attempts"],
                max_attempts=action["max_attempts"],
                same_root_cause_streak=streak,
                model_exposure_started=bool(action["model_started"]),
                operation_idempotent=bool(action["retry_safe"]),
            )
            now_text = time.time()
            db.execute(
                "UPDATE attempts SET status='failed',failure_class=?,root_cause=?,detail_sha256=?,detail_text=?,detail_ref=?,ended_at=? "
                "WHERE mission_id=? AND action_id=? AND ordinal=?",
                (
                    failure_class,
                    root_cause,
                    detail_sha256,
                    detail,
                    detail_ref,
                    now_text,
                    mission_id,
                    action_id,
                    action["attempts"],
                ),
            )
            retry_after = (
                current + decision.retry_after_seconds
                if decision.retry_same_operation
                else 0
            )
            db.execute(
                "UPDATE actions SET status=?,retry_after=?,lease_owner=NULL,lease_expires=NULL,"
                "recommendation=?,updated_at=? WHERE mission_id=? AND action_id=?",
                (
                    decision.action_status,
                    retry_after,
                    decision.next_action,
                    now_text,
                    mission_id,
                    action_id,
                ),
            )
            spent = json.loads(mission["spent_json"])
            limits = json.loads(mission["budget_json"])
            # After model exposure, actual work may be unknown; charge the reservation.
            charge = (
                {key: request[key] for key in RESOURCE_KEYS}
                if action["model_started"]
                else actual.copy()
            )
            charge["candidate_attempts"] = 0
            if any(spent[key] + charge[key] > limits[key] for key in RESOURCE_KEYS):
                raise MissionError(
                    "failed action resource use exceeds mission-wide budget"
                )
            for key in RESOURCE_KEYS:
                spent[key] += charge[key]
            db.execute(
                "UPDATE missions SET spent_json=? WHERE mission_id=?",
                (_canonical(spent), mission_id),
            )
            self._event(
                db,
                mission_id,
                "action_failed",
                {
                    "action_id": action_id,
                    "generation": generation,
                    "failure_class": failure_class,
                    "root_cause": root_cause,
                    "detail_sha256": detail_sha256,
                    "detail_ref": detail_ref,
                    "actual_resources": actual,
                    "charged_resources": charge,
                    "model_exposure_started": bool(action["model_started"]),
                    "decision": decision.to_dict(),
                },
            )
            return decision

    def advance(
        self,
        mission_id: str,
        *,
        expected_phase: str,
        next_phase: str,
        decision_sha256: str,
        decision_ref: str,
    ) -> None:
        _require_sha(decision_sha256, "decision_sha256")
        with self._connection(write=True) as db:
            mission = self._mission(db, mission_id)
            self._check_source(mission)
            self._assert_chain(db, mission)
            self._assert_artifacts(db, mission)
            decision_ref = _verified_evidence_ref(
                decision_ref,
                decision_sha256,
                source_root=mission["repo_root"],
            )
            if (
                mission["phase"] != expected_phase
                or PHASE_NEXT.get(expected_phase) != next_phase
            ):
                raise MissionError(
                    "phase transition is stale or not in the frozen research state machine"
                )
            pending = db.execute(
                "SELECT COUNT(*) FROM actions WHERE mission_id=? AND phase=? AND status IN ('queued','leased','uncertain')",
                (mission_id, expected_phase),
            ).fetchone()[0]
            if pending:
                raise MissionError(
                    "cannot advance while phase actions remain queued, leased, or uncertain"
                )
            completed = db.execute(
                "SELECT COUNT(*) FROM actions WHERE mission_id=? AND phase=? AND status='completed'",
                (mission_id, expected_phase),
            ).fetchone()[0]
            if not completed:
                raise MissionError(
                    "phase transition requires at least one completed action as evidence"
                )
            db.execute(
                "UPDATE missions SET phase=? WHERE mission_id=?",
                (next_phase, mission_id),
            )
            self._event(
                db,
                mission_id,
                "phase_advanced",
                {
                    "from": expected_phase,
                    "to": next_phase,
                    "decision_sha256": decision_sha256,
                    "decision_ref": decision_ref,
                },
            )

    def audit(self, mission_id: str, *, check_source: bool = True) -> dict[str, Any]:
        with self._connection() as db:
            mission = self._mission(db, mission_id)
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise IntegrityError("SQLite integrity_check failed")
            if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise IntegrityError("SQLite foreign-key integrity check failed")
            if check_source:
                self._check_source(mission)
            self._assert_chain(db, mission)
            self._assert_artifacts(db, mission)
            actions = db.execute(
                "SELECT status,COUNT(*) AS n FROM actions WHERE mission_id=? GROUP BY status ORDER BY status",
                (mission_id,),
            ).fetchall()
            tasks = db.execute(
                "SELECT split,eligibility,exposure,COUNT(*) AS n FROM tasks WHERE mission_id=? "
                "GROUP BY split,eligibility,exposure ORDER BY split,eligibility,exposure",
                (mission_id,),
            ).fetchall()
            return {
                "schema_version": 1,
                "mission_id": mission_id,
                "status": mission["status"],
                "phase": mission["phase"],
                "source": {
                    "repo_root": mission["repo_root"],
                    "revision": mission["source_revision"],
                },
                "runtime": json.loads(mission["runtime_json"]),
                "event_count": mission["event_seq"] + 1,
                "event_head_sha256": mission["event_head"],
                "budget": json.loads(mission["budget_json"]),
                "spent": json.loads(mission["spent_json"]),
                "action_counts": {row["status"]: row["n"] for row in actions},
                "task_counts": [dict(row) for row in tasks],
            }

    def resume(self, mission_id: str, *, now: float | None = None) -> dict[str, Any]:
        current = time.time() if now is None else now
        with self._connection(write=True) as db:
            mission = self._mission(db, mission_id)
            self._check_source(mission)
            self._assert_chain(db, mission)
            self._assert_artifacts(db, mission)
            self._expire_leases(db, mission_id, current)
            mission = self._mission(db, mission_id)
            queued = db.execute(
                "SELECT * FROM actions WHERE mission_id=? AND phase=? AND status='queued' AND retry_after<=? ORDER BY action_id",
                (mission_id, mission["phase"], current),
            ).fetchall()
            limits = json.loads(mission["budget_json"])
            spent = json.loads(mission["spent_json"])
            active = db.execute(
                "SELECT request_json FROM actions WHERE mission_id=? AND status='leased' AND lease_expires>?",
                (mission_id, current),
            ).fetchall()
            reserved = {key: 0 for key in RESOURCE_KEYS}
            for row in active:
                req = json.loads(row["request_json"])
                for key in RESOURCE_KEYS:
                    reserved[key] += req[key]
            next_action = None
            for row in queued:
                req = json.loads(row["request_json"])
                if all(
                    spent[key] + reserved[key] + req[key] <= limits[key]
                    for key in RESOURCE_KEYS
                ):
                    next_action = self._action_dict(row)
                    break
            recommendations = db.execute(
                "SELECT action_id,recommendation,status FROM actions WHERE mission_id=? AND recommendation IS NOT NULL "
                "ORDER BY updated_at DESC,action_id",
                (mission_id,),
            ).fetchall()
            return {
                "schema_version": 1,
                "mission_id": mission_id,
                "phase": mission["phase"],
                "status": mission["status"],
                "next_action": next_action,
                "recommendations": [dict(row) for row in recommendations[:10]],
                "resources": {"budget": limits, "spent": spent, "reserved": reserved},
                "executor": "not_configured; this command plans only and never starts external work",
            }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vare research",
        description="Durable mission-state journal and bounded next-action planner",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser(
        "init", help="create a new mission journal from a clean Git checkout"
    )
    init.add_argument("--db", required=True)
    init.add_argument("--mission-id", required=True)
    init.add_argument("--repo", required=True)
    init.add_argument("--budget-json")
    task = sub.add_parser(
        "task-add", help="register an immutable task family and exposure certificate"
    )
    task.add_argument("--db", required=True)
    task.add_argument("--mission-id", required=True)
    task.add_argument(
        "--record-json", required=True, help="path to JSON task provenance record"
    )
    action = sub.add_parser("action-add", help="queue one hash-bound research action")
    action.add_argument("--db", required=True)
    action.add_argument("--mission-id", required=True)
    action.add_argument(
        "--record-json", required=True, help="path to JSON action identity/reservation"
    )
    resume = sub.add_parser(
        "resume", help="verify source and reconstruct the next safe action"
    )
    resume.add_argument("--db", required=True)
    resume.add_argument("--mission-id", required=True)
    audit = sub.add_parser(
        "audit", help="verify source identity and the append-only journal"
    )
    audit.add_argument("--db", required=True)
    audit.add_argument("--mission-id", required=True)
    claim = sub.add_parser(
        "claim", help="claim the next current-phase action under a fenced lease"
    )
    claim.add_argument("--db", required=True)
    claim.add_argument("--mission-id", required=True)
    claim.add_argument("--owner", required=True)
    claim.add_argument("--lease-seconds", type=int, default=120)
    exposure = sub.add_parser(
        "exposure-start",
        help="durably mark a model task consumed immediately before inference",
    )
    exposure.add_argument("--db", required=True)
    exposure.add_argument("--mission-id", required=True)
    exposure.add_argument("--action-id", required=True)
    exposure.add_argument("--owner", required=True)
    exposure.add_argument("--generation", type=int, required=True)
    complete = sub.add_parser(
        "complete", help="record a completed action with a hash-verified artifact"
    )
    complete.add_argument("--db", required=True)
    complete.add_argument("--mission-id", required=True)
    complete.add_argument("--action-id", required=True)
    complete.add_argument("--owner", required=True)
    complete.add_argument("--generation", type=int, required=True)
    complete.add_argument("--result-sha256", required=True)
    complete.add_argument("--result-ref")
    complete.add_argument("--actual-resources-json", required=True)
    fail = sub.add_parser(
        "fail", help="record a classified, bounded failure for the current lease"
    )
    fail.add_argument("--db", required=True)
    fail.add_argument("--mission-id", required=True)
    fail.add_argument("--action-id", required=True)
    fail.add_argument("--owner", required=True)
    fail.add_argument("--generation", type=int, required=True)
    fail.add_argument(
        "--class", dest="failure_class", choices=sorted(FAILURE_CLASSES), required=True
    )
    fail.add_argument("--root-cause", required=True)
    fail.add_argument("--detail-file", required=True)
    fail.add_argument("--actual-resources-json", required=True)
    advance = sub.add_parser(
        "advance", help="advance one phase with a frozen decision digest"
    )
    advance.add_argument("--db", required=True)
    advance.add_argument("--mission-id", required=True)
    advance.add_argument("--expected-phase", choices=PHASES, required=True)
    advance.add_argument("--next-phase", choices=PHASES, required=True)
    advance.add_argument("--decision-sha256", required=True)
    advance.add_argument("--decision-ref", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            budget = (
                json.loads(Path(args.budget_json).read_text(encoding="utf-8"))
                if args.budget_json
                else None
            )
            journal = MissionJournal.create(
                args.db, mission_id=args.mission_id, repo_root=args.repo, budget=budget
            )
            payload = journal.audit(args.mission_id)
        else:
            journal = MissionJournal(args.db)
            if args.command == "resume":
                payload = journal.resume(args.mission_id)
            elif args.command == "audit":
                payload = journal.audit(args.mission_id)
            elif args.command == "task-add":
                record = json.loads(Path(args.record_json).read_text(encoding="utf-8"))
                journal.register_task(args.mission_id, **record)
                payload = journal.audit(args.mission_id)
            elif args.command == "action-add":
                record = json.loads(Path(args.record_json).read_text(encoding="utf-8"))
                payload = journal.add_action(args.mission_id, **record)
            elif args.command == "claim":
                payload = journal.claim_next(
                    args.mission_id, owner=args.owner, lease_seconds=args.lease_seconds
                )
            elif args.command == "exposure-start":
                journal.start_model_exposure(
                    args.mission_id,
                    args.action_id,
                    owner=args.owner,
                    generation=args.generation,
                )
                payload = journal.audit(args.mission_id)
            elif args.command == "complete":
                actual = json.loads(
                    Path(args.actual_resources_json).read_text(encoding="utf-8")
                )
                journal.complete(
                    args.mission_id,
                    args.action_id,
                    owner=args.owner,
                    generation=args.generation,
                    result_sha256=args.result_sha256,
                    actual_resources=actual,
                    result_ref=args.result_ref,
                )
                payload = journal.audit(args.mission_id)
            elif args.command == "fail":
                detail_path = Path(args.detail_file).expanduser().resolve(strict=True)
                detail = detail_path.read_bytes().decode("utf-8")
                actual = json.loads(
                    Path(args.actual_resources_json).read_text(encoding="utf-8")
                )
                decision = journal.fail(
                    args.mission_id,
                    args.action_id,
                    owner=args.owner,
                    generation=args.generation,
                    failure_class=args.failure_class,
                    root_cause=args.root_cause,
                    detail=detail,
                    detail_ref=str(detail_path),
                    actual_resources=actual,
                )
                payload = decision.to_dict()
            else:
                journal.advance(
                    args.mission_id,
                    expected_phase=args.expected_phase,
                    next_phase=args.next_phase,
                    decision_sha256=args.decision_sha256,
                    decision_ref=args.decision_ref,
                )
                payload = journal.audit(args.mission_id)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (MissionError, OSError, json.JSONDecodeError, sqlite3.Error) as exc:
        print(f"research mission error: {exc}", file=sys.stderr)
        return 2
