"""Durable resume store for experiment runs.

The store is the authority for which step results were committed. A crash
before commit leaves the step uncommitted so a later resume may execute it
again. A second commit of a different result is rejected. Replaying the same
committed bytes is a duplicate and does not count as a new result.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import sqlite3
import threading
from typing import Any, Iterator

from .jsonutil import canonical_json, digest_bytes, sha256_json, strict_load
from .process_owner import process_identity as process_identity

TERMINAL_STATES = frozenset({"complete", "cancelled", "failed_closed"})
_ZERO = "0" * 64


class StoreError(RuntimeError):
    pass


class FenceError(StoreError):
    pass


class DuplicateCommitError(StoreError):
    pass


class ResumeStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_symlink():
            raise StoreError("linked state database")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            self._create()

    @contextmanager
    def _connection(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        with self._lock:
            db = sqlite3.connect(str(self.path), timeout=5, isolation_level=None, check_same_thread=False)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
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

    def _create(self) -> None:
        db = sqlite3.connect(str(self.path), isolation_level=None)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.executescript(
                """
                CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE runs(
                    run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    spec_sha256 TEXT NOT NULL,
                    config_sha256 TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    protected_fingerprint TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    owner TEXT,
                    owner_pid INTEGER,
                    owner_starttime INTEGER,
                    child_pid INTEGER,
                    child_starttime INTEGER,
                    cancel_requested INTEGER NOT NULL,
                    failure_class TEXT,
                    manifest_json TEXT,
                    manifest_sha256 TEXT,
                    accepted INTEGER,
                    candidate_incorrect INTEGER
                );
                CREATE TABLE steps(
                    run_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt INTEGER,
                    record_json TEXT,
                    record_sha256 TEXT,
                    stdout BLOB,
                    stderr BLOB,
                    PRIMARY KEY(run_id, step_index)
                );
                CREATE TABLE events(
                    sequence INTEGER PRIMARY KEY,
                    body TEXT NOT NULL,
                    event_hash TEXT NOT NULL
                );
                """
            )
            db.execute("INSERT INTO meta VALUES('schema_version', '1')")
        finally:
            db.close()

    def _event(self, db: sqlite3.Connection, run_id: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        last = db.execute("SELECT sequence, event_hash FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
        sequence = 0 if last is None else int(last["sequence"]) + 1
        prev = _ZERO if last is None else str(last["event_hash"])
        body = {"sequence": sequence, "run_id": run_id, "kind": kind, "payload": payload, "prev_hash": prev}
        encoded = canonical_json(body)
        digest = sha256_json(body)
        db.execute("INSERT INTO events(sequence, body, event_hash) VALUES(?,?,?)", (sequence, encoded, digest))
        return {**body, "event_hash": digest}

    def enroll(
        self,
        *,
        run_id: str,
        task_id: str,
        spec_json: str,
        spec_sha256: str,
        config_sha256: str,
        source_fingerprint: dict[str, str],
        protected_fingerprint: dict[str, str],
        step_names: list[str],
    ) -> str:
        with self._connection(True) as db:
            existing = db.execute("SELECT spec_sha256, state FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if existing is not None:
                if existing["spec_sha256"] != spec_sha256:
                    raise StoreError("run_id is already enrolled with a different spec")
                return "exists"
            db.execute(
                """INSERT INTO runs(run_id, task_id, state, spec_json, spec_sha256, config_sha256,
                   source_fingerprint, protected_fingerprint, attempt, cancel_requested)
                   VALUES(?,?,?,?,?,?,?,?,0,0)""",
                (
                    run_id, task_id, "prepared", spec_json, spec_sha256, config_sha256,
                    canonical_json(source_fingerprint), canonical_json(protected_fingerprint),
                ),
            )
            for index, name in enumerate(step_names):
                db.execute(
                    "INSERT INTO steps(run_id, step_index, name, state) VALUES(?,?,?,?)",
                    (run_id, index, name, "pending"),
                )
            self._event(db, run_id, "enrolled", {"spec_sha256": spec_sha256, "config_sha256": config_sha256})
            return "enrolled"

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connection() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return None if row is None else dict(row)

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        with self._connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM steps WHERE run_id=? ORDER BY step_index", (run_id,)
            )]

    def events(self, run_id: str | None = None) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute("SELECT sequence, body, event_hash FROM events ORDER BY sequence")
            events = []
            for row in rows:
                body = strict_load(row["body"])
                body["event_hash"] = row["event_hash"]
                if run_id is None or body["run_id"] == run_id:
                    events.append(body)
            return events

    def update_source_fingerprint(self, run_id: str, fingerprint: dict[str, str], patch_sha256: str) -> None:
        with self._connection(True) as db:
            row = db.execute("SELECT state FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise StoreError("unknown run")
            if row["state"] != "prepared":
                raise FenceError("patches are only admitted before execution")
            db.execute(
                "UPDATE runs SET source_fingerprint=? WHERE run_id=?",
                (canonical_json(fingerprint), run_id),
            )
            self._event(db, run_id, "patch_applied", {"patch_sha256": patch_sha256, "files": sorted(fingerprint)})

    def reject_patch(self, run_id: str, failure_class: str, patch_sha256: str) -> None:
        with self._connection(True) as db:
            row = db.execute("SELECT state FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None or row["state"] in TERMINAL_STATES:
                raise FenceError("patch rejection is not admitted on a terminal run")
            self._event(db, run_id, "patch_rejected", {"failure_class": failure_class, "patch_sha256": patch_sha256})

    def claim(self, run_id: str, owner: str) -> str:
        identity = process_identity(os.getpid())
        if identity is None:
            raise StoreError("owner process identity is unavailable")
        pid, starttime = identity
        with self._connection(True) as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise StoreError("unknown run")
            if row["state"] in TERMINAL_STATES:
                return "terminal"
            if row["state"] == "running" and _same(row["owner_pid"], row["owner_starttime"]):
                return "busy"
            if row["state"] == "running":
                running = db.execute(
                    "SELECT step_index, attempt FROM steps WHERE run_id=? AND state='running' ORDER BY step_index",
                    (run_id,),
                ).fetchall()
                for step in running:
                    db.execute(
                        "UPDATE steps SET state='pending', attempt=NULL WHERE run_id=? AND step_index=?",
                        (run_id, step["step_index"]),
                    )
                    self._event(db, run_id, "step_interrupted", {
                        "step_index": step["step_index"], "attempt": step["attempt"],
                    })
                self._event(db, run_id, "worker_interrupted", {"owner_pid": row["owner_pid"]})
            attempt = int(row["attempt"]) + 1
            db.execute(
                """UPDATE runs SET state='running', owner=?, owner_pid=?, owner_starttime=?,
                   child_pid=NULL, child_starttime=NULL, attempt=? WHERE run_id=?""",
                (owner, pid, starttime, attempt, run_id),
            )
            self._event(db, run_id, "claimed", {"owner": owner, "attempt": attempt})
            return "claimed"

    def request_cancel(self, run_id: str) -> dict[str, Any] | None:
        with self._connection(True) as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise StoreError("unknown run")
            if row["state"] in TERMINAL_STATES:
                return dict(row)
            db.execute("UPDATE runs SET cancel_requested=1 WHERE run_id=?", (run_id,))
            self._event(db, run_id, "cancel_requested", {})
            return dict(db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone())

    def cancel_requested(self, run_id: str) -> bool:
        with self._connection() as db:
            row = db.execute("SELECT cancel_requested FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise StoreError("unknown run")
            return bool(row["cancel_requested"])

    def set_child(self, run_id: str, child_pid: int, child_starttime: int) -> None:
        identity = process_identity(os.getpid())
        with self._connection(True) as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None or identity is None or row["owner_pid"] != identity[0] or row["state"] != "running":
                raise FenceError("lost run ownership before the child was recorded")
            db.execute(
                "UPDATE runs SET child_pid=?, child_starttime=? WHERE run_id=?",
                (child_pid, child_starttime, run_id),
            )

    def mark_step_running(self, run_id: str, step_index: int) -> None:
        identity = process_identity(os.getpid())
        with self._connection(True) as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            step = db.execute(
                "SELECT * FROM steps WHERE run_id=? AND step_index=?", (run_id, step_index)
            ).fetchone()
            if row is None or step is None or identity is None or row["owner_pid"] != identity[0]:
                raise FenceError("lost run ownership")
            if step["state"] == "committed":
                raise FenceError("refusing to rerun a committed step")
            db.execute(
                "UPDATE steps SET state='running', attempt=? WHERE run_id=? AND step_index=?",
                (row["attempt"], run_id, step_index),
            )
            self._event(db, run_id, "step_started", {"step_index": step_index, "attempt": row["attempt"]})

    def commit_step(self, run_id: str, step_index: int, record: dict[str, Any], stdout: bytes, stderr: bytes) -> bool:
        """Return True when this call publishes the step, False when it is a duplicate."""
        encoded = canonical_json(record)
        record_hash = sha256_json(record)
        if digest_bytes(stdout) != record["stdout_sha256"] or digest_bytes(stderr) != record["stderr_sha256"]:
            raise StoreError("step record does not match captured streams")
        identity = process_identity(os.getpid())
        with self._connection(True) as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            step = db.execute(
                "SELECT * FROM steps WHERE run_id=? AND step_index=?", (run_id, step_index)
            ).fetchone()
            if row is None or step is None or identity is None:
                raise StoreError("unknown step")
            if step["state"] == "committed":
                if step["record_sha256"] == record_hash and step["stdout"] == stdout and step["stderr"] == stderr:
                    return False
                raise DuplicateCommitError("conflicting duplicate step result")
            if row["state"] != "running" or row["owner_pid"] != identity[0] or row["owner_starttime"] != identity[1]:
                raise FenceError("lost run ownership before step commit")
            db.execute(
                """UPDATE steps SET state='committed', record_json=?, record_sha256=?, stdout=?, stderr=?
                   WHERE run_id=? AND step_index=?""",
                (encoded, record_hash, stdout, stderr, run_id, step_index),
            )
            self._event(db, run_id, "step_committed", {"step_index": step_index, "record_sha256": record_hash})
            return True

    def finalize(
        self,
        run_id: str,
        *,
        state: str,
        failure_class: str,
        accepted: bool,
        candidate_incorrect: bool,
        metrics: dict[str, float] | None,
        notes: list[str],
        timeout_counts_as_incorrect: bool,
    ) -> tuple[dict[str, Any], bool]:
        """Publish one terminal manifest. The second identical publish is a duplicate."""
        if state not in TERMINAL_STATES:
            raise StoreError("terminal state is not recognized")
        identity = process_identity(os.getpid())
        with self._connection(True) as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise StoreError("unknown run")
            steps = [dict(item) for item in db.execute(
                "SELECT * FROM steps WHERE run_id=? ORDER BY step_index", (run_id,)
            )]
            manifest = _manifest_from_row(
                row, steps, state=state, failure_class=failure_class, accepted=accepted,
                candidate_incorrect=candidate_incorrect, metrics=metrics, notes=notes,
                timeout_counts_as_incorrect=timeout_counts_as_incorrect,
            )
            encoded = canonical_json(manifest)
            digest = sha256_json(manifest)
            if row["state"] in TERMINAL_STATES:
                if row["manifest_sha256"] == digest:
                    stored = strict_load(row["manifest_json"])
                    return stored, True
                raise DuplicateCommitError("conflicting duplicate run manifest")
            if identity is None or row["state"] != "running" or row["owner_pid"] != identity[0]:
                raise FenceError("lost run ownership before terminal commit")
            if manifest["accepted"] is not True:
                accepted_flag = 0
            else:
                accepted_flag = 1
            db.execute(
                """UPDATE runs SET state=?, failure_class=?, manifest_json=?, manifest_sha256=?,
                   accepted=?, candidate_incorrect=?, owner=NULL, owner_pid=NULL, owner_starttime=NULL,
                   child_pid=NULL, child_starttime=NULL WHERE run_id=?""",
                (
                    manifest["state"], manifest["failure_class"], encoded, digest, accepted_flag,
                    1 if manifest["candidate_incorrect"] else 0, run_id,
                ),
            )
            self._event(db, run_id, "terminal", {"manifest_sha256": digest, "failure_class": manifest["failure_class"]})
            return manifest, False

    def audit(self) -> dict[str, Any]:
        with self._connection() as db:
            prev = _ZERO
            rows = list(db.execute("SELECT sequence, body, event_hash FROM events ORDER BY sequence"))
            for index, row in enumerate(rows):
                body = strict_load(row["body"])
                if body.get("sequence") != index or body.get("prev_hash") != prev or sha256_json(body) != row["event_hash"]:
                    raise StoreError("event chain mismatch")
                prev = row["event_hash"]
            runs = list(db.execute("SELECT run_id, state, manifest_sha256, manifest_json FROM runs ORDER BY run_id"))
            for run in runs:
                if run["state"] in TERMINAL_STATES:
                    if not run["manifest_json"] or sha256_json(strict_load(run["manifest_json"])) != run["manifest_sha256"]:
                        raise StoreError("stored manifest hash mismatch")
            return {"events": len(rows), "runs": len(runs)}


def _same(pid: int | None, starttime: int | None) -> bool:
    if pid is None or starttime is None:
        return False
    return process_identity(int(pid)) == (int(pid), int(starttime))


def _manifest_from_row(
    row: sqlite3.Row,
    steps: list[dict[str, Any]],
    *,
    state: str,
    failure_class: str,
    accepted: bool,
    candidate_incorrect: bool,
    metrics: dict[str, float] | None,
    notes: list[str],
    timeout_counts_as_incorrect: bool,
) -> dict[str, Any]:
    summaries = []
    can_accept = accepted and state == "complete" and failure_class == "completed" and metrics is not None
    for step in steps:
        if step["state"] != "committed" or not step["record_json"]:
            can_accept = False
            summaries.append({
                "index": step["step_index"],
                "name": step["name"],
                "state": step["state"],
                "record_sha256": None,
                "stdout_sha256": None,
                "stderr_sha256": None,
                "returncode": None,
                "failure_class": None,
                "wall_s": None,
            })
            continue
        record = strict_load(step["record_json"])
        if digest_bytes(step["stdout"]) != record["stdout_sha256"] or digest_bytes(step["stderr"]) != record["stderr_sha256"]:
            can_accept = False
            failure_class = "acceptance_invariant"
        if record.get("returncode") != 0 or record.get("failure_class") != "completed":
            can_accept = False
        summaries.append({
            "index": step["step_index"],
            "name": step["name"],
            "state": "committed",
            "record_sha256": step["record_sha256"],
            "stdout_sha256": record["stdout_sha256"],
            "stderr_sha256": record["stderr_sha256"],
            "returncode": record["returncode"],
            "failure_class": record["failure_class"],
            "wall_s": record["wall_s"],
        })
    if not steps:
        can_accept = False
    if accepted and not can_accept:
        accepted = False
        candidate_incorrect = False
        state = "failed_closed"
        failure_class = "acceptance_invariant"
    if not accepted and state == "complete":
        state = "failed_closed"
    return {
        "schema_version": 1,
        "run_id": row["run_id"],
        "task_id": row["task_id"],
        "state": state,
        "failure_class": failure_class,
        "accepted": accepted,
        "candidate_incorrect": candidate_incorrect,
        "spec_sha256": row["spec_sha256"],
        "config_sha256": row["config_sha256"],
        "isolation": strict_load(row["spec_json"])["isolation"],
        "backend": strict_load(row["spec_json"])["backend"],
        "timeout_counts_as_incorrect": timeout_counts_as_incorrect,
        "source_fingerprint": strict_load(row["source_fingerprint"]),
        "protected_fingerprint": strict_load(row["protected_fingerprint"]),
        "steps": summaries,
        "metrics": metrics,
        "notes": list(notes),
    }
