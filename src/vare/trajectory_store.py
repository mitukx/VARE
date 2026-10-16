"""Single-host, step-boundary recovery; tools execute at least once.

Reuses the evaluation store's SQLite transactions and append-only hash chain.
The caller owns agent state serialization and external tool idempotency. A
checkpoint identity is a recorded assertion until an inference adapter checks
the actual loaded weights. No optimizer or promotion transaction lives here.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time

from .durable import FenceError, RuntimeMismatchError, Store
from .evidence import canonical_json, sha256_file, sha256_json
from .runner import digest, encode
from .types import Task, Verification, snapshot_verification


def identity(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value) is None:
        raise ValueError("invalid bounded identity")
    return value


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("invalid bounded integer")
    return value


def sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid SHA-256")
    return value


def clock(value):
    value = time.time() if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("invalid clock")
    return float(value)


class BackpressureError(RuntimeError):
    """The producer must wait for a consumer to free capacity."""


@dataclass(frozen=True, slots=True)
class PolicyCheckpoint:
    policy_id: str
    version: int
    checkpoint_sha256: str

    def __post_init__(self):
        identity(self.policy_id)
        integer(self.version, 0, 2**53)
        sha(self.checkpoint_sha256)


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One resumable boundary, including its resulting agent/environment state.

    generation uses RVL's existing native Generation JSON when available.
    action/observation/state are bounded JSON; state is NOT an OS snapshot.
    Tool callers must reuse operation_id across retries and declare semantics.
    """
    index: int
    event_id: str
    policy: PolicyCheckpoint
    action: dict
    observation: dict
    state: dict
    generation: dict | None = None

    def __post_init__(self):
        integer(self.index, 0, 4095)
        identity(self.event_id)
        if not isinstance(self.policy, PolicyCheckpoint):
            raise TypeError("step requires a policy checkpoint")
        for value in (self.action, self.observation, self.state):
            if type(value) is not dict:
                raise TypeError("step fields must be JSON objects")
        if self.action.get("kind") not in {"tool", "inference", "answer", "observation"}:
            raise ValueError("unknown action boundary")
        if self.action["kind"] == "tool":
            identity(self.action.get("operation_id"))
            if self.action.get("semantics") not in {"at_least_once", "externally_idempotent"}:
                raise ValueError("tool must declare external execution semantics")
        if self.generation is not None:
            g = self.generation
            if (type(g) is not dict or set(g) != {"prompt_id", "prompt", "response", "logprob", "token_count", "latency_s", "metadata"}
                    or any(not isinstance(g.get(k), str) for k in ("prompt_id", "prompt", "response"))
                    or type(g.get("metadata")) is not dict):
                raise ValueError("invalid generation boundary")
            tokens = g["metadata"].get("response_token_ids")
            probs = g["metadata"].get("response_token_logprobs")
            prompt_tokens = g["metadata"].get("prompt_token_ids")
            if prompt_tokens is not None and (not isinstance(prompt_tokens, list)
                    or any(type(t) is not int or t < 0 for t in prompt_tokens)):
                raise ValueError("invalid prompt tokens")
            if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("invalid response tokens")
            if probs is not None and (
                not isinstance(probs, list) or len(tokens) != len(probs)
                or any(type(p) not in (int, float) or not math.isfinite(p) for p in probs)
            ):
                raise ValueError("token/logprob alignment mismatch")
            if (type(g["token_count"]) is not int or g["token_count"] != len(tokens)
                    or type(g["logprob"]) not in (int, float) or not math.isfinite(g["logprob"])
                    or type(g["latency_s"]) not in (int, float) or not math.isfinite(g["latency_s"]) or g["latency_s"] < 0
                    or (probs is not None and not math.isclose(g["logprob"], sum(probs), abs_tol=1e-6, rel_tol=1e-6))):
                raise ValueError("generation count/logprob/latency mismatch")


@dataclass(frozen=True, slots=True)
class TrajectoryLease:
    trajectory_id: str
    stage: str
    generation: int
    worker_id: str

    def __post_init__(self):
        identity(self.trajectory_id)
        identity(self.worker_id)
        integer(self.generation, 1, 10)
        if self.stage not in {"execute", "verify"}:
            raise ValueError("unsupported lease stage")


class TrajectoryStore:
    """Incremental bounded records and durable execute/verify work queues.

    Each step and cursor transition commits atomically. Delivery/re-execution
    is at least once; effective step keys are unique. Leases fence DB commits,
    not external tool execution. All DB access assumes a trusted local owner.
    """

    event = staticmethod(Store.event)

    def __init__(self, path):
        self._database = Store(path)
        self.path = self._database.path

    @classmethod
    def create(cls, path, *, run_id, max_inflight=2048, record_bytes=65536,
               storage_bytes=1024**3):
        identity(run_id)
        integer(max_inflight, 1, 100000)
        integer(record_bytes, 1024, 1024**2)
        integer(storage_bytes, 1024**2, 32 * 1024**3)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        config = dict(run_id=run_id, max_inflight=max_inflight, record_bytes=record_bytes,
                      storage_bytes=storage_bytes, source_sha256=sha256_file(__file__), schema=1)
        db = sqlite3.connect(path, isolation_level=None)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.executescript("""
                CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE policies(id TEXT PRIMARY KEY,version INTEGER UNIQUE NOT NULL,
                    checkpoint TEXT NOT NULL);
                CREATE TABLE verifiers(name TEXT NOT NULL,version INTEGER NOT NULL,
                    config TEXT NOT NULL,PRIMARY KEY(name,version));
                CREATE TABLE trajectories(id TEXT PRIMARY KEY,task TEXT NOT NULL,group_id TEXT NOT NULL,
                    member INTEGER NOT NULL,group_size INTEGER NOT NULL,state TEXT NOT NULL,
                    next_step INTEGER NOT NULL DEFAULT 0,expected_steps INTEGER NOT NULL,
                    created REAL NOT NULL,expires REAL NOT NULL,head TEXT NOT NULL,
                    verification TEXT,reason TEXT,UNIQUE(group_id,member));
                CREATE INDEX trajectory_state ON trajectories(state,id);
                CREATE INDEX trajectory_group ON trajectories(group_id,member);
                CREATE INDEX trajectory_expiry ON trajectories(expires,state);
                CREATE TABLE work(trajectory_id TEXT NOT NULL,stage TEXT NOT NULL,state TEXT NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 0,owner TEXT,deadline REAL,
                    max_attempts INTEGER NOT NULL,PRIMARY KEY(trajectory_id,stage),
                    FOREIGN KEY(trajectory_id) REFERENCES trajectories(id));
                CREATE INDEX work_queue ON work(stage,state,trajectory_id);
                CREATE INDEX work_deadlines ON work(stage,state,deadline);
                CREATE TABLE attempts(trajectory_id TEXT NOT NULL,stage TEXT NOT NULL,
                    generation INTEGER NOT NULL,worker_id TEXT NOT NULL,state TEXT NOT NULL,
                    PRIMARY KEY(trajectory_id,stage,generation));
                CREATE TABLE steps(trajectory_id TEXT NOT NULL,step INTEGER NOT NULL,
                    event_id TEXT NOT NULL,event_sequence INTEGER UNIQUE NOT NULL,hash TEXT NOT NULL,
                    PRIMARY KEY(trajectory_id,step),UNIQUE(trajectory_id,event_id));
                CREATE TABLE events(sequence INTEGER PRIMARY KEY,event TEXT NOT NULL,event_hash TEXT NOT NULL);
                CREATE INDEX event_identity ON events(json_extract(event,'$.job_id'),sequence);
            """)
            db.execute("INSERT INTO meta VALUES('config',?)", (canonical_json(config),))
        finally:
            db.close()
        return cls(path)

    @contextmanager
    def connection(self, write=False):
        with self._database.connection(write) as db:
            config = json.loads(db.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
            if config["schema"] != 1 or config["source_sha256"] != sha256_file(__file__):
                raise RuntimeMismatchError("resume with the enrolled trajectory runtime source")
            db.execute("PRAGMA wal_autocheckpoint=64")
            db.execute(f"PRAGMA max_page_count={config['storage_bytes'] // 4096}")
            yield db

    def config(self):
        with self.connection() as db:
            return json.loads(db.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])

    @staticmethod
    def _bounded(db, value):
        encoded = canonical_json(value)
        limit = json.loads(db.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])["record_bytes"]
        if len(encoded.encode()) > limit:
            raise ValueError("record exceeds configured byte budget")
        return encoded

    def register_policy(self, policy: PolicyCheckpoint):
        if not isinstance(policy, PolicyCheckpoint):
            raise TypeError("policy checkpoint required")
        with self.connection(True) as db:
            row = db.execute("SELECT * FROM policies WHERE id=? OR version=?",
                             (policy.policy_id, policy.version)).fetchone()
            expected = (policy.policy_id, policy.version, policy.checkpoint_sha256)
            if row is not None:
                if tuple(row) != expected:
                    raise ValueError("policy identity/version cannot be rebound")
                return
            db.execute("INSERT INTO policies VALUES(?,?,?)", expected)
            self.event(db, "policy_registered", policy.policy_id, policy.version, asdict(policy))

    def register_verifier(self, name, version, config_sha256):
        identity(name)
        integer(version, 0, 2**53)
        sha(config_sha256)
        with self.connection(True) as db:
            row = db.execute("SELECT config FROM verifiers WHERE name=? AND version=?", (name, version)).fetchone()
            if row is not None:
                if row[0] != config_sha256:
                    raise ValueError("verifier version cannot be rebound")
                return
            db.execute("INSERT INTO verifiers VALUES(?,?,?)", (name, version, config_sha256))
            self.event(db, "verifier_registered", name, version, {"config_sha256": config_sha256})

    def submit(self, trajectory_id, task: Task, *, expected_steps, group_id, member,
               group_size, max_attempts=3, lifetime_seconds=3600, now=None):
        for value in (trajectory_id, task.id, group_id):
            identity(value)
        integer(expected_steps, 1, 4096)
        integer(group_size, 1, 1024)
        integer(member, 0, group_size - 1)
        integer(max_attempts, 1, 10)
        if not .2 <= lifetime_seconds <= 86400 or not math.isfinite(lifetime_seconds):
            raise ValueError("invalid trajectory lifetime")
        now = clock(now)
        spec = dict(task=asdict(task), expected_steps=expected_steps, group_id=group_id,
                    member=member, group_size=group_size, max_attempts=max_attempts,
                    lifetime_seconds=lifetime_seconds)
        with self.connection(True) as db:
            encoded = self._bounded(db, spec)
            existing = db.execute("SELECT task FROM trajectories WHERE id=?", (trajectory_id,)).fetchone()
            if existing is not None:
                if existing[0] != encoded:
                    raise ValueError("trajectory identity reused with different enrollment")
                return False
            config = json.loads(db.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
            if group_size > config["max_inflight"]:
                raise ValueError("group cannot fit in bounded backlog")
            group = db.execute("SELECT task,group_size,state FROM trajectories WHERE group_id=? LIMIT 1", (group_id,)).fetchone()
            if group and (group["group_size"] != group_size or json.loads(group["task"])["task"] != asdict(task)):
                raise ValueError("group task or size mismatch")
            if group and group["state"] in {"consumed", "quarantined", "cancelled"}:
                raise ValueError("group is terminal; missing members cannot resurrect it")
            if group is None:
                # Reserve the WHOLE group with its first member, under the same
                # write transaction. Concurrent producers cannot strand siblings.
                reserved = db.execute("SELECT coalesce(sum(size),0) FROM (SELECT max(group_size) AS size FROM trajectories WHERE state NOT IN ('consumed','quarantined','cancelled') GROUP BY group_id)").fetchone()[0]
                if reserved + group_size > config["max_inflight"]:
                    raise BackpressureError("trajectory group backlog full")
            head = sha256_json({"trajectory_id": trajectory_id, "spec": spec})
            db.execute("INSERT INTO trajectories(id,task,group_id,member,group_size,state,expected_steps,created,expires,head) VALUES(?,?,?,?,?,'queued',?,?,?,?)",
                       (trajectory_id, encoded, group_id, member, group_size, expected_steps, now, now + lifetime_seconds, head))
            db.execute("INSERT INTO work(trajectory_id,stage,state,max_attempts) VALUES(?,'execute','queued',?)", (trajectory_id, max_attempts))
            self.event(db, "trajectory_enrolled", trajectory_id, 0, {"spec": spec, "head": head, "created": now, "expires": now + lifetime_seconds})
            return True

    @staticmethod
    def _owned(db, lease, now):
        row = db.execute("SELECT w.*,t.expires FROM work w JOIN trajectories t ON t.id=w.trajectory_id WHERE w.trajectory_id=? AND w.stage=?",
                         (lease.trajectory_id, lease.stage)).fetchone()
        if (row is None or row["state"] != "running" or row["generation"] != lease.generation
                or row["owner"] != lease.worker_id or row["deadline"] <= now or row["expires"] <= now):
            raise FenceError("expired or superseded trajectory lease")
        return row

    def _retry(self, db, row, reason):
        current = db.execute("SELECT state,generation FROM work WHERE trajectory_id=? AND stage=?", (row["trajectory_id"], row["stage"])).fetchone()
        if current["state"] != "running" or current["generation"] != row["generation"]:
            return  # A sibling's exhausted retry already terminated this group.
        exhausted = row["generation"] >= row["max_attempts"]
        db.execute("UPDATE attempts SET state=? WHERE trajectory_id=? AND stage=? AND generation=?",
                   (reason, row["trajectory_id"], row["stage"], row["generation"]))
        db.execute("UPDATE work SET state=?,owner=NULL,deadline=NULL WHERE trajectory_id=? AND stage=?",
                   ("done" if exhausted else "queued", row["trajectory_id"], row["stage"]))
        if exhausted:
            self._terminate_group(db, row["trajectory_id"], "quarantined", "retry_exhausted")
        self.event(db, reason, row["trajectory_id"], row["generation"], {"stage": row["stage"], "exhausted": exhausted})

    def claim(self, stage, worker_id, *, lease_seconds=5, now=None):
        if stage not in {"execute", "verify"}:
            raise ValueError("unsupported work stage")
        identity(worker_id)
        if type(lease_seconds) not in (int, float) or not math.isfinite(lease_seconds) or not .05 <= lease_seconds <= 3600:
            raise ValueError("invalid lease duration")
        now = clock(now)
        with self.connection(True) as db:
            # Bound recovery work per claim; repeated calls drain a large expired queue.
            timed_out = db.execute("SELECT id FROM trajectories WHERE expires<=? AND state NOT IN ('consumed','quarantined','cancelled') ORDER BY expires LIMIT 64", (now,)).fetchall()
            for trajectory in timed_out:
                self._cancel(db, trajectory["id"], "lifetime_exceeded")
            expired = db.execute("SELECT * FROM work WHERE stage=? AND state='running' AND deadline<=? ORDER BY deadline LIMIT 64", (stage, now)).fetchall()
            for row in expired:
                self._retry(db, row, "lease_expired")
            row = db.execute("SELECT w.*,t.expires FROM work w JOIN trajectories t ON t.id=w.trajectory_id WHERE w.stage=? AND w.state='queued' ORDER BY w.trajectory_id LIMIT 1", (stage,)).fetchone()
            if row is None:
                return None
            if row["expires"] <= now:
                self._cancel(db, row["trajectory_id"], "lifetime_exceeded")
                return None
            generation = row["generation"] + 1
            db.execute("UPDATE work SET state='running',generation=?,owner=?,deadline=? WHERE trajectory_id=? AND stage=?", (generation, worker_id, min(now + lease_seconds, row["expires"]), row["trajectory_id"], stage))
            db.execute("INSERT INTO attempts VALUES(?,?,?,?,'running')", (row["trajectory_id"], stage, generation, worker_id))
            if stage == "execute":
                db.execute("UPDATE trajectories SET state='running' WHERE id=?", (row["trajectory_id"],))
            self.event(db, "work_claimed", row["trajectory_id"], generation, {"stage": stage, "worker_id": worker_id, "deadline": min(now + lease_seconds, row["expires"])})
            return TrajectoryLease(row["trajectory_id"], stage, generation, worker_id)

    def heartbeat(self, lease, *, lease_seconds=5, now=None):
        if type(lease_seconds) not in (int, float) or not math.isfinite(lease_seconds) or not .05 <= lease_seconds <= 3600:
            raise ValueError("invalid heartbeat duration")
        now = clock(now)
        with self.connection(True) as db:
            row = self._owned(db, lease, now)
            db.execute("UPDATE work SET deadline=? WHERE trajectory_id=? AND stage=?", (min(now + lease_seconds, row["expires"]), lease.trajectory_id, lease.stage))
            # The lease row is mutable; lifecycle/step history remains append-only.

    def retry(self, lease, *, now=None):
        with self.connection(True) as db:
            row = self._owned(db, lease, clock(now))
            self._retry(db, row, "worker_failed")

    def append_step(self, lease, step: AgentStep, *, now=None):
        if lease.stage != "execute" or not isinstance(step, AgentStep):
            raise ValueError("execution lease and AgentStep required")
        payload = asdict(step)
        # Frozen dataclasses can still contain caller-mutable JSON dictionaries.
        AgentStep(**{**payload, "policy": PolicyCheckpoint(**payload["policy"])})
        with self.connection(True) as db:
            self._owned(db, lease, clock(now))
            self._bounded(db, payload)
            policy = db.execute("SELECT version,checkpoint FROM policies WHERE id=?", (step.policy.policy_id,)).fetchone()
            if policy is None or tuple(policy) != (step.policy.version, step.policy.checkpoint_sha256):
                raise ValueError("unregistered policy checkpoint")
            row = db.execute("SELECT * FROM trajectories WHERE id=?", (lease.trajectory_id,)).fetchone()
            duplicate = db.execute("SELECT hash FROM steps WHERE trajectory_id=? AND event_id=?", (lease.trajectory_id, step.event_id)).fetchone()
            payload_hash = sha256_json(payload)
            if duplicate:
                if duplicate[0] != payload_hash:
                    raise ValueError("event identity reused with different payload")
                return False
            if step.index != row["next_step"] or step.index >= row["expected_steps"]:
                raise ValueError("out-of-order or excess step")
            head = sha256_json({"previous": row["head"], "step_sha256": payload_hash})
            self.event(db, "step_committed", lease.trajectory_id, lease.generation, {"step": payload, "previous": row["head"], "head": head, "worker_id": lease.worker_id})
            sequence = db.execute("SELECT max(sequence) FROM events").fetchone()[0]
            db.execute("INSERT INTO steps VALUES(?,?,?,?,?)", (lease.trajectory_id, step.index, step.event_id, sequence, payload_hash))
            db.execute("UPDATE trajectories SET next_step=next_step+1,head=? WHERE id=?", (head, lease.trajectory_id))
            return True

    def trajectory(self, trajectory_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM trajectories WHERE id=?", (trajectory_id,)).fetchone()
            if row is None:
                raise KeyError(trajectory_id)
            return dict(row)

    def iter_steps(self, trajectory_id, *, after=-1) -> Iterator[dict]:
        # A cursor holds at most one bounded event, never the whole trajectory.
        with self.connection() as db:
            rows = db.execute("SELECT e.event FROM steps s JOIN events e ON e.sequence=s.event_sequence WHERE s.trajectory_id=? AND s.step>? ORDER BY s.step", (trajectory_id, after))
            for row in rows:
                yield json.loads(row[0])["details"]["step"]

    def resume_state(self, trajectory_id):
        row = self.trajectory(trajectory_id)
        last = next(self.iter_steps(trajectory_id, after=row["next_step"] - 2), None)
        return {"next_step": row["next_step"], "state": last["state"] if last else {},
                "spec": json.loads(row["task"]), "head": row["head"]}

    def finish(self, lease, *, now=None):
        if lease.stage != "execute":
            raise ValueError("execution lease required")
        with self.connection(True) as db:
            owned = self._owned(db, lease, clock(now))
            row = db.execute("SELECT * FROM trajectories WHERE id=?", (lease.trajectory_id,)).fetchone()
            if row["next_step"] != row["expected_steps"]:
                raise ValueError("incomplete trajectory cannot complete")
            db.execute("UPDATE trajectories SET state='complete' WHERE id=?", (lease.trajectory_id,))
            self._done(db, lease)
            db.execute("INSERT INTO work(trajectory_id,stage,state,max_attempts) VALUES(?,'verify','queued',?)", (lease.trajectory_id, owned["max_attempts"]))
            self.event(db, "trajectory_complete", lease.trajectory_id, lease.generation, {"head": row["head"], "steps": row["next_step"]})

    @staticmethod
    def _done(db, lease):
        db.execute("UPDATE work SET state='done',owner=NULL,deadline=NULL WHERE trajectory_id=? AND stage=?", (lease.trajectory_id, lease.stage))
        db.execute("UPDATE attempts SET state='done' WHERE trajectory_id=? AND stage=? AND generation=?", (lease.trajectory_id, lease.stage, lease.generation))

    def record_verification(self, lease, verification: Verification, *, now=None):
        value = snapshot_verification(verification)
        if lease.stage != "verify":
            raise ValueError("verification lease required")
        with self.connection(True) as db:
            self._owned(db, lease, clock(now))
            row = db.execute("SELECT * FROM trajectories WHERE id=?", (lease.trajectory_id,)).fetchone()
            registered = db.execute("SELECT config FROM verifiers WHERE name=? AND version=?", (value.verifier_name, value.verifier_version)).fetchone()
            if (row["state"] != "complete" or registered is None
                    or value.metadata.get("verifier_config_sha256") != registered[0]
                    or value.metadata.get("trajectory_sha256") != row["head"]):
                raise ValueError("verification must bind the complete trace and registered verifier")
            serialized = self._bounded(db, asdict(value))
            db.execute("UPDATE trajectories SET state='verified',verification=? WHERE id=?", (serialized, lease.trajectory_id))
            self._done(db, lease)
            self.event(db, "trajectory_verified", lease.trajectory_id, lease.generation, {"head": row["head"], "verification": asdict(value)})

    def close_verified(self, trajectory_id, *, head, reason="consumed"):
        """A consumer acknowledgment, NOT an optimizer-commit acknowledgment."""
        if reason not in {"consumed", "quarantined"}:
            raise ValueError("invalid close reason")
        with self.connection(True) as db:
            row = db.execute("SELECT state,head,group_size FROM trajectories WHERE id=?", (trajectory_id,)).fetchone()
            if row is None or row["head"] != head or row["state"] != "verified" or row["group_size"] != 1:
                raise FenceError("singleton verified trace required; grouped trajectories use close_group")
            db.execute("UPDATE trajectories SET state=?,reason=? WHERE id=?", (reason, reason, trajectory_id))
            self.event(db, "consumer_acknowledged", trajectory_id, 0, {"head": head, "state": reason})

    def _terminate_group(self, db, trajectory_id, state, reason):
        row = db.execute("SELECT state,group_id FROM trajectories WHERE id=?", (trajectory_id,)).fetchone()
        if row is None:
            raise KeyError(trajectory_id)
        if row["state"] in {"consumed", "quarantined", "cancelled"}:
            return False
        members = db.execute("SELECT id FROM trajectories WHERE group_id=? ORDER BY member", (row["group_id"],)).fetchall()
        for member in members:
            ident = member["id"]
            db.execute("UPDATE trajectories SET state=?,reason=? WHERE id=?", (state, reason, ident))
            db.execute("UPDATE work SET state='done',owner=NULL,deadline=NULL WHERE trajectory_id=?", (ident,))
            db.execute("UPDATE attempts SET state=? WHERE trajectory_id=? AND state='running'", (state, ident))
            self.event(db, f"trajectory_{state}", ident, 0, {"reason": reason})
        return True

    def _cancel(self, db, trajectory_id, reason):
        return self._terminate_group(db, trajectory_id, "cancelled", reason)

    def cancel(self, trajectory_id):
        with self.connection(True) as db:
            return self._cancel(db, trajectory_id, "operator_cancelled")

    def counts(self):
        with self.connection() as db:
            return {row[0]: row[1] for row in db.execute("SELECT state,count(*) FROM trajectories GROUP BY state")}

    def group(self, group_id):
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM trajectories WHERE group_id=? ORDER BY member LIMIT 1025", (group_id,))]

    def verified_groups(self, limit=16):
        integer(limit, 1, 1024)
        with self.connection() as db:
            candidates = db.execute("SELECT group_id FROM trajectories WHERE state='verified' GROUP BY group_id HAVING count(*)=max(group_size) ORDER BY group_id LIMIT ?", (limit,))
            return [row[0] for row in candidates]

    def close_group(self, group_id, heads, *, reason="consumed", decision=None):
        """Atomic consumer acknowledgment of an intact group; no optimizer claim."""
        if reason not in {"consumed", "quarantined"}:
            raise ValueError("invalid close reason")
        with self.connection(True) as db:
            rows = db.execute("SELECT id,state,head,group_size FROM trajectories WHERE group_id=? ORDER BY member", (group_id,)).fetchall()
            if (not rows or len(rows) != rows[0]["group_size"]
                    or any(row["state"] != "verified" for row in rows)
                    or {row["id"]: row["head"] for row in rows} != heads):
                raise FenceError("consumer must acknowledge the complete verified group")
            details = {"heads": heads, "state": reason, "decision": decision or {}}
            self._bounded(db, details)
            db.execute("UPDATE trajectories SET state=?,reason=? WHERE group_id=?", (reason, reason, group_id))
            self.event(db, "group_acknowledged", group_id, 0, details)

    def backup(self, path):
        path = Path(path)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        with sqlite3.connect(self.path) as source, sqlite3.connect(path) as target:
            source.backup(target, pages=64)
        return sha256_file(path)


def audit_trajectory_store(path):
    """Stream a quiescent snapshot; validate bytes, indices and terminal coverage."""
    store = TrajectoryStore(path)
    head, count = "0" * 64, 0
    with store.connection() as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity failure")
        for row in db.execute("SELECT * FROM events ORDER BY sequence"):
            event = json.loads(row["event"])
            if (event["sequence"] != count or event["previous_hash"] != head
                    or digest(encode(event)) != row["event_hash"]):
                raise ValueError("event history hash mismatch")
            head, count = row["event_hash"], count + 1
        step_count = 0
        for row in db.execute("SELECT s.*,e.event FROM steps s JOIN events e ON e.sequence=s.event_sequence ORDER BY s.trajectory_id,s.step"):
            event = json.loads(row["event"])
            payload = event["details"]["step"]
            if (event["kind"] != "step_committed" or event["job_id"] != row["trajectory_id"]
                    or payload["index"] != row["step"] or payload["event_id"] != row["event_id"]
                    or sha256_json(payload) != row["hash"]):
                raise ValueError("step index/evidence mismatch")
            AgentStep(**{**payload, "policy": PolicyCheckpoint(**payload["policy"])})
            step_count += 1
        for row in db.execute("SELECT * FROM trajectories ORDER BY id"):
            spec = json.loads(row["task"])
            history = [json.loads(e[0]) for e in db.execute("SELECT event FROM events WHERE json_extract(event,'$.job_id')=? AND json_extract(event,'$.kind') IN ('trajectory_enrolled','trajectory_complete','trajectory_verified','consumer_acknowledged','trajectory_cancelled') ORDER BY sequence", (row["id"],))]
            enrolled = [event for event in history if event["kind"] == "trajectory_enrolled"]
            completed = [event for event in history if event["kind"] == "trajectory_complete"]
            verified = [event for event in history if event["kind"] == "trajectory_verified"]
            if (len(enrolled) != 1 or enrolled[0]["details"]["spec"] != spec
                    or enrolled[0]["details"]["created"] != row["created"]
                    or enrolled[0]["details"]["expires"] != row["expires"]):
                raise ValueError("enrollment/history mismatch")
            expected = sha256_json({"trajectory_id": row["id"], "spec": spec})
            n = 0
            for step in db.execute("SELECT s.step,s.hash,e.event FROM steps s JOIN events e ON e.sequence=s.event_sequence WHERE s.trajectory_id=? ORDER BY s.step", (row["id"],)):
                body = json.loads(step["event"])["details"]
                updated = sha256_json({"previous": expected, "step_sha256": step["hash"]})
                if step["step"] != n or body["previous"] != expected or body["head"] != updated:
                    raise ValueError("trajectory step chain mismatch")
                expected, n = updated, n + 1
            if n != row["next_step"] or expected != row["head"] or n > row["expected_steps"]:
                raise ValueError("trajectory cursor/chain mismatch")
            if row["state"] in {"complete", "verified", "consumed"} and n != row["expected_steps"]:
                raise ValueError("terminal trajectory is incomplete")
            if completed and (len(completed) != 1 or completed[0]["details"] != {"head": row["head"], "steps": n}):
                raise ValueError("completion/history mismatch")
            if row["state"] in {"complete", "verified", "consumed"} and not completed:
                raise ValueError("missing completion event")
            if row["state"] in {"verified", "consumed"}:
                verification = snapshot_verification(Verification(**json.loads(row["verification"])))
                if verification.metadata.get("trajectory_sha256") != row["head"]:
                    raise ValueError("verification chain binding mismatch")
                if len(verified) != 1 or verified[0]["details"]["verification"] != asdict(verification):
                    raise ValueError("verification/history mismatch")
            if row["state"] == "consumed":
                acknowledged = any(e["kind"] == "consumer_acknowledged" and e["details"] == {"head": row["head"], "state": "consumed"} for e in history)
                for event in db.execute("SELECT event FROM events WHERE json_extract(event,'$.job_id')=? AND json_extract(event,'$.kind')='group_acknowledged'", (row["group_id"],)):
                    body = json.loads(event[0])["details"]
                    acknowledged |= body["state"] == "consumed" and body["heads"].get(row["id"]) == row["head"]
                if not acknowledged:
                    raise ValueError("missing consumer acknowledgment")
        completions = db.execute("SELECT count(*) FROM events WHERE json_extract(event,'$.kind')='trajectory_complete'").fetchone()[0]
        unique_completions = db.execute("SELECT count(DISTINCT json_extract(event,'$.job_id')) FROM events WHERE json_extract(event,'$.kind')='trajectory_complete'").fetchone()[0]
        if completions != unique_completions:
            raise ValueError("duplicated effective trajectory completion")
        effective_steps = db.execute("SELECT count(*) FROM events WHERE json_extract(event,'$.kind')='step_committed'").fetchone()[0]
        if effective_steps != step_count:
            raise ValueError("missing or duplicated effective step")
        for group in db.execute("SELECT group_id,count(*) AS n,max(group_size) AS size,sum(state IN ('consumed','quarantined','cancelled')) AS terminal,sum(state='consumed') AS consumed FROM trajectories GROUP BY group_id"):
            if group["terminal"] not in (0, group["n"]) or (group["consumed"] and group["consumed"] != group["size"]):
                raise ValueError("comparison group has mixed terminal/active or partial consumed members")
        return {"pass": True, "events": count, "steps": step_count, "completions": completions,
                "effective_duplicates": completions - unique_completions,
                "ledger_head_sha256": head, "database_sha256": sha256_file(path)}
