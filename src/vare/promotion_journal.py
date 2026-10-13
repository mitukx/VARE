"""Durable, single-host promotion intent/pointer journal (CPU, stdlib SQLite).

The journal makes promotion *recoverably uncertain*, not magically atomic with
external model/optimizer state. After a crash during promote, all future rounds
must stop until a trusted operator reconciles independently hashed checkpoints.
"""
from __future__ import annotations

from contextlib import contextmanager
import re
import sqlite3
from pathlib import Path
from typing import Iterator


_DIGEST = re.compile(r"[a-f0-9]{64}\Z")


class PromotionRecoveryRequired(RuntimeError):
    """Unknown/ambiguous promotion state: refuse subsequent model updates."""


def _validate(policy_id: str, checkpoint_sha256: str) -> None:
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError("policy ID must be nonempty")
    if not isinstance(checkpoint_sha256, str) or _DIGEST.fullmatch(checkpoint_sha256) is None:
        raise ValueError("checkpoint digest must be lowercase SHA-256")


class PromotionJournal:
    """SQLite FULL-sync local transaction register and recovery fence.

    The initial policy and checkpoint must come from an operator-controlled,
    independently verified durable artifact store; NOT from candidate metadata.
    A database file in an agent-writable directory is not a security boundary.
    """

    def __init__(
        self, path: str | Path, *, initial_policy_id: str, initial_checkpoint_sha256: str
    ) -> None:
        _validate(initial_policy_id, initial_checkpoint_sha256)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS active ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
                "policy_id TEXT NOT NULL, checkpoint_sha256 TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS transactions ("
                "tx_id TEXT PRIMARY KEY, incumbent_id TEXT NOT NULL, "
                "incumbent_sha256 TEXT NOT NULL, candidate_id TEXT NOT NULL, "
                "candidate_sha256 TEXT NOT NULL, evaluation_sha256 TEXT NOT NULL, "
                "state TEXT NOT NULL CHECK(state IN ('prepared','committed','aborted')))"
            )
            db.execute(
                "INSERT OR IGNORE INTO active VALUES (1, ?, ?)",
                (initial_policy_id, initial_checkpoint_sha256),
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA journal_mode=DELETE")
            with db:
                db.execute("BEGIN IMMEDIATE")
                yield db
        finally:
            db.close()

    @staticmethod
    def _active(db: sqlite3.Connection) -> tuple[str, str]:
        row = db.execute(
            "SELECT policy_id, checkpoint_sha256 FROM active WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise PromotionRecoveryRequired("missing durable active checkpoint")
        return str(row[0]), str(row[1])

    @staticmethod
    def _pending(db: sqlite3.Connection) -> tuple[str, str, str, str, str] | None:
        rows = db.execute(
            "SELECT tx_id, incumbent_id, incumbent_sha256, candidate_id, candidate_sha256 "
            "FROM transactions WHERE state='prepared'"
        ).fetchall()
        if len(rows) > 1:
            raise PromotionRecoveryRequired("multiple pending promotions")
        return tuple(str(v) for v in rows[0]) if rows else None  # type: ignore[return-value]

    def assert_clean(self, *, policy_id: str, checkpoint_sha256: str) -> None:
        """Call before any new rollout, training or candidate evaluation."""
        _validate(policy_id, checkpoint_sha256)
        with self._connection() as db:
            if self._pending(db) is not None:
                raise PromotionRecoveryRequired("unresolved promotion transaction")
            if self._active(db) != (policy_id, checkpoint_sha256):
                raise PromotionRecoveryRequired("runtime and durable checkpoint pointer disagree")

    def prepare(
        self,
        *,
        tx_id: str,
        incumbent_id: str,
        incumbent_sha256: str,
        candidate_id: str,
        candidate_sha256: str,
        evaluation_sha256: str,
    ) -> None:
        _validate(incumbent_id, incumbent_sha256)
        _validate(candidate_id, candidate_sha256)
        if (
            not isinstance(tx_id, str) or not 16 <= len(tx_id) <= 256
            or candidate_id == incumbent_id
            or candidate_sha256 == incumbent_sha256
            or not isinstance(evaluation_sha256, str)
            or _DIGEST.fullmatch(evaluation_sha256) is None
        ):
            raise ValueError("invalid or repeated promotion candidate/nonce/evidence")
        with self._connection() as db:
            if self._pending(db) is not None or self._active(db) != (incumbent_id, incumbent_sha256):
                raise PromotionRecoveryRequired("cannot prepare while state is dirty")
            try:
                db.execute(
                    "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, 'prepared')",
                    (tx_id, incumbent_id, incumbent_sha256, candidate_id,
                     candidate_sha256, evaluation_sha256),
                )
            except sqlite3.IntegrityError as exc:
                raise PromotionRecoveryRequired("promotion nonce already consumed") from exc

    def commit(self, tx_id: str) -> None:
        """Call ONLY after hook promotion succeeded and live checkpoint was verified."""
        with self._connection() as db:
            pending = self._pending(db)
            if pending is None or pending[0] != tx_id:
                raise PromotionRecoveryRequired("missing prepared promotion")
            _, old_id, old_sha, new_id, new_sha = pending
            if self._active(db) != (old_id, old_sha):
                raise PromotionRecoveryRequired("durable incumbent changed during promotion")
            db.execute(
                "UPDATE active SET policy_id=?, checkpoint_sha256=? WHERE singleton=1",
                (new_id, new_sha),
            )
            db.execute(
                "UPDATE transactions SET state='committed' WHERE tx_id=?", (tx_id,)
            )

    def reconcile(
        self, *, tx_id: str, observed_policy_id: str, observed_checkpoint_sha256: str
    ) -> str:
        """Operator-only recovery after verifying the *actual* persisted checkpoint.

        The external checkpoint store must be independently trusted. A candidate
        merely claiming one of these hashes must never count as observation.
        """
        _validate(observed_policy_id, observed_checkpoint_sha256)
        with self._connection() as db:
            pending = self._pending(db)
            if pending is None or pending[0] != tx_id:
                raise PromotionRecoveryRequired("no matching pending promotion")
            _, old_id, old_sha, new_id, new_sha = pending
            if self._active(db) != (old_id, old_sha):
                raise PromotionRecoveryRequired("unexpected durable champion during recovery")
            if (observed_policy_id, observed_checkpoint_sha256) == (old_id, old_sha):
                state = "aborted"
            elif (observed_policy_id, observed_checkpoint_sha256) == (new_id, new_sha):
                state = "committed"
                db.execute(
                    "UPDATE active SET policy_id=?, checkpoint_sha256=? WHERE singleton=1",
                    (new_id, new_sha),
                )
            else:
                raise PromotionRecoveryRequired("unknown runtime checkpoint: halt and inspect")
            db.execute(
                "UPDATE transactions SET state=? WHERE tx_id=?", (state, tx_id)
            )
            return state

    def status(self) -> dict[str, str | None]:
        with self._connection() as db:
            policy_id, digest = self._active(db)
            pending = self._pending(db)
            return {
                "active_policy_id": policy_id,
                "active_checkpoint_sha256": digest,
                "pending_tx_id": pending[0] if pending else None,
            }
