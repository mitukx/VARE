"""Operator-owned content-addressed checkpoint files for promotion provenance.

The store verifies actual bytes each time an artifact is referenced. A trusted
owner must protect this directory and the database from candidate processes:
hashes and POSIX permissions within a shared UID are not a sandbox.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class CheckpointIntegrityError(RuntimeError):
    """Missing, modified or ambiguously identified persisted checkpoint."""


class TrustedCheckpointStore:
    """Local immutable blob import + durable policy-ID -> byte-digest binding.

    This is an artifact transport for a trusted evaluator and the VARE
    `checkpoint_hasher` callback. Exporting a full trainer snapshot to a
    deterministic byte format remains the backend's responsibility.
    """

    def __init__(self, root: str | Path, *, max_bytes: int = 4 * 1024**3) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = Path(root).resolve()
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "artifacts.sqlite"
        self.max_bytes = max_bytes
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS artifacts ("
                "policy_id TEXT PRIMARY KEY, digest TEXT NOT NULL, size INTEGER NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=10)
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA journal_mode=DELETE")
        return db

    @staticmethod
    def _check_policy_id(policy_id: str) -> None:
        if not isinstance(policy_id, str) or not policy_id.strip() or len(policy_id) > 256:
            raise ValueError("policy_id must be nonempty and bounded")

    @staticmethod
    def _read_chunks(path: Path, *, max_bytes: int):
        # Reject leaf symlinks even when source is passed by a caller.
        if path.is_symlink():
            raise CheckpointIntegrityError("symlinked checkpoint file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise CheckpointIntegrityError("checkpoint object cannot be opened") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
                raise CheckpointIntegrityError("checkpoint is not a bounded regular file")
            with os.fdopen(fd, "rb", closefd=False) as handle:
                total = 0
                while block := handle.read(1024 * 1024):
                    total += len(block)
                    if total > max_bytes:
                        raise CheckpointIntegrityError("checkpoint exceeded byte budget")
                    yield block
        finally:
            os.close(fd)

    def _lookup(self, policy_id: str) -> tuple[str, int]:
        self._check_policy_id(policy_id)
        with self._connect() as db:
            row = db.execute(
                "SELECT digest, size FROM artifacts WHERE policy_id = ?", (policy_id,)
            ).fetchone()
        if row is None:
            raise CheckpointIntegrityError("policy has no registered checkpoint")
        digest, size = row
        if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
            raise CheckpointIntegrityError("invalid stored checkpoint digest")
        if type(size) is not int or not 0 <= size <= self.max_bytes:
            raise CheckpointIntegrityError("invalid stored checkpoint size")
        return digest, size

    def register_file(self, policy_id: str, source: str | Path) -> str:
        """Copy bytes into a new content-addressed object, never replace a policy."""
        self._check_policy_id(policy_id)
        fd, tmp = tempfile.mkstemp(prefix=".checkpoint-", dir=self.objects)
        temp = Path(tmp)
        hasher = hashlib.sha256()
        n = 0
        try:
            with os.fdopen(fd, "wb") as out:
                for block in self._read_chunks(Path(source), max_bytes=self.max_bytes):
                    out.write(block)
                    hasher.update(block)
                    n += len(block)
                out.flush()
                os.fsync(out.fileno())
            digest = hasher.hexdigest()
            dest = self.objects / digest
            try:
                os.link(temp, dest)
            except FileExistsError:
                self._verify_file(dest, digest, n)
            directory_fd = os.open(self.objects, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT digest, size FROM artifacts WHERE policy_id = ?", (policy_id,)
                ).fetchone()
                if row is not None:
                    if row != (digest, n):
                        raise CheckpointIntegrityError("policy already registered to different bytes")
                else:
                    db.execute(
                        "INSERT INTO artifacts (policy_id,digest,size) VALUES (?,?,?)",
                        (policy_id, digest, n),
                    )
                db.commit()
            return digest
        finally:
            temp.unlink(missing_ok=True)

    def _verify_file(self, path: Path, digest: str, expected_size: int) -> None:
        hasher = hashlib.sha256()
        n = 0
        for block in self._read_chunks(path, max_bytes=self.max_bytes):
            hasher.update(block)
            n += len(block)
        if n != expected_size or hasher.hexdigest() != digest:
            raise CheckpointIntegrityError("checkpoint object hash or size mismatch")

    def digest(self, policy_id: str) -> str:
        """Trusted callback for CapabilityLoop: re-hash actual persisted bytes."""
        digest, size = self._lookup(policy_id)
        self._verify_file(self.objects / digest, digest, size)
        return digest

    def read_verified(self, policy_id: str, *, max_read_bytes: int = 16 * 1024 * 1024) -> tuple[bytes, str]:
        """Small-model CPU fixtures: read and hash bytes from the same descriptor."""
        digest, size = self._lookup(policy_id)
        if type(max_read_bytes) is not int or not 0 < max_read_bytes <= self.max_bytes or size > max_read_bytes:
            raise CheckpointIntegrityError("checkpoint too large for in-memory evaluation")
        blocks = list(self._read_chunks(self.objects / digest, max_bytes=self.max_bytes))
        raw = b"".join(blocks)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise CheckpointIntegrityError("loaded checkpoint does not match manifest")
        return raw, digest

    def artifact_path(self, policy_id: str) -> Path:
        """For trusted large-model loaders. Hash again after loading to catch races."""
        digest = self.digest(policy_id)
        return self.objects / digest
