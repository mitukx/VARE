from __future__ import annotations

import hashlib
import json
import os
import fcntl
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    path: str
    bytes: int
    sha256: str


def artifact_manifest(paths: Iterable[str | Path]) -> list[ArtifactRecord]:
    out: list[ArtifactRecord] = []
    for p in paths:
        path = Path(p)
        out.append(ArtifactRecord(path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path)))
    return out


@dataclass(frozen=True, slots=True)
class CapabilityScore:
    before: float
    after: float
    gpu_hours: float
    rollout_tokens: int = 0
    learner_tokens: int = 0
    verifier_cost: float = 0.0

    @property
    def gain(self) -> float:
        return self.after - self.before

    @property
    def gain_per_gpu_hour(self) -> float | None:
        return None if self.gpu_hours <= 0 else self.gain / self.gpu_hours

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "gain": self.gain,
            "gain_per_gpu_hour": self.gain_per_gpu_hour,
        }


class HashChainLedger:
    """Same-host, multi-writer JSONL hash chain on POSIX filesystems.

    Each append holds a file lock while validating the *current* tail and
    publishing a newline-terminated, fsynced record. Locks serialize writers;
    hashes detect accidental edits but do not authenticate an unanchored log.
    This is not a network-filesystem or adversarial-host security boundary.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq = 0
        self.head = "0" * 64
        if self.path.exists():
            self._recover()

    @staticmethod
    def _validate_text(contents: str) -> tuple[int, str]:
        head = "0" * 64
        seq = 0
        if contents and not contents.endswith("\n"):
            raise ValueError("ledger has an incomplete final row")
        for line in contents.splitlines():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("ledger row must be a JSON object")
            body = {k: v for k, v in row.items() if k != "sha256"}
            if type(body.get("seq")) is not int or body["seq"] != seq:
                raise ValueError("ledger sequence is not contiguous")
            if body.get("prev_sha256") != head:
                raise ValueError("ledger chain is broken")
            digest = sha256_json(body)
            if row.get("sha256") != digest:
                raise ValueError("ledger row digest mismatch")
            head = digest
            seq += 1
        return seq, head

    def _recover(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                self.seq, self.head = self._validate_text(handle.read())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def append(self, event: str, payload: dict[str, Any]) -> str:
        # A second HashChainLedger instance may have appended after this object
        # was constructed. Never trust its cached sequence or previous digest.
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                seq, head = self._validate_text(handle.read())
                body = {
                    "seq": seq,
                    "event": event,
                    "payload": payload,
                    "prev_sha256": head,
                }
                digest = sha256_json(body)
                row = {**body, "sha256": digest}
                handle.seek(0, os.SEEK_END)
                handle.write(canonical_json(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                self.seq = seq + 1
                self.head = digest
                return digest
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
