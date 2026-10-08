from __future__ import annotations

import hashlib
import json
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
    """Append-only JSONL ledger with a tamper-evident hash chain."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq = 0
        self.head = "0" * 64
        if self.path.exists() and self.path.stat().st_size:
            self._recover()

    def _recover(self) -> None:
        head = "0" * 64
        seq = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            body = {k: v for k, v in row.items() if k != "sha256"}
            if body.get("prev_sha256") != head:
                raise ValueError("ledger chain is broken")
            digest = sha256_json(body)
            if row.get("sha256") != digest:
                raise ValueError("ledger row digest mismatch")
            head = digest
            seq = int(body["seq"]) + 1
        self.seq = seq
        self.head = head

    def append(self, event: str, payload: dict[str, Any]) -> str:
        body = {
            "seq": self.seq,
            "event": event,
            "payload": payload,
            "prev_sha256": self.head,
        }
        digest = sha256_json(body)
        row = {**body, "sha256": digest}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(canonical_json(row) + "\n")
        self.seq += 1
        self.head = digest
        return digest
