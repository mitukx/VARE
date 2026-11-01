from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence import canonical_json, sha256_json


@dataclass(frozen=True, slots=True)
class ProtocolLock:
    name: str
    config: dict[str, Any]
    hypotheses: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    acceptance: dict[str, Any]
    sha256: str

    @classmethod
    def create(
        cls,
        *,
        name: str,
        config: dict[str, Any],
        hypotheses: list[str] | tuple[str, ...],
        primary_metrics: list[str] | tuple[str, ...],
        acceptance: dict[str, Any],
    ) -> "ProtocolLock":
        body = {
            "name": name,
            "config": config,
            "hypotheses": list(hypotheses),
            "primary_metrics": list(primary_metrics),
            "acceptance": acceptance,
        }
        return cls(
            name=name,
            config=dict(config),
            hypotheses=tuple(hypotheses),
            primary_metrics=tuple(primary_metrics),
            acceptance=dict(acceptance),
            sha256=sha256_json(body),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config,
            "hypotheses": list(self.hypotheses),
            "primary_metrics": list(self.primary_metrics),
            "acceptance": self.acceptance,
            "sha256": self.sha256,
        }

    def write(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "ProtocolLock":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        lock = cls.create(
            name=str(raw["name"]),
            config=dict(raw["config"]),
            hypotheses=tuple(raw["hypotheses"]),
            primary_metrics=tuple(raw["primary_metrics"]),
            acceptance=dict(raw["acceptance"]),
        )
        if raw.get("sha256") != lock.sha256:
            raise ValueError("protocol lock digest mismatch")
        return lock

    def canonical_body(self) -> str:
        return canonical_json({k: v for k, v in self.payload().items() if k != "sha256"})
