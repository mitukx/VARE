from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from ..types import Attempt, Experience, Task, Verification


_ALLOWED_STATUSES = {"pending_verification", "verifying", "ready", "consumed", "stale", "quarantined"}


@dataclass(frozen=True, slots=True)
class RVLReplaySnapshot:
    groups: int
    generations: int
    statuses: dict[str, int]
    mean_reward: float | None
    policy_versions: tuple[int, ...]
    verifier_versions: tuple[int, ...]
    max_policy_lag: int
    max_verifier_lag: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": self.groups,
            "generations": self.generations,
            "statuses": dict(self.statuses),
            "mean_reward": self.mean_reward,
            "policy_versions": list(self.policy_versions),
            "verifier_versions": list(self.verifier_versions),
            "max_policy_lag": self.max_policy_lag,
            "max_verifier_lag": self.max_verifier_lag,
        }


class RVLTokenReplayReader:
    """Read RVL TokenReplay SQLite without mutating the training database.

    The adapter is deliberately read-only. RVL remains the owner of immutable
    behavior data, leases, verifier rewrites and learner-consumption state.
    VARE consumes provenance for outer-loop diagnosis and curriculum decisions.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.path.resolve()}?mode=ro"
        db = sqlite3.connect(uri, uri=True)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _validate_status(status: str) -> str:
        if status not in _ALLOWED_STATUSES:
            raise ValueError(f"unknown RVL replay status: {status}")
        return status

    @staticmethod
    def _require_version(value: Any, field: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{field} must be a non-negative integer version")
        return value

    @staticmethod
    def _reject_future_version(recorded: int, current: int | None, field: str) -> None:
        if current is not None and recorded > current:
            raise ValueError(
                f"{field} version {recorded} is ahead of current version {current}"
            )

    @classmethod
    def _validate_current_version(cls, value: int | None, field: str) -> int | None:
        if value is None:
            return None
        return cls._require_version(value, field)

    @staticmethod
    def _decode_payload(raw: str) -> list[dict[str, Any]]:
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError("RVL replay payload must be a list")
        return [dict(x) for x in payload]

    def snapshot(
        self,
        *,
        current_policy_version: int | None = None,
        current_verifier_version: int | None = None,
    ) -> RVLReplaySnapshot:
        current_policy_version = self._validate_current_version(
            current_policy_version, "current_policy"
        )
        current_verifier_version = self._validate_current_version(
            current_verifier_version, "current_verifier"
        )
        statuses: dict[str, int] = {}
        rewards: list[float] = []
        policies: set[int] = set()
        verifiers: set[int] = set()
        generations = 0
        groups = 0
        with self._connect() as db:
            rows = db.execute(
                "SELECT policy_version,payload,status,verifier_version FROM groups ORDER BY rowid"
            ).fetchall()
        for row in rows:
            groups += 1
            status = self._validate_status(str(row["status"]))
            statuses[status] = statuses.get(status, 0) + 1
            pver = self._require_version(row["policy_version"], "stored policy")
            self._reject_future_version(pver, current_policy_version, "stored policy")
            raw_vver = row["verifier_version"]
            if type(raw_vver) is not int:
                raise ValueError("stored verifier version must be an integer")
            vver = raw_vver
            if status == "ready" and vver < 0:
                raise ValueError("ready group has an invalid verifier version")
            if vver >= 0:
                self._reject_future_version(vver, current_verifier_version, "stored verifier")
            policies.add(pver)
            if vver >= 0:
                verifiers.add(vver)
            for item in self._decode_payload(str(row["payload"])):
                generations += 1
                if status == "ready" and "verifier_version" in item:
                    item_vver = self._require_version(
                        item["verifier_version"], "stored experience verifier"
                    )
                    self._reject_future_version(
                        item_vver, current_verifier_version, "stored experience verifier"
                    )
                    verifiers.add(item_vver)
                if "reward" in item:
                    rewards.append(float(item["reward"]))
        max_policy_lag = 0
        if current_policy_version is not None and policies:
            max_policy_lag = max(max(0, current_policy_version - v) for v in policies)
        max_verifier_lag = 0
        if current_verifier_version is not None and verifiers:
            max_verifier_lag = max(max(0, current_verifier_version - v) for v in verifiers)
        return RVLReplaySnapshot(
            groups=groups,
            generations=generations,
            statuses=statuses,
            mean_reward=fmean(rewards) if rewards else None,
            policy_versions=tuple(sorted(policies)),
            verifier_versions=tuple(sorted(verifiers)),
            max_policy_lag=max_policy_lag,
            max_verifier_lag=max_verifier_lag,
        )

    def ready_experiences(
        self,
        *,
        current_policy_version: int,
        current_verifier_version: int,
        default_family: str = "rvl",
        trusted_default: bool = False,
        pass_threshold: float = 0.5,
    ) -> list[Experience]:
        current_policy_version = self._require_version(
            current_policy_version, "current_policy"
        )
        current_verifier_version = self._require_version(
            current_verifier_version, "current_verifier"
        )
        out: list[Experience] = []
        with self._connect() as db:
            rows = db.execute(
                """SELECT rowid,id,policy_version,payload,status,verifier_version
                   FROM groups WHERE status='ready' ORDER BY rowid"""
            ).fetchall()
        for row in rows:
            pver = self._require_version(row["policy_version"], "stored policy")
            self._reject_future_version(pver, current_policy_version, "stored policy")
            group_vver = self._require_version(
                row["verifier_version"], "stored verifier"
            )
            self._reject_future_version(
                group_vver, current_verifier_version, "stored verifier"
            )
            for index, item in enumerate(self._decode_payload(str(row["payload"]))):
                generation = dict(item.get("generation", item))
                if "prompt_id" not in generation or "prompt" not in generation or "response" not in generation:
                    raise ValueError("RVL generation missing prompt_id/prompt/response")
                gmeta = dict(generation.get("metadata") or {})
                vmeta = dict(item.get("metadata") or {})
                family = str(gmeta.get("family", vmeta.get("family", default_family)))
                task = Task(
                    id=str(generation["prompt_id"]),
                    prompt=str(generation["prompt"]),
                    family=family,
                    metadata={"rvl_group_id": str(row["id"]), **gmeta},
                )
                reward = float(item.get("reward", 0.0))
                verifier_version = self._require_version(
                    item.get("verifier_version", group_vver),
                    "stored experience verifier",
                )
                self._reject_future_version(
                    verifier_version,
                    current_verifier_version,
                    "stored experience verifier",
                )
                disagreement = float(vmeta.get("disagreement", 0.0))
                confidence = float(vmeta.get("confidence", 1.0 - min(1.0, disagreement)))
                trusted = bool(vmeta.get("trusted", trusted_default))
                attempt = Attempt(
                    task=task,
                    output=str(generation["response"]),
                    policy_id=str(gmeta.get("policy_id", f"rvl-policy-v{pver}")),
                    policy_version=pver,
                    created_step=int(row["rowid"]),
                    logprob=float(generation["logprob"]) if generation.get("logprob") is not None else None,
                    latency_ms=(float(generation["latency_s"]) * 1000.0) if generation.get("latency_s") is not None else None,
                    metadata={
                        "rvl_group_id": str(row["id"]),
                        "rvl_generation_index": index,
                        "token_count": generation.get("token_count"),
                        **gmeta,
                    },
                )
                verification = Verification(
                    score=reward,
                    passed=reward >= pass_threshold,
                    confidence=max(0.0, min(1.0, confidence)),
                    verifier_version=verifier_version,
                    verifier_name=str(vmeta.get("verifier_name", "rvl")),
                    disagreement=max(0.0, disagreement),
                    trusted=trusted,
                    metadata={"rvl_group_id": str(row["id"]), **vmeta},
                )
                out.append(
                    Experience(
                        attempt=attempt,
                        verification=verification,
                        policy_lag=max(0, current_policy_version - pver),
                        verifier_lag=max(0, current_verifier_version - verifier_version),
                        shift_score=0.0,
                    )
                )
        return out
