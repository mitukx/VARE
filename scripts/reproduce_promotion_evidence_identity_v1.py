#!/usr/bin/env python3
"""Reproduce promotion decisions with missing slices and mismatched policy IDs."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.promotion import PromotionGate
from vare.types import Attempt, EvaluationReport, Task, Verification
from vare.verifiers import VerifierEnsemble, VerifierMember


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "promotion_evidence_identity_v1.json"


class _Verifier:
    name = "valid"
    version = 1
    trusted = True

    async def verify(self, attempt: Attempt) -> Verification:
        return Verification(0.75, True, 1.0, 1, self.name, trusted=True)


class _Hooks:
    def __init__(self, *, wrong_candidate_id: bool = False) -> None:
        self.wrong_candidate_id = wrong_candidate_id
        self.promoted: list[str] = []
        self.discarded: list[str] = []

    def active_policy(self) -> tuple[str, int]:
        return "p0", 0

    async def rollout(self, task: Task, policy_id: str, policy_version: int, step: int) -> Attempt:
        return Attempt(task, "answer", policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id: str, experiences) -> str:
        return "p1"

    async def evaluate(self, policy_id: str) -> EvaluationReport:
        report_id = "p0" if self.wrong_candidate_id and policy_id == "p1" else policy_id
        return EvaluationReport(report_id, 0.8 if policy_id == "p1" else 0.5,
                                {"hard": 0.4, "easy": 0.7}, n=10)

    async def promote(self, candidate_id: str) -> None:
        self.promoted.append(candidate_id)

    async def discard(self, candidate_id: str) -> None:
        self.discarded.append(candidate_id)


async def _engine_case(*, wrong_candidate_id: bool) -> dict[str, Any]:
    hooks = _Hooks(wrong_candidate_id=wrong_candidate_id)
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=VerifierEnsemble([VerifierMember(_Verifier())]),
        config=EngineConfig(
            replay_batch_size=1,
            promotion=PromotionConfig(min_primary_gain=0.01, min_eval_examples=1),
        ),
        seed=0,
    )
    result = await loop.run_round([Task("task", "answer")], round_index=0)
    return {
        "accepted": result.decision.accepted,
        "reasons": list(result.decision.reasons),
        "promoted": hooks.promoted,
        "discarded": hooks.discarded,
    }


def _slice_omission_case() -> dict[str, Any]:
    gate = PromotionGate(PromotionConfig(min_primary_gain=0.01, min_eval_examples=1))
    incumbent = EvaluationReport("p0", 0.5, {"hard": 0.4, "easy": 0.7}, n=10)
    candidate = EvaluationReport("p1", 0.8, {"easy": 0.7}, n=10)
    decision = gate.decide(incumbent, candidate)
    return {"accepted": decision.accepted, "reasons": list(decision.reasons)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    omission = _slice_omission_case()
    wrong_policy = asyncio.run(_engine_case(wrong_candidate_id=True))
    control = asyncio.run(_engine_case(wrong_candidate_id=False))
    observed = {
        "candidate_omits_slice": "accepted" if omission["accepted"] else "rejected",
        "candidate_report_wrong_policy": "promoted" if wrong_policy["promoted"] else "rejected",
        "matching_control": "accepted" if control["promoted"] == ["p1"] else "rejected",
    }
    payload = {
        "protocol_id": json.loads(PROTOCOL.read_text())["protocol_id"],
        "protocol_sha256": _sha256(PROTOCOL),
        "script_sha256": _sha256(Path(__file__)),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "results": {
            "candidate_omits_slice": omission,
            "candidate_report_wrong_policy": wrong_policy,
            "matching_control": control,
        },
        "classification": observed,
        "acceptance_passed": observed == {
            "candidate_omits_slice": "rejected",
            "candidate_report_wrong_policy": "rejected",
            "matching_control": "accepted",
        },
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if payload["acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
