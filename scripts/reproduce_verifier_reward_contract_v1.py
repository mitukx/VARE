#!/usr/bin/env python3
"""Exercise the verifier-to-training reward boundary without model dependencies."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.types import Attempt, EvaluationReport, Task, Verification
from vare.verifiers import FunctionalAttemptVerifier, VerifierEnsemble, VerifierMember


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "verifier_reward_contract_v1.json"


class _Hooks:
    def __init__(self) -> None:
        self.training_rewards: list[float] = []

    def active_policy(self) -> tuple[str, int]:
        return "policy-0", 0

    async def rollout(self, task: Task, policy_id: str, policy_version: int, step: int) -> Attempt:
        return Attempt(task, "answer", policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id: str, experiences) -> str:
        for exp in experiences:
            reward = float(exp.verification.score)
            self.training_rewards.append(reward if math.isfinite(reward) else "NaN")
        return incumbent_id

    async def evaluate(self, policy_id: str) -> EvaluationReport:
        return EvaluationReport(policy_id, 0.5, n=1)

    async def promote(self, candidate_id: str) -> None:
        raise AssertionError("the fixture never creates a distinct candidate")

    async def discard(self, candidate_id: str) -> None:
        raise AssertionError("the fixture never creates a distinct candidate")


class _StructuredVerifier:
    name = "fixture"
    version = 1
    trusted = True

    def __init__(self, value: Verification) -> None:
        self.value = value

    async def verify(self, attempt: Attempt) -> Verification:
        return self.value


def _valid_verification(score: float = 0.75) -> Verification:
    return Verification(score, score >= 0.5, 1.0, 1, "fixture", trusted=True)


async def _case(case_id: str) -> dict[str, Any]:
    verification = _valid_verification()
    if case_id == "mutated_score_2":
        verification.score = 2.0
        verifier = FunctionalAttemptVerifier(lambda _: verification, name="functional")
    elif case_id == "nan_score":
        verification.score = math.nan
        verifier = _StructuredVerifier(verification)
    elif case_id == "confidence_out_of_range":
        verification.confidence = -0.1
        verifier = _StructuredVerifier(verification)
    elif case_id == "valid_control":
        verifier = _StructuredVerifier(verification)
    else:
        raise ValueError(f"unknown case: {case_id}")

    hooks = _Hooks()
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=VerifierEnsemble([VerifierMember(verifier)]),
        config=EngineConfig(
            replay_batch_size=1,
            promotion=PromotionConfig(min_eval_examples=1),
        ),
        seed=0,
    )
    error = None
    try:
        await loop.run_round([Task("task-1", "Say answer")], round_index=0)
    except Exception as exc:  # Captured as data so pre/post-fix behavior is comparable.
        error = f"{type(exc).__name__}: {exc}"
    return {"case": case_id, "training_rewards": hooks.training_rewards, "error": error}


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
    cases = ["mutated_score_2", "nan_score", "confidence_out_of_range", "valid_control"]
    results = [asyncio.run(_case(case_id)) for case_id in cases]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    expected = {
        "mutated_score_2": "rejected" if results[0]["error"] else "accepted_invalid_reward",
        "nan_score": "rejected" if results[1]["error"] else "accepted_invalid_reward",
        "confidence_out_of_range": "rejected" if results[2]["error"] else "accepted_invalid_reward",
        "valid_control": "valid_reward_reached_training" if results[3]["training_rewards"] == [0.75] else "control_failed",
    }
    payload = {
        "protocol_id": json.loads(PROTOCOL.read_text())["protocol_id"],
        "protocol_sha256": _sha256(PROTOCOL),
        "script_sha256": _sha256(Path(__file__)),
        "commit": commit,
        "results": results,
        "classification": expected,
        "acceptance_passed": all(
            expected[key] == "rejected"
            for key in ("mutated_score_2", "nan_score", "confidence_out_of_range")
        ) and expected["valid_control"] == "valid_reward_reached_training",
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if payload["acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
