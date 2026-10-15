#!/usr/bin/env python3
"""Reproduce stale source-version admission on pinned VARE main.

Run this file from the pinned baseline checkout after copying it there from the
candidate branch. It intentionally uses the baseline adapter and test fixture;
it does not patch VARE code or perform model training.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


BASELINE_COMMIT = "e912938125b8a135347961801e84f624c2b833eb"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_rvl_reward_attestation import (  # noqa: E402
    _RecordingTrainer,
    _fixture,
    _manifest,
    _read,
    _root,
    _strict_rvl_hooks,
)


def main() -> None:
    baseline = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if baseline != BASELINE_COMMIT:
        raise SystemExit(
            f"expected pinned baseline {BASELINE_COMMIT}, found {baseline}"
        )

    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="vare-rvl-stale-baseline-") as directory:
        experiences = _read(_fixture(Path(directory)))
        manifest = _manifest(_root(), experiences)
        trainer = _RecordingTrainer()
        hooks = _strict_rvl_hooks(trainer, manifest)
        source_policy_versions = sorted(
            {experience.attempt.policy_version for experience in experiences}
        )
        source_verifier_versions = sorted(
            {experience.verification.verifier_version for experience in experiences}
        )
        signed = all(manifest.verified(experience) for experience in experiences)
        candidate = asyncio.run(hooks.train_candidate("policy-0", experiences))

    result = {
        "baseline_commit": baseline,
        "reproducer": "scripts/reproduce_rvl_direct_trainer_stale_source_v1.py",
        "reproducer_sha256": runner_sha256,
        "active_policy_version": 6,
        "active_verifier_version": 5,
        "comparison_max_policy_lag": 1,
        "comparison_max_verifier_lag": 1,
        "source_policy_versions": source_policy_versions,
        "source_verifier_versions": source_verifier_versions,
        "source_policy_lags": [6 - version for version in source_policy_versions],
        "source_verifier_lags": [5 - version for version in source_verifier_versions],
        "cached_lags_before_training": [
            [experience.policy_lag, experience.verifier_lag]
            for experience in experiences
        ],
        "signed_evidence_verified": signed,
        "candidate": candidate,
        "trainer_calls": trainer.train_calls,
        "interpretation": (
            "pinned main admitted authenticated source versions beyond both "
            "comparison lag caps"
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
