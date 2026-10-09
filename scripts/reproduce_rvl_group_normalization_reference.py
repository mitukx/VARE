#!/usr/bin/env python3
"""Execute the pinned RVL GRPO normalizer on the locked group-collision fixtures.

This downloads one immutable, small source file and runs its exact
``compute_group_advantages`` implementation with stub types for unused trainer
objects. It performs no model inference or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from types import ModuleType, SimpleNamespace


RVL_REVISION = "c7e646b043cb56e5ea3c2623bb8a61e065451f72"
RVL_SOURCE_URL = (
    "https://raw.githubusercontent.com/mitukx/Recursive-Verification-Lag/"
    f"{RVL_REVISION}/src/rvl_systems/grpo.py"
)
RVL_SOURCE_SHA256 = "a2ef6217385a206f5e6c0f43fb40e900ae3bf1b2066a1c8d1ab1878fd25d724d"


def _load_pinned_normalizer(source: bytes):
    package = ModuleType("rvl_systems")
    package.__path__ = []
    backends = ModuleType("rvl_systems.backends")
    backends.ToyTabularBackend = type("ToyTabularBackend", (), {})
    types_module = ModuleType("rvl_systems.types")
    types_module.TrainRecord = lambda **fields: SimpleNamespace(**fields)
    types_module.VerifiedGeneration = type("VerifiedGeneration", (), {})
    sys.modules.update(
        {
            "rvl_systems": package,
            "rvl_systems.backends": backends,
            "rvl_systems.types": types_module,
        }
    )
    module = ModuleType("rvl_systems.grpo")
    module.__package__ = "rvl_systems"
    sys.modules["rvl_systems.grpo"] = module
    exec(compile(source, RVL_SOURCE_URL, "exec"), module.__dict__)
    return module.compute_group_advantages


def _samples(rewards: list[float], prompt_ids: list[str]):
    return [
        SimpleNamespace(
            generation=SimpleNamespace(
                prompt_id=prompt_id,
                response=f"response-{index}",
                logprob=-0.1,
            ),
            reward=reward,
        )
        for index, (prompt_id, reward) in enumerate(zip(prompt_ids, rewards, strict=True))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = urllib.request.urlopen(RVL_SOURCE_URL, timeout=20).read()
    source_hash = hashlib.sha256(source).hexdigest()
    if source_hash != RVL_SOURCE_SHA256:
        raise SystemExit(f"pinned RVL source hash mismatch: {source_hash}")
    compute_group_advantages = _load_pinned_normalizer(source)

    fixtures = {
        "frozen_v1_constant_group_witness": [0.0, 0.0, 1.0, 1.0],
        "supplemental_nonconstant_group_witness": [0.0, 1.0, 0.75, 1.0],
    }
    prompt_ids = ["same-task"] * 4
    results = {}
    for name, rewards in fixtures.items():
        records = compute_group_advantages(
            _samples(rewards, prompt_ids), eps=1e-6, clip=5.0
        )
        results[name] = {
            "rewards": rewards,
            "prompt_ids": prompt_ids,
            "prompt_merged_advantages": [record.advantage for record in records],
        }

    report = {
        "rvl_revision": RVL_REVISION,
        "source_url": RVL_SOURCE_URL,
        "source_sha256": source_hash,
        "method": "Executed pinned RVL compute_group_advantages; no training or model inference.",
        "fixtures": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
