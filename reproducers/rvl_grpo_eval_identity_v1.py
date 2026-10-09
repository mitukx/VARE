#!/usr/bin/env python3
"""Reproduce duplicate evaluation-ID score loss in RVLGRPOHooks."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from types import ModuleType

from vare.integrations.rvl_grpo import RVLGRPOHooks
from vare.types import Task


BASELINE_REVISION = "31615b6"
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Generation:
    prompt_id: str
    prompt: str
    response: str
    logprob: float = -0.1
    token_count: int = 1
    latency_s: float = 0.001
    metadata: dict | None = None


class Trainer:
    def snapshot_training_state(self):
        return {"weight": 0}

    def restore_training_state(self, state):
        pass


class Backend:
    def __init__(self):
        self.calls = []

    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        assert n == 1
        self.calls.append((prompt_id, prompt))
        response = {"first prompt": "wrong", "second prompt": "right"}[prompt]
        return [Generation(prompt_id=prompt_id, prompt=prompt, response=response)]


def _hook_class_at(revision: str):
    source = subprocess.check_output(
        ["git", "show", f"{revision}:src/vare/integrations/rvl_grpo.py"],
        cwd=ROOT,
        text=True,
    )
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    source = source.replace("from ..types import", "from vare.types import")
    module_name = "vare_rvl_hooks_baseline"
    module = ModuleType(module_name)
    sys.modules[module_name] = module
    exec(compile(source, f"{revision}:src/vare/integrations/rvl_grpo.py", "exec"), module.__dict__)
    return module.RVLGRPOHooks, source_hash


def _hooks(hooks_cls, tasks, backend):
    return hooks_cls(
        backend=backend,
        trainer=Trainer(),
        eval_tasks=tasks,
        score_fn=lambda task, response: float(response == "right"),
        verified_generation_factory=lambda **kwargs: kwargs,
        generation_factory=Generation,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    duplicate_tasks = [
        Task("duplicate", "first prompt", "qa"),
        Task("duplicate", "second prompt", "qa"),
    ]

    old_class, old_source_hash = _hook_class_at(BASELINE_REVISION)
    old_backend = Backend()
    old_report = asyncio.run(_hooks(old_class, duplicate_tasks, old_backend).evaluate("policy-0"))
    expected_rows = [0.0, 1.0]  # Independent oracle: score each requested task separately.
    oracle = {"n": len(expected_rows), "primary": fmean(expected_rows)}

    fixed_backend = Backend()
    try:
        _hooks(RVLGRPOHooks, duplicate_tasks, fixed_backend)
        fixed = {"rejected": False, "backend_calls_before_rejection": len(fixed_backend.calls)}
    except ValueError as exc:
        fixed = {
            "rejected": "evaluation task IDs must be unique" in str(exc),
            "error": str(exc),
            "backend_calls_before_rejection": len(fixed_backend.calls),
        }

    unique_tasks = [
        Task("task-0", "first prompt", "qa"),
        Task("task-1", "second prompt", "qa"),
    ]
    control_backend = Backend()
    control = asyncio.run(_hooks(RVLGRPOHooks, unique_tasks, control_backend).evaluate("policy-0"))

    result = {
        "reproducer": "rvl_grpo_eval_identity_v1",
        "baseline_revision": BASELINE_REVISION,
        "baseline_source_sha256": old_source_hash,
        "oracle": oracle,
        "duplicate_id_case": {
            "requested_rows": len(duplicate_tasks),
            "baseline_backend_calls": len(old_backend.calls),
            "baseline_report_n": old_report.n,
            "baseline_primary": old_report.primary,
            "baseline_per_task_scores": old_report.metadata["per_task_scores"],
            "fixed_adapter": fixed,
        },
        "unique_id_negative_control": {
            "report_n": control.n,
            "primary": control.primary,
            "per_task_scores": control.metadata["per_task_scores"],
            "backend_calls": len(control_backend.calls),
            "matches_oracle": control.n == oracle["n"] and control.primary == oracle["primary"],
        },
        "claim_boundary": "Constructed CPU fixture on VARE's actual RVL evaluation adapter; no pretrained model or capability result.",
    }
    if old_report.n != 1 or old_report.primary != 1.0:
        raise AssertionError("pinned baseline failed to reproduce overwritten evaluation row")
    if not fixed["rejected"] or fixed["backend_calls_before_rejection"] != 0:
        raise AssertionError("current adapter did not reject duplicate IDs before generation")
    if not result["unique_id_negative_control"]["matches_oracle"]:
        raise AssertionError("unique-ID negative control disagrees with row-wise oracle")
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
