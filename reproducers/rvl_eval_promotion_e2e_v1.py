#!/usr/bin/env python3
"""Reproduce the evaluation-to-promotion report-integrity regression.

Runs the same actual CapabilityLoop + RVLGRPOHooks flow with the historical
PromotionGate from 2572ac7 and with the current gate. The evaluator produces
real per-task reports; the adapter then simulates a report that retains n and
the aggregate but drops one of two task rows. This is a boundary-contract test,
not a model-quality experiment.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from vare import engine
from vare.promotion import PromotionGate as CurrentPromotionGate


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "2572ac7"


def _historical_gate():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_COMMIT}:src/vare/promotion.py"],
        cwd=ROOT,
        text=True,
    )
    module_name = "vare._historical_promotion_gate_e2e"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "vare"
    sys.modules[module_name] = module
    exec(compile(source, f"{BASELINE_COMMIT}:src/vare/promotion.py", "exec"), module.__dict__)
    return module.PromotionGate


def _historical_engine():
    source = subprocess.check_output(
        ["git", "show", "a4e0c3152150f16b60c42995a8c3f66d57ed09f6:src/vare/engine.py"],
        cwd=ROOT,
        text=True,
    )
    module_name = "vare._historical_engine_e2e"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "vare"
    sys.modules[module_name] = module
    exec(compile(source, "a4e0c31:src/vare/engine.py", "exec"), module.__dict__)
    return module.CapabilityLoop


def _test_fixture():
    sys.path.insert(0, str(ROOT / "tests"))
    import test_rvl_evaluation_promotion_e2e as fixture

    return fixture


def _run(gate_class, evidence_case):
    previous = engine.PromotionGate
    engine.PromotionGate = gate_class
    try:
        fixture = _test_fixture()
        loop, hooks, trainer, _backend = fixture._run_case(evidence_case)
        result = asyncio.run(
            loop.run_round([fixture.Task("train", "train prompt", "math")], round_index=0)
        )
        return {
            "decision_accepted": result.decision.accepted,
            "decision_reasons": list(result.decision.reasons),
            "incumbent_primary": result.incumbent_eval.primary,
            "candidate_primary": result.candidate_eval.primary,
            "incumbent_n": result.incumbent_eval.n,
            "candidate_n": result.candidate_eval.n,
            "incumbent_retained_task_scores": len(result.incumbent_eval.metadata["per_task_scores"]),
            "candidate_retained_task_scores": len(result.candidate_eval.metadata["per_task_scores"]),
            "active_policy": hooks.active_policy(),
            "trainer_state": trainer.snapshot_training_state(),
            "remaining_snapshots": sorted(hooks._states),
        }
    finally:
        engine.PromotionGate = previous


def _run_evaluator_exception(loop_class):
    fixture = _test_fixture()
    current_loop, hooks, trainer, _backend = fixture._run_case(
        "valid", misbind_candidate=True
    )
    loop = loop_class(
        hooks=hooks,
        verifier=current_loop.verifier,
        config=current_loop.config,
        seed=19,
    )
    try:
        asyncio.run(
            loop.run_round([fixture.Task("train", "train prompt", "math")], round_index=0)
        )
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError("misbound evaluator output must raise")
    return {
        "error": error,
        "active_policy": hooks.active_policy(),
        "trainer_state": trainer.snapshot_training_state(),
        "remaining_snapshots": sorted(hooks._states),
    }


def main():
    evidence = {
        "baseline_commit": BASELINE_COMMIT,
        "current_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                ROOT / "src/vare/engine.py",
                ROOT / "src/vare/promotion.py",
                ROOT / "src/vare/integrations/rvl_grpo.py",
                ROOT / "tests/test_rvl_evaluation_promotion_e2e.py",
            ]
        },
        "protocol": "two actual RVL eval rows; remove one retained task score while preserving n and primary; run CapabilityLoop decision and promotion/rollback",
        "historical_gate": _run(_historical_gate(), "underreported"),
        "current_gate": _run(CurrentPromotionGate, "underreported"),
        "valid_report_control_current": _run(CurrentPromotionGate, "valid"),
        "historical_engine_evaluation_exception": _run_evaluator_exception(_historical_engine()),
        "current_engine_evaluation_exception": _run_evaluator_exception(engine.CapabilityLoop),
        "limitations": [
            "CPU synthetic trainer/backend only; establishes integration behavior, not model capability.",
            "The injected report corruption represents an invalid evaluator boundary payload; it is not evidence of a production incident.",
        ],
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
