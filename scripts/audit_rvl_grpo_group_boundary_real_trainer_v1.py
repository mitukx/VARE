#!/usr/bin/env python3
"""Independently replay the frozen group-advantage and update-boundary checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/rvl_grpo_group_boundary_real_trainer_v1.lock.json"
RUNNER = ROOT / "scripts/run_rvl_grpo_group_boundary_real_trainer_v1.py"
ADAPTER = ROOT / "src/vare/integrations/rvl_grpo.py"
DEFAULT_BUNDLE = ROOT / "results/rvl-grpo-group-boundary-real-trainer-v1/run-1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(rewards: list[float], eps: float, clip: float) -> list[float]:
    mean = fmean(rewards)
    variance = fmean((reward - mean) ** 2 for reward in rewards)
    scale = math.sqrt(variance + eps)
    return [max(-clip, min(clip, (reward - mean) / scale)) for reward in rewards]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--rvl-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    rvl_source = args.rvl_source.resolve()
    model_path = args.model_path.resolve()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}

    def check(name: str, condition: bool) -> None:
        checks[name] = bool(condition)

    check("protocol_hash", summary["protocol_sha256"] == sha(PROTOCOL))
    check("runner_hash", sha(RUNNER) == protocol["runner_sha256"] == summary["runner_sha256"])
    check("auditor_hash", sha(Path(__file__).resolve()) == protocol["auditor_sha256"])
    check("adapter_hash", sha(ADAPTER) == protocol["vare_adapter_sha256"] == summary["vare_adapter_sha256"])
    rvl_hashes = {
        path.relative_to(rvl_source).as_posix(): sha(path)
        for path in sorted((rvl_source / "src/rvl_systems").rglob("*.py"))
    }
    check("rvl_source_hashes", rvl_hashes == protocol["rvl_source_files_sha256"])
    model_hashes = {
        name: sha(model_path / name) for name in protocol["model"]["files_sha256"]
    }
    check("model_hashes", model_hashes == protocol["model"]["files_sha256"])

    sampling = protocol["sampling"]
    rows = summary["responses"]
    rewards = [value for group in sampling["group_rewards"] for value in group]
    check("completed_cpu_run", summary["status"] == "completed" and summary["device"] == "cpu")
    check("runner_frozen_gates_all_pass", bool(summary["frozen_gate_checks"]) and all(summary["frozen_gate_checks"].values()))
    check("sample_and_reward_cardinality", len(rows) == len(rewards) == sampling["samples"])
    check(
        "frozen_prompt_identity",
        all(row["prompt_id"] == sampling["prompt_id"] and row["prompt"] == sampling["rendered_prompt"] for row in rows),
    )
    check("frozen_reward_assignment", [row["reward"] for row in rows] == rewards)
    check(
        "positive_response_diversity_gate",
        rows[1]["response_token_ids"]
        and all(rows[1]["response_token_ids"] != rows[index]["response_token_ids"] for index in (0, 2, 3)),
    )
    check("single_frozen_incumbent", len({arm["initial_parameter_fingerprint"] for arm in summary["arms"].values()}) == 1 and next(iter(summary["arms"].values()))["initial_parameter_fingerprint"] == summary["incumbent_parameter_fingerprint"])

    eps = float(protocol["update"]["advantage_epsilon"])
    clip = float(protocol["update"]["advantage_clip"])
    native_expected = normalized(rewards, eps, clip)
    group_expected = normalized(rewards[:4], eps, clip) + normalized(rewards[4:], eps, clip)
    native = summary["arms"]["native_prompt_grouping"]
    control = summary["arms"]["adapter_single_group_control"]
    treatment = summary["arms"]["adapter_two_group_treatment"]
    check("native_advantages_replayed", all(abs(x-y) <= 1e-10 for x, y in zip(native_expected, native["consumed_advantages"], strict=True)))
    check("same_group_control_advantages", all(abs(x-y) <= 1e-10 for x, y in zip(native_expected, control["consumed_advantages"], strict=True)))
    check("treatment_group_advantages", all(abs(x-y) <= 1e-10 for x, y in zip(group_expected, treatment["consumed_advantages"], strict=True)))

    check("same_group_control_matches_native_update", control["post_step_parameter_fingerprint"] == native["post_step_parameter_fingerprint"] and control["max_abs_parameter_difference_vs_native"] == 0.0)
    check("two_group_treatment_changes_update", treatment["post_step_parameter_fingerprint"] != native["post_step_parameter_fingerprint"] and treatment["max_abs_parameter_difference_vs_native"] > 0.0)
    for name, arm in summary["arms"].items():
        check(f"{name}_one_update", arm["train_step_calls"] == 1 and arm["optimizer_step_calls"] == 1)
        check(f"{name}_same_initial_state", arm["initial_state_matched"] and arm["initial_parameter_fingerprint"] == summary["incumbent_parameter_fingerprint"])
        check(
            f"{name}_finite_nonzero_gradients",
            arm["gradient_tensor_count"] > 0
            and arm["finite_gradient_tensor_count"] == arm["gradient_tensor_count"]
            and arm["nonzero_gradient_tensor_count"] > 0,
        )
        check(f"{name}_weights_changed", arm["changed_parameter_tensor_count"] > 0 and arm["max_abs_parameter_delta"] > 0.0)
        check(f"{name}_incumbent_restored", arm["incumbent_restored_exact"])
    for name, control in summary["negative_controls"].items():
        check(
            f"{name}_rejected_before_mutation",
            control["rejected"]
            and control["train_step_calls"] == 0
            and control["optimizer_step_calls"] == 0
            and control["incumbent_unchanged"],
        )
    runtime = protocol["runtime"]
    check("resource_caps", summary["wall_seconds"] <= runtime["max_wall_seconds"] and summary["peak_rss_bytes"] <= runtime["max_peak_rss_bytes"])

    audit = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha(PROTOCOL),
        "auditor_sha256": sha(Path(__file__).resolve()),
        "summary_sha256": sha(bundle / "summary.json"),
        "checks": checks,
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "decision": "pass" if all(checks.values()) else "fail",
        "native_advantages_replayed": native_expected,
        "groupwise_advantages_replayed": group_expected,
        "primary_difference_max_abs": treatment["max_abs_parameter_difference_vs_native"],
        "audit_scope": "same-host separate-code replay; no optimizer rerun or capability evaluation",
        "limitations": [
            "The auditor verifies parameter fingerprints and recorded per-arm update summaries, not retained full model checkpoints.",
            "Rewards are an intentionally synthetic group-boundary witness; update divergence establishes integration sensitivity only.",
            "No held-out task success, policy quality, external replication, or real verifier correctness was measured.",
        ],
    }
    (bundle / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
