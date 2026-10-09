#!/usr/bin/env python3
"""Audit v3 measurement gates separately from its post-gate runner error.

The RVL checkout and cached model are explicit inputs so the audit can run
from a clean checkout instead of relying on paths from the original host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/rvl_cpu_real_model_update_path_v3.lock.json"
RUNNER = ROOT / "scripts/run_rvl_cpu_real_model_update_path_v3.py"
ADAPTER = ROOT / "src/vare/integrations/rvl_grpo.py"
DEFAULT_BUNDLE = ROOT / "results/rvl-cpu-real-model-update-path-v3/run-1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reward(text: str, answer_key: str) -> float:
    match = re.match(r"^\s*([A-D])(?=$|[\s).,:;])", text)
    return float(bool(match and match.group(1) == answer_key))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--rvl-source",
        type=Path,
        required=True,
        help="RVL checkout at the revision recorded in the frozen protocol",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="local snapshot directory for the pinned model revision",
    )
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    rvl_source = args.rvl_source.resolve()
    model_path = args.model_path.resolve()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}

    def check(name: str, condition: bool) -> None:
        checks[name] = bool(condition)
        assert condition, name

    check("protocol_hash", summary["protocol_sha256"] == sha(PROTOCOL))
    check("runner_hash", sha(RUNNER) == protocol["runner_sha256"] == summary["runner_sha256"])
    check(
        "adapter_hash",
        sha(ADAPTER) == protocol["vare_adapter_sha256"] == summary["vare_adapter_sha256"],
    )

    rvl_hashes = {
        path.relative_to(rvl_source).as_posix(): sha(path)
        for path in sorted((rvl_source / "src/rvl_systems").rglob("*.py"))
    }
    check("rvl_source_hashes", rvl_hashes == protocol["rvl_source_files_sha256"])
    model_hashes = {
        name: sha(model_path / name) for name in protocol["model"]["files_sha256"]
    }
    check("model_hashes", model_hashes == protocol["model"]["files_sha256"])
    check(
        "retained_post_gate_cleanup_failure",
        summary["status"] == "failed"
        and summary["error_type"] == "UnboundLocalError"
        and "loaded_tokenizer" in summary["error"],
    )

    groups = summary["responses"]
    tasks = protocol["task"]["groups"]
    check(
        "groups_are_protocol_prefix",
        [group["prompt_id"] for group in groups]
        == [task["prompt_id"] for task in tasks[: len(groups)]],
    )

    replayed_rewards: list[list[float]] = []
    for group, task in zip(groups, tasks, strict=False):
        values = [reward(row["text"], task["answer"]) for row in group["responses"]]
        check(
            f"{task['prompt_id']}_reward_replay",
            values == group["rewards"]
            and group["question"] == task["question"]
            and group["answer_key"] == task["answer"],
        )
        replayed_rewards.append(values)
        if len(set(values)) > 1:
            break

    selected_index = next(
        index for index, values in enumerate(replayed_rewards) if len(set(values)) > 1
    )
    selected_values = replayed_rewards[selected_index]
    check(
        "first_mixed_group_selected",
        summary["selected_prompt_id"] == tasks[selected_index]["prompt_id"]
        and summary["selected_rewards"] == selected_values
        and all(len(set(values)) == 1 for values in replayed_rewards[:selected_index]),
    )
    mean = sum(selected_values) / len(selected_values)
    variance = sum((value - mean) ** 2 for value in selected_values) / len(selected_values)
    eps = protocol["update"]["advantage_epsilon"]
    clip = protocol["update"]["advantage_clip"]
    advantages = [
        max(-clip, min(clip, (value - mean) / math.sqrt(variance + eps)))
        for value in selected_values
    ]
    check(
        "advantages_recomputed",
        all(
            abs(actual - expected) < 1e-10
            for actual, expected in zip(advantages, summary["received_advantages"], strict=True)
        ),
    )
    check(
        "one_optimizer_step_and_finite_nonzero_gradients",
        summary["optimizer_step_calls"] == 1
        and summary["finite_gradient_tensor_count"] > 0
        and summary["nonzero_gradient_tensor_count"] > 0,
    )
    check(
        "candidate_weights_changed",
        summary["changed_parameter_tensor_count"] > 0
        and summary["max_abs_parameter_delta"] > 0,
    )
    check(
        "incumbent_model_optimizer_rng_mode_restored",
        summary["incumbent_restored_exact"]
        and summary["incumbent_rng_restored_exact"]
        and summary["incumbent_training_mode_restored_exact"],
    )
    check("tokenizer_ids_match_on_frozen_prompts", summary["tokenizer_prompt_roundtrip_exact"])
    check(
        "runner_recorded_model_fingerprint_roundtrip",
        summary["roundtrip_exact"] and bool(summary["reloaded_parameter_fingerprint"]),
    )
    check(
        "cpu_resource_gates",
        summary["device"] == "cpu"
        and summary["mps_available_but_disabled"]
        and summary["wall_seconds"] <= protocol["runtime"]["max_wall_seconds"]
        and summary["peak_rss_bytes"] <= protocol["runtime"]["max_peak_rss_bytes"],
    )

    result = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha(PROTOCOL),
        "auditor_sha256": sha(Path(__file__).resolve()),
        "summary_sha256": sha(bundle / "summary.json"),
        "checks": checks,
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "measurement_gate_evidence": (
            "recorded update, restore, tokenizer/model round-trip, and resource fields "
            "are present and satisfy their frozen thresholds"
        ),
        "runner_terminal_status": (
            "failed after these fields were recorded, during redundant cleanup of an "
            "already deleted tokenizer variable"
        ),
        "audit_type": "separate same-host author replay, not external reproduction",
        "limits": [
            "The model fingerprint round-trip is checked against the runner's retained "
            "boolean and fingerprint string; the deleted temporary checkpoint is not "
            "independently available for byte-level reconstruction.",
            "The audit verifies retained claims and hashes; it does not rerun the optimizer.",
            "This is execution feasibility only; no task-success or model-capability effect is measured.",
        ],
    }
    (bundle / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
