"""CPU-only data, cohort, and answer-target audit for Qwen3 DeepMath v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "protocols/qwen3_deepmath_grpo_math500_v1.lock.json"

import pyarrow.parquet as parquet
from math_verify import parse as reward_parse

from scripts.math500_grader_v2 import exact_match
from scripts.prompt_id_hash_v3 import canonical_prompt_id_sha256
from scripts.qwen3_deepmath_protocol_v1 import answer_target, cohort_rows, prompt_id
from scripts.validate_math500_study_v1 import (
    audit as audit_math_data,
    deduplicate_train_rows,
    select_splits,
)


def _target_is_verified(solution: str) -> bool:
    try:
        return bool(reward_parse(solution)) and exact_match(
            rf"\boxed{{{answer_target(solution)}}}", solution
        )
    except Exception:
        return False


def build_report(data_dir: Path) -> dict:
    base_audit = audit_math_data(data_dir)
    if base_audit.get("status") != "pass":
        raise ValueError("the existing MATH-500/DeepMath CPU audit did not pass")
    deepmath_path = data_dir / "deepmath-train.parquet"
    raw = parquet.read_table(deepmath_path, columns=["prompt", "solution"]).to_pylist()
    eligible, dedupe = deduplicate_train_rows(raw)
    legacy = select_splits([], eligible)
    retired = set(legacy["base_gate_ids"]) | set(legacy["train_ids"])
    gate, train, cohort_stats = cohort_rows(
        eligible, retired, is_training_target_valid=_target_is_verified
    )
    for row in gate + train:
        if not _target_is_verified(row["solution"]):
            raise AssertionError("selected training label failed independent self-grading")
    gate_ids = {prompt_id(row["prompt_key"]) for row in gate}
    train_ids = {prompt_id(row["prompt_key"]) for row in train}
    if gate_ids & train_ids or (gate_ids | train_ids) & retired:
        raise AssertionError("new cohorts overlap each other or the retired v7 selection")
    return {
        "status": "pass",
        "data": {
            "deepmath_train_sha256": base_audit["deepmath_train"]["sha256"],
            "math500_sha256": base_audit["math500"]["sha256"],
            "deepmath_rows": len(raw),
            "math500_rows": base_audit["math500"]["rows"],
            "exact_normalized_prompt_overlap": base_audit["cross_dataset"]["exact_normalized_prompt_overlap"],
        },
        "deduplication": {
            "raw_rows": dedupe["raw_rows"],
            "eligible_unique_prompts": dedupe["eligible_unique_prompts"],
            "duplicate_excess_rows_removed": dedupe["duplicate_excess_rows_deduplicated"],
            "conflicting_answer_prompts_excluded": dedupe["conflicting_answer_prompts_excluded"],
            "unparseable_duplicate_groups_excluded": dedupe["unparseable_reward_solution_prompts_excluded"],
        },
        "cohorts": cohort_stats,
        "legacy_exclusion": {
            "protocol_id": "qwen25_deepmath_grpo_math500_v7",
            "gate_ids_sha256": canonical_prompt_id_sha256(legacy["base_gate_ids"]),
            "planned_update_ids_sha256": canonical_prompt_id_sha256(legacy["train_ids"]),
            "excluded_prompt_count": len(retired),
        },
        "math500_split": base_audit["splits"]["selection_fingerprints"],
        "sft_targets": {
            "source": "math_verify.parse(solution), preferring its rendered answer string",
            "selected_train_count": len(train),
            "all_independently_self_graded": True,
            "maximum_target_characters": max(len(answer_target(row["solution"])) for row in train),
        },
        "limitations": [
            "exact normalized overlap does not detect semantic near-duplicates",
            "the audit cannot rule out model pretraining exposure",
            "MATH-500 is public and its confirmation partition is protocol-reserved rather than secret",
        ],
    }


def verify_locked_inputs(report: dict, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "qwen3_deepmath_grpo_math500_v1":
        raise ValueError("unexpected study protocol ID")
    expected = protocol["training_data"]["cohorts"]
    observed = report["cohorts"]
    for key in (
        "gate_count", "gate_ids_sha256", "gate_unparseable_candidates_skipped",
        "update_count", "update_ids_sha256", "update_unparseable_candidates_skipped",
        "retired_prompt_count", "cohorts_disjoint",
    ):
        if observed[key] != expected[key]:
            raise ValueError(f"protocol cohort mismatch: {key}")
    if report["sft_targets"]["selected_train_count"] != expected["update_count"]:
        raise ValueError("SFT answer-target row count differs from protocol")
    if not report["sft_targets"]["all_independently_self_graded"]:
        raise ValueError("a selected SFT target failed the independent self-check")
    if report["data"]["deepmath_train_sha256"] != protocol["training_data"]["sha256"]:
        raise ValueError("DeepMath source hash differs from protocol")
    if report["data"]["math500_sha256"] != protocol["evaluation_data"]["sha256"]:
        raise ValueError("MATH-500 source hash differs from protocol")
    if report["legacy_exclusion"]["gate_ids_sha256"] != protocol["training_data"]["legacy_exclusion"]["legacy_gate_ids_sha256"]:
        raise ValueError("retired legacy gate cohort hash differs from protocol")
    if report["legacy_exclusion"]["planned_update_ids_sha256"] != protocol["training_data"]["legacy_exclusion"]["legacy_update_ids_sha256"]:
        raise ValueError("retired legacy update cohort hash differs from protocol")
    for split in ("development", "confirmation"):
        split_key = "development_ids_sha256" if split == "development" else "confirmation_ids_sha256"
        if report["math500_split"][split_key] != protocol["evaluation"]["splits"][split]["ids_sha256"]:
            raise ValueError(f"MATH-500 {split} split fingerprint differs from protocol")
    return {"protocol_id": protocol["protocol_id"], "protocol_lock_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(), "locked_cohorts_and_splits_match": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    report = build_report(args.data_dir)
    report["protocol_audit"] = verify_locked_inputs(report, args.protocol)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
