#!/usr/bin/env python3
"""Independent, study-specific audit for the frozen ARC GRPO/SFT comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/qwen_arc_grpo_sft_comparison_v1.lock.json"


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protocol_digest(lock: dict[str, Any]) -> str:
    body = {key: value for key, value in lock.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_label(text: str) -> str | None:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) > 1:
            value = "\n".join(lines[1:-1]).strip()
    value = re.sub(r"(?i)^(?:answer:|the\s+answer\s+is:|the\s+correct\s+answer\s+is:?)\s*", "", value)
    match = re.fullmatch(r"\(?([A-Da-d])\)?[.)]?", value.strip())
    return match.group(1).upper() if match else None


def choice_labels(row: dict[str, Any]) -> list[str]:
    data = row["choices"]
    labels = data["label"] if isinstance(data, dict) else [item["label"] for item in data]
    return [str(label) for label in labels]


def select(rows: list[dict[str, Any]], salt: str, n: int, excluded: set[str]) -> list[str]:
    eligible = [row for row in rows if choice_labels(row) == ["A", "B", "C", "D"] and str(row["id"]) not in excluded]
    ranked = sorted(eligible, key=lambda row: hashlib.sha256(f"{salt}|{row['id']}".encode()).hexdigest())
    return [str(row["id"]) for row in ranked[:n]]


def assert_protocol(lock: dict[str, Any]) -> int:
    checks = 0
    if protocol_digest(lock) != lock.get("sha256"):
        raise AssertionError("protocol canonical digest mismatch")
    checks += 1
    if lock["protocol_id"] != "qwen_arc_grpo_sft_comparison_v1":
        raise AssertionError("protocol ID mismatch")
    checks += 1
    if digest_file(Path(__file__)) != lock["source_hashes"]["auditor"]:
        raise AssertionError("auditor source hash mismatch")
    checks += 1
    for relpath, key in (("scripts/run_qwen_arc_grpo_sft_comparison_v1.py", "runner"),
                         ("src/vare/integrations/rvl_grpo.py", "vare_adapter")):
        if digest_file(ROOT / relpath) != lock["source_hashes"][key]:
            raise AssertionError(f"source hash mismatch: {relpath}")
        checks += 1
    if len(lock["selection"]["train_ids"]) != 32 or len(set(lock["selection"]["train_ids"])) != 32:
        raise AssertionError("training cohort cardinality/uniqueness mismatch")
    checks += 1
    if len(lock["selection"]["validation_ids"]) != 115 or len(set(lock["selection"]["validation_ids"])) != 115:
        raise AssertionError("validation cohort cardinality/uniqueness mismatch")
    checks += 1
    if set(lock["selection"]["validation_ids"]) & set(lock["selection"]["excluded_validation_ids"]):
        raise AssertionError("validation cohort contains a consumed item")
    checks += 1
    old_train: set[str] = set()
    old_validation: set[str] = set()
    for protocol_id in lock["parent_protocols"]:
        parent_path = ROOT / "protocols" / f"{protocol_id}.lock.json"
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        if protocol_id.startswith("qwen_arc_grpo_update_smoke_"):
            old_train.update(map(str, parent["selection"]["train_ids"]))
            old_train.update(map(str, parent["selection"].get("excluded_train_ids", [])))
            old_validation.update(map(str, parent["selection"]["validation_ids"]))
            old_validation.update(map(str, parent["selection"]["excluded_validation_ids"]))
    if not old_train <= set(lock["selection"]["excluded_train_ids"]):
        raise AssertionError("a prior smoke train item was not excluded")
    if not old_validation <= set(lock["selection"]["excluded_validation_ids"]):
        raise AssertionError("a prior smoke/base-gate validation item was not excluded")
    checks += 2
    return checks


def audit_locked_samples(lock: dict[str, Any], train_path: Path, val_path: Path) -> int:
    import pandas as pd
    if digest_file(train_path) != lock["dataset"]["train_sha256"] or digest_file(val_path) != lock["dataset"]["validation_sha256"]:
        raise AssertionError("dataset file digest mismatch")
    train_rows = pd.read_parquet(train_path).to_dict(orient="records")
    val_rows = pd.read_parquet(val_path).to_dict(orient="records")
    selection = lock["selection"]
    if select(train_rows, selection["train_salt"], 32, set(selection["excluded_train_ids"])) != selection["train_ids"]:
        raise AssertionError("independently reconstructed training IDs differ")
    if select(val_rows, selection["validation_salt"], 115, set(selection["excluded_validation_ids"])) != selection["validation_ids"]:
        raise AssertionError("independently reconstructed validation IDs differ")
    if sum(choice_labels(row) == ["A", "B", "C", "D"] and str(row["id"]) not in set(selection["excluded_validation_ids"]) for row in val_rows) != 115:
        raise AssertionError("fresh validation cohort size differs from frozen claim")
    return 3


def audit_run(lock: dict[str, Any], run_dir: Path, train_path: Path, val_path: Path) -> dict[str, Any]:
    import pandas as pd
    checks = assert_protocol(lock)
    if digest_file(train_path) != lock["dataset"]["train_sha256"] or digest_file(val_path) != lock["dataset"]["validation_sha256"]:
        raise AssertionError("dataset file digest mismatch")
    checks += 1
    train_rows = pd.read_parquet(train_path).to_dict(orient="records")
    val_rows = pd.read_parquet(val_path).to_dict(orient="records")
    selection = lock["selection"]
    if select(train_rows, selection["train_salt"], 32, set(selection["excluded_train_ids"])) != selection["train_ids"]:
        raise AssertionError("independently reconstructed training IDs differ")
    checks += 1
    if select(val_rows, selection["validation_salt"], 115, set(selection["excluded_validation_ids"])) != selection["validation_ids"]:
        raise AssertionError("independently reconstructed validation IDs differ")
    checks += 1
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise AssertionError(f"run status is {summary.get('status')!r}, not complete")
    checks += 1
    if summary.get("protocol_sha256") != digest_file(LOCK):
        raise AssertionError("summary references another protocol")
    checks += 1
    answer = {str(row["id"]): str(row["answerKey"]).upper() for row in val_rows}
    base = {item["task_id"]: item for item in summary["base_eval"]}
    if set(base) != set(selection["validation_ids"]):
        raise AssertionError("base evaluation ID set mismatch")
    checks += 1

    def verify_eval(items: list[dict[str, Any]], name: str) -> dict[str, bool]:
        nonlocal checks
        result = {}
        if len(items) != 115:
            raise AssertionError(f"{name}: expected 115 evaluation records")
        for item in items:
            qid = item["task_id"]
            if qid not in answer or item["answer_key"] != answer[qid]:
                raise AssertionError(f"{name}: answer key/data mismatch for {qid}")
            parsed = parse_label(item["raw_generation"])
            if parsed != item["parsed_label"] or (parsed == item["answer_key"]) != bool(item["exact"]):
                raise AssertionError(f"{name}: parsed/exact mismatch for {qid}")
            if not item.get("generated_token_ids"):
                raise AssertionError(f"{name}: missing generated token IDs for {qid}")
            result[qid] = bool(item["exact"])
            checks += 1
        if len(result) != 115 or set(result) != set(selection["validation_ids"]):
            raise AssertionError(f"{name}: duplicate or mismatched task IDs")
        checks += 1
        return result

    base_ok = verify_eval(summary["base_eval"], "base")
    train_answers = {str(row["id"]): str(row["answerKey"]).upper() for row in train_rows}
    train_answers = {k: v for k, v in train_answers.items() if k in selection["train_ids"]}
    rollout_summary: dict[str, Any] = {}
    for seed in lock["training"]["seeds"]:
        seed_key = str(seed)
        groups = summary["rollouts_by_seed"][seed_key]
        if [group["task_id"] for group in groups] != selection["train_ids"]:
            raise AssertionError(f"{seed_key}: rollout task IDs/order mismatch")
        checks += 1
        positive = 0
        total_tokens = 0
        for group in groups:
            if len(group["members"]) != 4 or len(group["rewards"]) != 4:
                raise AssertionError(f"{seed_key}/{group['task_id']}: incomplete group")
            if group["answer_key"] != train_answers[group["task_id"]]:
                raise AssertionError("rollout answer key disagrees with train parquet")
            for member in group["members"]:
                parsed = parse_label(member["raw_generation"])
                reward = float(parsed == train_answers[group["task_id"]])
                if parsed != member["parsed_label"] or reward != member["reward"]:
                    raise AssertionError("rollout parser/reward mismatch")
                if len(member["response_token_ids"]) != len(member["response_token_logprobs"]):
                    raise AssertionError("rollout token/logprob length mismatch")
                total_tokens += len(member["response_token_ids"])
                positive += int(reward)
                checks += 1
        readiness = summary["training_readiness"][seed_key]
        if readiness["positive_traces"] != positive or positive == 0 or readiness["mixed_groups"] == 0:
            raise AssertionError(f"{seed_key}: readiness gate mismatch/failure")
        checks += 1
        rollout_summary[seed_key] = {"positive_traces": positive, "rollout_response_tokens": total_tokens}

    per_seed = {}
    grpo_exact, sft_exact = {}, {}
    for seed in lock["training"]["seeds"]:
        sk = str(seed)
        arms = summary["arms"]
        for arm_name in ("grpo", "success_trace_sft"):
            arm = arms[arm_name][sk]
            if arm["optimizer_step_calls"] != 1 or arm["gradient_tensor_count"] <= 0:
                raise AssertionError(f"{arm_name}/{sk}: optimizer/gradient record missing")
            if arm["finite_gradient_tensor_count"] != arm["gradient_tensor_count"] or arm["nonzero_gradient_tensor_count"] <= 0:
                raise AssertionError(f"{arm_name}/{sk}: invalid gradient check")
            if arm["candidate_parameter_fingerprint"] == summary["base_model_fingerprint"]:
                raise AssertionError(f"{arm_name}/{sk}: candidate equals base")
            if not arm["tokenizer_roundtrip_exact"] or len(arm["eval"]) != 115:
                raise AssertionError(f"{arm_name}/{sk}: reload/evaluation gate mismatch")
            vals = verify_eval(arm["eval"], f"{arm_name}/{sk}")
            if arm_name == "grpo":
                grpo_exact[sk] = vals
            else:
                sft_exact[sk] = vals
            checks += 3
        per_seed[sk] = {
            "grpo_accuracy": sum(grpo_exact[sk].values()) / 115,
            "sft_accuracy": sum(sft_exact[sk].values()) / 115,
        }

    # Independent reconstruction of the declared task-stratified, seed-conditional bootstrap.
    answer_by_id = {qid: answer[qid] for qid in selection["validation_ids"]}
    strata = [[qid for qid in selection["validation_ids"] if answer_by_id[qid] == k] for k in "ABCD"]
    rng = random.Random(lock["analysis"]["bootstrap_seed"])
    draws_gs, draws_gb = [], []
    for _ in range(lock["analysis"]["bootstrap_replicates"]):
        sample = [qid for ids in strata for qid in rng.choices(ids, k=len(ids))]
        draws_gs.append(sum(sum(grpo_exact[str(seed)][qid] - sft_exact[str(seed)][qid] for qid in sample) / len(sample)
                             for seed in lock["training"]["seeds"]) / 3)
        draws_gb.append(sum(sum(grpo_exact[str(seed)][qid] - base_ok[qid] for qid in sample) / len(sample)
                             for seed in lock["training"]["seeds"]) / 3)

    def interval(values: list[float]) -> list[float]:
        values.sort()
        return [values[math.floor(.025 * (len(values) - 1))], values[math.floor(.975 * (len(values) - 1))]]

    means = {k: sum(v.values()) / 3 for k, v in (("grpo", grpo_exact), ("sft", sft_exact))}
    base_acc = sum(base_ok.values()) / 115
    observed = {
        "primary_grpo_minus_sft": means["grpo"] - means["sft"],
        "primary_grpo_minus_sft_task_bootstrap_95_ci": interval(draws_gs),
        "secondary_grpo_minus_base": means["grpo"] - base_acc,
        "secondary_grpo_minus_base_task_bootstrap_95_ci": interval(draws_gb),
    }
    reported = summary["analysis"]
    for key, val in observed.items():
        if isinstance(val, list):
            if any(abs(x - y) > 1e-12 for x, y in zip(val, reported[key], strict=True)):
                raise AssertionError(f"independent bootstrap mismatch in {key}")
        elif abs(val - reported[key]) > 1e-12:
            raise AssertionError(f"independent metric mismatch in {key}")
        checks += 1
    if summary["analysis"]["base_accuracy"] != base_acc:
        raise AssertionError("reported base accuracy mismatch")
    checks += 1
    return {
        "status": "pass", "checks": checks,
        "protocol_sha256": digest_file(LOCK), "run_summary_sha256": digest_file(run_dir / "summary.json"),
        "observed": observed, "per_seed": per_seed, "training_readiness": rollout_summary,
        "auditor_limitations": [
            "same-host independent implementation; not outside human reproduction",
            "candidate checkpoint files were temporary and deleted after reload; recorded per-file hashes cannot be rehashed",
            "task bootstrap is conditional on three locked training seeds and does not estimate seed-population uncertainty",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol-only", action="store_true")
    ap.add_argument("--train-parquet", type=Path)
    ap.add_argument("--validation-parquet", type=Path)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if args.protocol_only:
        count = assert_protocol(lock)
        if args.train_parquet and args.validation_parquet:
            count += audit_locked_samples(lock, args.train_parquet.resolve(), args.validation_parquet.resolve())
        result = {"status": "pass", "protocol_checks": count,
                  "protocol_sha256": digest_file(LOCK), "dataset_sample_audit": bool(args.train_parquet and args.validation_parquet)}
    else:
        if not all((args.run_dir, args.train_parquet, args.validation_parquet, args.output)):
            ap.error("run audit requires --run-dir, --train-parquet, --validation-parquet, and --output")
        result = audit_run(lock, args.run_dir.resolve(), args.train_parquet.resolve(), args.validation_parquet.resolve())
        args.output.resolve().write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"AUDIT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
