#!/usr/bin/env python3
"""Independently reconstruct the frozen ARC GRPO update-smoke evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/qwen_arc_grpo_update_smoke_v2.lock.json"
RUNNER = ROOT / "scripts/run_qwen_arc_grpo_update_smoke_v2.py"
ADAPTER = ROOT / "src/vare/integrations/rvl_grpo.py"
SYSTEM = "Answer the science question by choosing one of the listed options. Reply with exactly one option label: A, B, C, or D. Do not explain."


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(obj: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in obj.items() if k != "sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def choices(row: dict) -> list[tuple[str, str]]:
    value = row["choices"]
    pairs = zip(value["label"], value["text"], strict=True) if isinstance(value, dict) else ((x["label"], x["text"]) for x in value)
    return [(str(a), str(b)) for a, b in pairs]


def select(rows: list[dict], salt: str, n: int) -> list[dict]:
    eligible = [r for r in rows if [a for a, _ in choices(r)] == ["A", "B", "C", "D"]]
    return sorted(eligible, key=lambda r: hashlib.sha256(f"{salt}|{r['id']}".encode()).hexdigest())[:n]


def render(tokenizer, row: dict) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in choices(row))
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Question:\n{row['question']}\n\nOptions:\n{opts}"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_independent(raw: str) -> str | None:
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    prefixes = ("the correct answer is", "the answer is:", "answer:")
    lower = text.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
            if text.startswith((":", "-")):
                text = text[1:].strip()
            break
    found = re.fullmatch(r"\(?([A-Da-d])\)?[.)]?", text.strip())
    return found.group(1).upper() if found else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rvl-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    args = parser.parse_args()
    run_dir, rvl_path, model_path = args.run_dir.resolve(), args.rvl_source.resolve(), args.model_path.resolve()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    import pandas as pd
    from transformers import AutoTokenizer

    checks: dict[str, bool] = {}
    def check(name: str, predicate: bool) -> None:
        checks[name] = bool(predicate)
        if not predicate:
            raise AssertionError(name)

    check("canonical_protocol_digest", canonical_sha(lock) == lock.get("sha256"))
    check("runner_lock", sha_file(RUNNER) == lock["source_hashes"]["runner"] == summary["runner_sha256"])
    check("auditor_lock", sha_file(Path(__file__).resolve()) == lock["source_hashes"]["auditor"])
    check("adapter_lock", sha_file(ADAPTER) == lock["source_hashes"]["vare_adapter"])
    rvl_hashes = {p.relative_to(rvl_path).as_posix(): sha_file(p) for p in sorted((rvl_path / "src/rvl_systems").rglob("*.py"))}
    check("pinned_rvl_sources", rvl_hashes == lock["source_hashes"]["rvl_files"])
    check("model_files", all(sha_file(model_path / n) == h for n, h in lock["model"]["files_sha256"].items()))
    check("data_hashes", sha_file(args.train_parquet) == lock["dataset"]["train_sha256"] and sha_file(args.validation_parquet) == lock["dataset"]["validation_sha256"])
    check("recorded_protocol", summary["protocol_id"] == lock["protocol_id"] and summary["protocol_sha256"] == sha_file(LOCK))

    train_rows = pd.read_parquet(args.train_parquet).to_dict(orient="records")
    val_rows = pd.read_parquet(args.validation_parquet).to_dict(orient="records")
    excluded_train = set(lock["selection"]["excluded_train_ids"])
    train = select([r for r in train_rows if str(r["id"]) not in excluded_train], lock["selection"]["train_salt"], lock["selection"]["train_n"])
    excluded = set(lock["selection"]["excluded_validation_ids"])
    val_pool = [r for r in val_rows if str(r["id"]) not in excluded]
    val = select(val_pool, lock["selection"]["validation_salt"], lock["selection"]["validation_n"])
    check("hash_ranked_row_selection", [str(r["id"]) for r in train] == lock["selection"]["train_ids"] and [str(r["id"]) for r in val] == lock["selection"]["validation_ids"])
    used_ids = {str(r["id"]) for r in val}
    for version in (1, 2):
        path = ROOT / f"results/qwen-arc-challenge-base-gate-v{version}/run-1/responses.jsonl"
        old_ids = {json.loads(line)["task_id"] for line in path.read_text().splitlines()}
        check(f"disjoint_from_base_gate_v{version}", used_ids.isdisjoint(old_ids))
    failed_smoke = json.loads((ROOT / "results/qwen-arc-grpo-update-smoke-v1/run-1/summary.json").read_text(encoding="utf-8"))
    failed_smoke_ids = {x["task_id"] for x in failed_smoke["base_eval"]}
    check("disjoint_from_failed_smoke", used_ids.isdisjoint(failed_smoke_ids))
    prior_train_ids = set(json.loads((ROOT / "protocols/qwen_arc_grpo_update_smoke_v1.lock.json").read_text(encoding="utf-8"))["selection"]["train_ids"])
    check("train_sample_disjoint_from_v1_candidates", set(lock["selection"]["train_ids"]).isdisjoint(prior_train_ids))

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    by_train = {str(r["id"]): r for r in train}
    by_val = {str(r["id"]): r for r in val}
    attempts = summary["train_group_attempts"]
    check("candidate_group_attempt_prefix", [x["task_id"] for x in attempts] == [str(x["id"]) for x in train[:len(attempts)]])
    first_mixed = None
    for index, (group, row) in enumerate(zip(attempts, train, strict=True)):
        qid = str(row["id"])
        prompt = render(tokenizer, row)
        decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in group["generated_token_ids"]]
        rewards = [float(parse_independent(x) == str(row["answerKey"]).upper()) for x in decoded]
        check(f"train_group_{index}_replay", group["task_id"] == qid and group["answer_key"] == str(row["answerKey"]).upper() and group["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest() and decoded == group["raw_generations"] and rewards == group["rewards"])
        if len(set(rewards)) > 1:
            first_mixed = index
            break
    check("selected_first_mixed_train_group", first_mixed is not None and summary.get("selected_train_index") == first_mixed and summary.get("selected_train_task_id") == attempts[first_mixed]["task_id"])

    chosen_rewards = attempts[first_mixed]["rewards"]
    mean = sum(chosen_rewards) / len(chosen_rewards)
    variance = sum((x - mean) ** 2 for x in chosen_rewards) / len(chosen_rewards)
    eps, clip = lock["update"]["advantage_epsilon"], lock["update"]["advantage_clip"]
    expected_adv = [max(-clip, min(clip, (x-mean)/math.sqrt(variance+eps))) for x in chosen_rewards]
    actual_adv = summary["group_advantages"]
    check("group_advantages", len(actual_adv) == len(expected_adv) and all(abs(a-b)<1e-10 for a,b in zip(actual_adv, expected_adv, strict=True)))

    def audit_eval(name: str, recorded: list[dict]) -> tuple[int, float]:
        check(f"{name}_evaluation_cardinality", len(recorded) == len(val))
        expected = []
        for row in val:
            qid = str(row["id"])
            prompt = render(tokenizer, row)
            matching = [x for x in recorded if x["task_id"] == qid]
            check(f"{name}_{qid}_present_once", len(matching) == 1)
            item = matching[0]
            decoded = tokenizer.decode(item["generated_token_ids"], skip_special_tokens=True)
            parsed = parse_independent(decoded)
            answer = str(row["answerKey"]).upper()
            check(f"{name}_{qid}_replay", item["answer_key"] == answer and item["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest() and item["raw_generation"] == decoded and item["parsed_label"] == parsed and item["exact"] == (parsed == answer))
            expected.append((parsed, answer))
        acc = sum(a == b for a,b in expected)
        parse_rate = sum(a is not None for a,_ in expected)/len(expected)
        return acc, parse_rate

    base_acc, base_parse = audit_eval("base", summary["base_eval"])
    post_acc, post_parse = audit_eval("post_reload", summary["post_reload_eval"])
    check("base_metrics", summary["base_exact_successes"] == base_acc and abs(summary["base_parse_rate"]-base_parse)<1e-12)
    check("post_metrics", summary["post_reload_exact_successes"] == post_acc and abs(summary["post_reload_parse_rate"]-post_parse)<1e-12)
    check("update_path", summary.get("optimizer_step_calls") == 1 and summary.get("gradient_tensor_count",0)>0 and summary.get("gradient_tensor_count")==summary.get("finite_gradient_tensor_count") and summary.get("nonzero_gradient_tensor_count",0)>0 and summary.get("changed_parameter_tensor_count",0)>0 and summary.get("max_abs_parameter_delta",0)>0)
    check("transactional_incumbent_restore", bool(summary.get("incumbent_restored_exact")))
    check("checkpoint_roundtrip", bool(summary.get("checkpoint_roundtrip_exact")) and bool(summary.get("tokenizer_roundtrip_exact")) and summary.get("candidate_parameter_fingerprint")==summary.get("reloaded_parameter_fingerprint"))
    check("checkpoint_files_hashed", bool(summary.get("checkpoint_file_sha256")) and all(len(h)==64 for h in summary["checkpoint_file_sha256"].values()))
    gates = summary["gate"]
    expected_gate = (len(summary["post_reload_eval"]) == lock["selection"]["validation_n"]
        and summary["finite_gradient_tensor_count"] == summary["gradient_tensor_count"]
        and summary["nonzero_gradient_tensor_count"] > 0 and summary["optimizer_step_calls"] == 1
        and summary["incumbent_restored_exact"] and summary["checkpoint_roundtrip_exact"] and summary["tokenizer_roundtrip_exact"]
        and summary["wall_seconds"] <= lock["resources"]["wall_seconds_cap"]
        and summary["peak_rss_mib"] <= lock["resources"]["peak_rss_mib_cap"])
    check("frozen_gate", bool(gates) == expected_gate and expected_gate and summary["status"] == "passed")
    result = {
        "protocol_id": lock["protocol_id"], "protocol_sha256": sha_file(LOCK),
        "auditor_sha256": sha_file(Path(__file__).resolve()), "checks": checks,
        "checks_passed": len(checks), "checks_total": len(checks), "audit_status": "pass",
        "run_gate": bool(expected_gate), "n_validation": len(val),
        "base_exact_successes": base_acc, "base_parse_rate": base_parse,
        "post_reload_exact_successes": post_acc, "post_reload_parse_rate": post_parse,
        "claim_limit": "Same-host retained-record audit. The 16-item before/after smoke is not a task-improvement estimate and no checkpoint is retained.",
    }
    (run_dir / "independent_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
