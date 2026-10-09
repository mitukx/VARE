#!/usr/bin/env python3
"""Independently verify ASDiv labels, sample selection, and paired-margin results."""
from __future__ import annotations

import ast
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import unicodedata
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/asdiv_dpo_semantic_transfer_v1.json"
LOCK = ROOT / "protocols/asdiv_dpo_semantic_transfer_v1.lock.json"
DATA = ROOT.parent / "work/private/asdiv-dpo-transfer-v1/asdiv.parquet"
GSM8K_DIR = Path.home() / ".cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower()))


def evaluate(node: ast.AST) -> Decimal:
    """Independent Decimal oracle; accepts only numeric literals and basic arithmetic."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = evaluate(node.left), evaluate(node.right)
        operations = {ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
                      ast.Mult: lambda: left * right, ast.Div: lambda: left / right}
        for operator, apply in operations.items():
            if isinstance(node.op, operator):
                return apply()
    raise ValueError(f"unsupported AST: {type(node).__name__}")


def source_answer(row: dict[str, str]) -> Decimal:
    equation = row["formula"].replace("−", "-").replace("×", "*").replace("÷", "/")
    equation = re.sub(r"(?<=\d)\s*\([A-Za-z][A-Za-z /-]*\)", "", equation)
    equation = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", equation)
    if equation.count("=") != 1:
        raise ValueError("formula is not a single equality")
    lhs, rhs = equation.split("=")
    rhs_match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:[A-Za-z][A-Za-z /-]*)?\s*", rhs)
    answer_match = re.match(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", row["answer"])
    if not rhs_match or not answer_match:
        raise ValueError("formula/result is not a supported scalar")
    computed = evaluate(ast.parse(lhs, mode="eval").body)
    stated = Decimal(rhs_match.group(1))
    answer = Decimal(answer_match.group(1))
    if computed != stated or stated != answer:
        raise ValueError("formula, equation result, and answer field disagree")
    return answer


def sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def nll(margin: float) -> float:
    return max(-margin, 0.0) + math.log1p(math.exp(-abs(margin)))


def paired_nll(semantic: float, position: float) -> float:
    return (nll(semantic + position) + nll(semantic - position)) / 2


def paired_kl(base: float, updated: float) -> float:
    p, q = sigmoid(base), sigmoid(updated)
    return q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    lock_hash = lock.pop("sha256", None)
    protocol_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if lock_hash != protocol_hash or canonical(lock) != canonical(spec):
        raise ValueError("protocol lock mismatch")
    if digest(DATA) != spec["source"]["sha256"]:
        raise ValueError("ASDiv source file hash mismatch")
    result_manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if result_manifest.get("algorithm") != "sha256":
        raise ValueError("result bundle has no SHA-256 manifest")
    for name, expected_hash in result_manifest.get("files", {}).items():
        if digest(bundle / name) != expected_hash:
            raise ValueError(f"pre-audit bundle hash mismatch: {name}")
    run_summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    if run_summary["protocol_sha256"] != protocol_hash or run_summary["runner_sha256"] != spec["implementation"]["runner_sha256"]:
        raise ValueError("run summary does not match locked protocol/runner")
    if run_summary["device"] != "cpu" or run_summary["paid_compute"] or not run_summary["network_disabled"]:
        raise ValueError("run violated frozen compute constraints")
    if run_summary["n_items"] != spec["selection"]["count"] or run_summary["n_prompts"] != 2 * spec["selection"]["count"]:
        raise ValueError("run prompt/item count differs from frozen protocol")
    if digest(GSM8K_DIR / "gsm8k-train.arrow") != "6146bca074f4c6c5e07de7e96cc968adab3833dbdafeda21830f27558eea3d9b":
        raise ValueError("pinned GSM8K train source is not available")
    if digest(GSM8K_DIR / "gsm8k-test.arrow") != "45965b000311d1550e5619b60b5bf31cf76edebfd8b8eddc62a876fbf8c9be95":
        raise ValueError("pinned GSM8K test source is not available")

    rows = pq.read_table(DATA).to_pylist()
    from datasets import load_dataset
    import os
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    gsm = load_dataset("openai/gsm8k", "main")
    gsm_texts = {normalized(row["question"]) for split in ("train", "test") for row in gsm[split]}
    eligible = []
    overlap_count = 0
    formula_rejections = 0
    for index, row in enumerate(rows):
        combined = (row["body"].strip() + " " + row["question"].strip()).strip()
        key = normalized(combined)
        if normalized(row["question"]) in gsm_texts or key in gsm_texts:
            overlap_count += 1
            continue
        try:
            source_answer(row)
        except (ValueError, ArithmeticError, SyntaxError, TypeError):
            formula_rejections += 1
            continue
        eligible.append((index, key))
    # Duplicate combined prompts are excluded before ranking.
    unique = {}
    for item in eligible:
        unique.setdefault(item[1], item)
    ranked = sorted(unique.values(), key=lambda item: hashlib.sha256(("vare-asdiv-transfer-v1\0" + item[1]).encode()).hexdigest())
    expected = [(index, hashlib.sha256(key.encode()).hexdigest(), hashlib.sha256(("vare-asdiv-transfer-v1\0" + key).encode()).hexdigest())
                for index, key in ranked[:spec["selection"]["count"]]]
    locked = [(item["index"], item["problem_sha256"], item["rank_sha256"]) for item in spec["selection"]["rows"]]
    if expected != locked:
        raise ValueError("selected cohort does not reconstruct from pinned source and selection rule")
    if overlap_count != spec["overlap_audit"]["exact_overlaps_found"] or len(eligible) != spec["eligibility"]["count_before_unique"] or len(unique) != spec["eligibility"]["unique_combined_texts"]:
        raise ValueError("source overlap or eligibility counts differ from frozen protocol")

    score_rows = json.loads((bundle / "scores.json").read_text(encoding="utf-8"))
    if len(score_rows) != len(locked):
        raise ValueError("score count differs from frozen cohort")
    seeds = ["401", "503", "607"]
    # Independently check selected raw margins through the model's full logits path,
    # rather than the runner's hidden-state-plus-output-row implementation.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_dir = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    for name, expected_hash in spec["models"]["model_file_sha256"].items():
        if digest(model_dir / name) != expected_hash:
            raise ValueError(f"model checkpoint hash mismatch: {name}")
    torch.set_num_threads(spec["compute"]["threads"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    token_ids = spec["models"]["response_token_ids"]
    oracle_model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, torch_dtype=torch.float32)
    oracle_model.to("cpu").eval()
    adapter_records = json.loads((ROOT / spec["models"]["adapter_bundle"]).read_text(encoding="utf-8"))
    oracle_seeds = {seed: (torch.tensor(adapter_records[seed]["adapter"]["A_final"], dtype=torch.float32),
                           torch.tensor(adapter_records[seed]["adapter"]["B_final"], dtype=torch.float32)) for seed in seeds}
    oracle_indices = sorted(range(len(score_rows)), key=lambda i: hashlib.sha256(("independent-full-logit-oracle-v1\0" + score_rows[i]["problem_sha256"]).encode()).hexdigest())[:8]
    oracle_prompts = []
    for qi in oracle_indices:
        source_row = rows[locked[qi][0]]
        text = (source_row["body"].strip() + " " + source_row["question"].strip()).strip()
        answer = Decimal(score_rows[qi]["answer"])
        wrong = Decimal(score_rows[qi]["wrong"])
        for correct_is_a in (True, False):
            a, b = (answer, wrong) if correct_is_a else (wrong, answer)
            def fmt(value: Decimal) -> str:
                value = value.normalize()
                return format(value.quantize(Decimal(1)), "f") if value == value.to_integral_value() else format(value, "f")
            oracle_prompts.append("Solve the following word problem. Choose the correct final numeric answer.\n\n"
                                  f"Problem: {text}\n\nA) {fmt(a)}\nB) {fmt(b)}\n\nAnswer with A or B:")
    encoded = tokenizer(oracle_prompts, return_tensors="pt", padding=True, add_special_tokens=True, truncation=False)
    lengths = encoded["attention_mask"].sum(dim=1) - 1
    with torch.inference_mode():
        output = oracle_model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], use_cache=False)
        hidden_all = oracle_model.model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], use_cache=False).last_hidden_state
        hidden = hidden_all[torch.arange(len(lengths)), lengths]
        full = output.logits[torch.arange(len(lengths)), lengths][:, [token_ids["A"][0], token_ids["B"][0]]]
        head = torch.nn.functional.linear(hidden, oracle_model.lm_head.weight[[token_ids["A"][0], token_ids["B"][0]]])
        full_margins = (full[:, 0] - full[:, 1]).tolist()
        head_margins = (head[:, 0] - head[:, 1]).tolist()
        oracle_errors = [abs(float(x - y)) for x, y in zip(full_margins, head_margins)]
        adapter_errors = {seed: [] for seed in seeds}
        for seed in seeds:
            a, b = oracle_seeds[seed]
            delta = (hidden @ a) @ b
            adapted = full + delta
            adapted_margins = (adapted[:, 0] - adapted[:, 1]).tolist()
            for item_slot, qi in enumerate(oracle_indices):
                for side in range(2):
                    slot = item_slot * 2 + side
                    key = "correct_a" if side == 0 else "correct_b"
                    expected = float(score_rows[qi]["adapter_margins"][seed][key])
                    adapter_errors[seed].append(abs(float(adapted_margins[slot]) - expected))
    max_base_error = max(oracle_errors)
    max_adapter_error = max(max(values) for values in adapter_errors.values())
    if max_base_error > 5e-5 or max_adapter_error > 5e-5:
        raise ValueError(f"independent full-logit recomputation differs: base={max_base_error}, adapter={max_adapter_error}, per_seed={ {seed:max(values) for seed,values in adapter_errors.items()} }")

    seed_delta: dict[str, list[float]] = {seed: [] for seed in seeds}
    seed_correct_margins: dict[str, list[float]] = {seed: [] for seed in seeds}
    seed_nll_delta: dict[str, list[float]] = {seed: [] for seed in seeds}
    seed_content_nll: dict[str, list[float]] = {seed: [] for seed in seeds}
    seed_position_nll: dict[str, list[float]] = {seed: [] for seed in seeds}
    seed_position_delta: dict[str, list[float]] = {seed: [] for seed in seeds}
    seed_kl: dict[str, list[float]] = {seed: [] for seed in seeds}
    base_semantic, base_position = [], []
    base_correct_margins = []
    for index, record in enumerate(score_rows):
        row = rows[locked[index][0]]
        if record["index"] != locked[index][0] or record["problem_sha256"] != locked[index][1]:
            raise ValueError("score row provenance mismatch")
        answer = source_answer(row)
        if Decimal(record["answer"]) != answer:
            raise ValueError("scored answer differs from independently evaluated source formula")
        if abs(Decimal(record["wrong"]) - answer) != Decimal(1):
            raise ValueError("distractor is not exactly one unit from verified answer")
        m_a, m_b = float(record["base_margin_correct_a"]), float(record["base_margin_correct_b"])
        base_semantic.append((m_a - m_b) / 2)
        base_position.append((m_a + m_b) / 2)
        base_correct_margins.extend((m_a, -m_b))
        for seed in seeds:
            updated_a = float(record["adapter_margins"][seed]["correct_a"])
            updated_b = float(record["adapter_margins"][seed]["correct_b"])
            delta_semantic = ((updated_a - updated_b) - (m_a - m_b)) / 2
            delta_position = ((updated_a + updated_b) - (m_a + m_b)) / 2
            seed_delta[seed].append(delta_semantic)
            seed_position_delta[seed].append(delta_position)
            seed_correct_margins[seed].extend((updated_a, -updated_b))
            semantic_base, position_base = (m_a - m_b) / 2, (m_a + m_b) / 2
            semantic_updated, position_updated = (updated_a - updated_b) / 2, (updated_a + updated_b) / 2
            l00 = paired_nll(semantic_base, position_base)
            l10 = paired_nll(semantic_updated, position_base)
            l01 = paired_nll(semantic_base, position_updated)
            l11 = paired_nll(semantic_updated, position_updated)
            content_component = ((l10 - l00) + (l11 - l01)) / 2
            position_component = ((l01 - l00) + (l11 - l10)) / 2
            if not math.isclose(l11 - l00, content_component + position_component, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("NLL attribution components do not reconstruct paired loss change")
            seed_nll_delta[seed].append(l11 - l00)
            seed_content_nll[seed].append(content_component)
            seed_position_nll[seed].append(position_component)
            seed_kl[seed].append((paired_kl(m_a, updated_a) + paired_kl(m_b, updated_b)) / 2)

    per_seed = {}
    for seed in seeds:
        per_seed[seed] = {
            "mean_semantic_margin_change": statistics.fmean(seed_delta[seed]),
            "mean_position_bias_change": statistics.fmean(seed_position_delta[seed]),
            "base_correct_choice_accuracy": statistics.fmean(1.0 if x > 0 else .5 if x == 0 else 0.0 for x in base_correct_margins),
            "correct_choice_accuracy": statistics.fmean(1.0 if x > 0 else .5 if x == 0 else 0.0 for x in seed_correct_margins[seed]),
            "correct_choice_nll": statistics.fmean(nll(x) for x in seed_correct_margins[seed]),
            "correct_choice_nll_change": statistics.fmean(seed_nll_delta[seed]),
            "nll_change_content_component": statistics.fmean(seed_content_nll[seed]),
            "nll_change_position_component": statistics.fmean(seed_position_nll[seed]),
            "mean_bernoulli_kl_updated_to_base": statistics.fmean(seed_kl[seed]),
        }
    overall = statistics.fmean(value for seed in seeds for value in seed_delta[seed])
    rng = random.Random(20261010)
    bootstrap = []
    for _ in range(10000):
        means = [statistics.fmean(rng.choices(values, k=len(values))) for values in seed_delta.values()]
        bootstrap.append(statistics.fmean(means))
    bootstrap.sort()
    ci = [bootstrap[math.floor(.025 * (len(bootstrap) - 1))], bootstrap[math.floor(.975 * (len(bootstrap) - 1))]]
    success = overall > 0 and ci[0] > 0 and all(per_seed[s]["mean_semantic_margin_change"] > 0 for s in seeds)
    audit = {
        "protocol_id": spec["protocol_id"], "protocol_sha256": protocol_hash,
        "auditor_sha256": digest(Path(__file__)),
        "source_rows": len(rows), "exact_gsm8k_overlap_rows": overlap_count,
        "formula_rejections": formula_rejections, "eligible_rows": len(eligible), "unique_eligible_rows": len(unique),
        "selected_rows_verified": len(locked), "independent_oracle": "safe Decimal AST arithmetic with explicit basic-operator whitelist",
        "mean_base_semantic_margin": statistics.fmean(base_semantic),
        "mean_base_position_bias": statistics.fmean(base_position),
        "base_correct_choice_accuracy": statistics.fmean(1.0 if x > 0 else .5 if x == 0 else 0.0 for x in base_correct_margins),
        "base_correct_choice_nll": statistics.fmean(nll(x) for x in base_correct_margins),
        "mean_semantic_margin_change": overall,
        "mean_correct_choice_nll_change": statistics.fmean(value for seed in seeds for value in seed_nll_delta[seed]),
        "mean_nll_change_content_component": statistics.fmean(value for seed in seeds for value in seed_content_nll[seed]),
        "mean_nll_change_position_component": statistics.fmean(value for seed in seeds for value in seed_position_nll[seed]),
        "paired_seed_stratified_bootstrap_95pct": ci,
        "per_seed": per_seed,
        "independent_full_logit_oracle": {"items": len(oracle_indices), "prompts": len(oracle_prompts),
                                           "max_base_margin_abs_error": max_base_error,
                                           "max_adapter_margin_abs_error": max_adapter_error,
                                           "tolerance": 5e-5},
        "decision": "positive_cross_dataset_preference_transfer" if success else "non_pass_or_null",
        "claim_limit": spec["claim_boundary"],
        "bundle_summary": run_summary,
    }
    (bundle / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = {path.name: digest(path) for path in bundle.iterdir() if path.is_file() and path.name != "manifest.json"}
    manifest_path.write_text(json.dumps({"algorithm": "sha256", "files": manifest}, indent=2, sort_keys=True) + "\n")
    # Re-read every retained raw artifact after writing the auditor output.
    for name, expected_hash in manifest.items():
        if digest(bundle / name) != expected_hash:
            raise ValueError(f"bundle hash mismatch: {name}")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
