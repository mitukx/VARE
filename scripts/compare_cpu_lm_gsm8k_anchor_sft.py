#!/usr/bin/env python3
"""Compare paired, independently audited SFT and anchored-DPO development runs."""
from __future__ import annotations
import argparse
import copy
import json
import statistics
from pathlib import Path
from compare_cpu_lm_gsm8k_dpo_sft import bootstrap_mean_interval, checked_run, question_success, canonical


def compare(sft_dir: Path, anchor_dir: Path):
    checked = [checked_run(sft_dir, "sft"), checked_run(anchor_dir, "dpo_sft_anchor")]
    specs = [item[0] for item in checked]
    runs = [item[1] for item in checked]
    normalized = [copy.deepcopy(spec) for spec in specs]
    for spec in normalized:
        spec.pop("protocol_id", None)
        spec.pop("purpose", None)
        spec["learner"].pop("method", None)
        spec["learner"].pop("objective", None)
        spec["learner"].pop("sft_anchor_weight", None)
    if canonical(normalized[0]) != canonical(normalized[1]):
        raise ValueError("SFT and anchored-DPO locks differ beyond the declared objective fields")
    sft, anchor = runs
    for key in ("train_examples", "validation_examples"):
        if sft[key] != anchor[key]:
            raise ValueError(f"paired runs do not share identical {key}")
    for key in ("base_training_generation", "base_validation_generation"):
        if sft["summary"][key] != anchor["summary"][key]:
            raise ValueError(f"paired runs have different {key}")
    validation = sft["validation_examples"]
    base = sft["summary"]["base_validation_generation"]
    if len(base) != len(validation) or any(row["dataset_index"] != ex["dataset_index"] for row, ex in zip(base, validation)):
        raise ValueError("base generations are not aligned with validation questions")
    base_em = statistics.fmean(float(row["exact_match"]) for row in base)
    sft_q = question_success(sft)
    anchor_q = question_success(anchor)
    if len(sft_q) != len(anchor_q):
        raise ValueError("selected-generation validation lengths differ")
    diffs = [a-b for a,b in zip(anchor_q,sft_q)]

    def arm(data):
        summary=data["summary"]
        epoch=str(summary["selected_epochs"])
        ckpt=summary["checkpoint_summary"][epoch]
        return {"decision":summary["decision"],"selected_epoch":summary["selected_epochs"],
            "mean_exact_match":summary["selected_mean_exact_match_accuracy"],
            "validation_preference_nll":ckpt["mean_validation_dpo_preference_nll"],
            "preference_accuracy":ckpt["mean_validation_preference_accuracy"],
            "mean_token_kl":ckpt["mean_validation_token_kl_to_base"],
            "per_seed_exact_match":[statistics.fmean(float(x["exact_match"]) for x in y["generated_validation"])
                for y in summary["selected_adapters_and_generation"]]}
    return {"comparison":"matched_sft_vs_anchored_dpo_development", "claim_boundary":
        "Development-only; the same validation questions select checkpoints. Intervals are descriptive, not independent confirmation or capability evidence.",
        "base_exact_match":base_em,"validation_questions":len(validation),"seeds":specs[0]["learner"]["seeds"],
        "arms":{"sft":arm(sft),"dpo_sft_anchor":arm(anchor)},
        "anchor_minus_sft":{"mean_exact_match_difference":statistics.fmean(diffs),
            "question_bootstrap_95_percent_interval":bootstrap_mean_interval(diffs),
            "questions_better_tied_worse":{"better":sum(x>0 for x in diffs),"tied":sum(x==0 for x in diffs),"worse":sum(x<0 for x in diffs)}}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sft_run",type=Path)
    parser.add_argument("anchor_run",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=compare(args.sft_run.resolve(),args.anchor_run.resolve())
    encoded=json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded,encoding="utf-8")
    print(encoded,end="")

if __name__=="__main__":
    main()
