#!/usr/bin/env python3
"""Compare audited DPO, SFT and anchored-DPO BoolQ development runs."""
from __future__ import annotations
import argparse
import copy
import json
import statistics
from pathlib import Path
from compare_cpu_lm_gsm8k_dpo_sft import bootstrap_mean_interval, checked_run, canonical, question_success


def compare(dpo_dir: Path, sft_dir: Path, anchor_dir: Path):
    methods=["dpo","sft","dpo_sft_anchor"]
    dirs=[dpo_dir,sft_dir,anchor_dir]
    checked=[checked_run(path,method) for path,method in zip(dirs,methods)]
    specs=[x[0] for x in checked]; runs=[x[1] for x in checked]
    normalized=[copy.deepcopy(x) for x in specs]
    for spec in normalized:
        spec.pop("protocol_id",None); spec.pop("purpose",None)
        for key in ("method","objective","sft_anchor_weight"): spec["learner"].pop(key,None)
    if any(canonical(x)!=canonical(normalized[0]) for x in normalized[1:]):
        raise ValueError("BoolQ method locks differ beyond declared method/objective fields")
    ref=runs[0]
    for method,data in zip(methods[1:],runs[1:]):
        for key in ("train_examples","validation_examples"):
            if ref[key]!=data[key]: raise ValueError(f"paired runs differ in {key}: {method}")
        for key in ("base_training_generation","base_validation_generation"):
            if ref["summary"][key]!=data["summary"][key]: raise ValueError(f"paired runs differ in {key}: {method}")
        if ref["summary"]["base_validation_class_metrics"]!=data["summary"]["base_validation_class_metrics"]:
            raise ValueError(f"paired runs differ in base class metrics: {method}")
    base=ref["summary"]["base_validation_class_metrics"]
    validation_indices=[row["dataset_index"] for row in ref["validation_examples"]]
    for method,data in zip(methods,runs):
        summary=data["summary"]
        if [row["dataset_index"] for row in summary["base_validation_generation"]] != validation_indices:
            raise ValueError(f"base validation generations are misaligned: {method}")
        for item in summary["selected_adapters_and_generation"]:
            if [row["dataset_index"] for row in item["generated_validation"]] != validation_indices:
                raise ValueError(f"selected generations are misaligned: {method}/{item['seed']}")
    qmat={method:question_success(data) for method,data in zip(methods,runs)}
    if len({len(x) for x in qmat.values()})!=1: raise ValueError("validation generation lengths differ")
    def arm(data):
        s=data["summary"]; epoch=str(s["selected_epochs"]); ck=s["checkpoint_summary"][epoch]
        return {"decision":s["decision"],"selected_epoch":s["selected_epochs"],
            "mean_exact_match":s["selected_mean_exact_match_accuracy"],
            "mean_balanced_accuracy":s["selected_mean_balanced_accuracy"],
            "validation_preference_nll":ck["mean_validation_dpo_preference_nll"],
            "preference_accuracy":ck["mean_validation_preference_accuracy"],
            "mean_token_kl":ck["mean_validation_token_kl_to_base"],
            "per_seed_exact_match":[statistics.fmean(float(x["exact_match"]) for x in y["generated_validation"])
                for y in s["selected_adapters_and_generation"]],
            "selected_class_metrics_by_seed":s["selected_validation_class_metrics_by_seed"]}
    comparisons={}
    for name,method in (("dpo_minus_sft","dpo"),("anchor_minus_sft","dpo_sft_anchor"),("anchor_minus_dpo","dpo_sft_anchor")):
        control="dpo" if name=="anchor_minus_dpo" else "sft"
        diffs=[a-b for a,b in zip(qmat[method],qmat[control])]
        comparisons[name]={"mean_exact_match_difference":statistics.fmean(diffs),
            "question_bootstrap_95_percent_interval":bootstrap_mean_interval(diffs),
            "questions_better_tied_worse":{"better":sum(x>0 for x in diffs),"tied":sum(x==0 for x in diffs),"worse":sum(x<0 for x in diffs)}}
    return {"comparison":"matched_boolq_dpo_sft_anchor_development",
        "claim_boundary":"Development-only; checkpoints are selected on this validation cohort. Intervals are descriptive, not independent confirmation or capability evidence.",
        "base_class_metrics":base,"validation_questions":len(ref["validation_examples"]),
        "seeds":specs[0]["learner"]["seeds"],"arms":{m:arm(d) for m,d in zip(methods,runs)},
        "paired_question_comparisons":comparisons}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dpo_run",type=Path); parser.add_argument("sft_run",type=Path)
    parser.add_argument("anchor_run",type=Path); parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=compare(args.dpo_run.resolve(),args.sft_run.resolve(),args.anchor_run.resolve())
    encoded=json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(encoded,encoding="utf-8")
    print(encoded,end="")
if __name__=="__main__": main()
