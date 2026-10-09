#!/usr/bin/env python3
"""Independent, data-backed audit for the frozen OpenBookQA study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/"protocols/openbookqa_qwen_grpo_sft_v5.lock.json"


def sha_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def canonical_lock_sha(lock:dict)->str:
    body={k:v for k,v in lock.items() if k!="lock_sha256"}
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def read_lock()->dict:
    lock=json.loads(LOCK.read_text())
    if canonical_lock_sha(lock)!=lock.get("lock_sha256"):raise AssertionError("lock digest mismatch")
    for key,path in (("runner",ROOT/"scripts/run_openbookqa_qwen_grpo_sft_v5.py"),("auditor",Path(__file__))):
        if sha_file(path)!=lock["source_hashes"][key]:raise AssertionError(f"{key} source hash mismatch")
    if sha_file(ROOT/"src/vare/integrations/rvl_grpo.py")!=lock["source_hashes"]["vare_adapter"]:
        raise AssertionError("VARE adapter hash mismatch")
    return lock


def check_pinned_sources(lock:dict,model_path:Path,rvl_path:Path)->None:
    files={p.relative_to(rvl_path).as_posix():sha_file(p) for p in sorted((rvl_path/"src/rvl_systems").rglob("*.py"))}
    if files!=lock["source_hashes"]["rvl_python_files"]:raise AssertionError("RVL source revision/hash mismatch")
    model_hashes={name:sha_file(model_path/name) for name in lock["model"]["files_sha256"]}
    if model_hashes!=lock["model"]["files_sha256"]:raise AssertionError("base model hash mismatch")


def select(rows:list[dict],split:str,n:int)->list[dict]:
    salt=f"vare-openbookqa-qwen-v1-{split}-20261010"
    return sorted(rows,key=lambda r:hashlib.sha256(f"{salt}|{r['id']}".encode()).hexdigest())[:n]


def parse_independently(text:str)->str|None:
    # Accept only a choice token at the beginning; trailing rationale is ignored.
    value=str(text).strip()
    match=re.match(r"^([A-D])(?=$|[.)\s])",value)
    return None if match is None else match.group(1)


def rows_from_parquet(path:Path,split:str,lock:dict)->list[dict]:
    import pandas as pd
    if sha_file(path)!=lock["dataset"]["parquet_sha256"][split]:raise AssertionError(f"{split} data hash mismatch")
    rows=pd.read_parquet(path).to_dict(orient="records")
    if len({str(r["id"]) for r in rows})!=len(rows):raise AssertionError(f"duplicate {split} IDs")
    return rows


def check_eval_block(records:list[dict],expected_rows:list[dict],split:str)->dict:
    truths={str(x["id"]):str(x["answerKey"]) for x in expected_rows}
    expected_ids=set(truths)
    got=[str(x["task_id"]) for x in records]
    if len(got)!=len(set(got)) or set(got)!=expected_ids:raise AssertionError(f"{split} task-ID coverage mismatch")
    parse_errors=false_accepts=0;correct=0
    for item in records:
        qid=str(item["task_id"])
        parsed=parse_independently(item["raw_generation"])
        if parsed!=item.get("parsed_label"):raise AssertionError(f"parser mismatch for {qid}")
        if item.get("correct_label")!=truths[qid]:raise AssertionError(f"gold-label mismatch for {qid}")
        exact=parsed==truths[qid]
        if bool(item.get("exact"))!=exact:raise AssertionError(f"scoring mismatch for {qid}")
        parse_errors+=int(parsed is None);false_accepts+=int(exact and parsed!=truths[qid]);correct+=int(exact)
    return {"n":len(records),"correct":correct,"accuracy":correct/len(records),"parse_failures":parse_errors,
        "parse_rate":1-parse_errors/len(records),"false_accepts":false_accepts}


def audit_protocol(args,lock):
    train=rows_from_parquet(args.train_parquet,"train",lock)
    dev=rows_from_parquet(args.dev_parquet,"validation",lock)
    st=select(train,"train",lock["selection"]["train_n"])
    sd=select(dev,"validation",lock["selection"]["dev_n"])
    if [str(x["id"]) for x in st]!=lock["selection"]["train_ids"]:raise AssertionError("train selection mismatch")
    if [str(x["id"]) for x in sd]!=lock["selection"]["dev_ids"]:raise AssertionError("development selection mismatch")
    if {x["id"] for x in st}&{x["id"] for x in sd}:raise AssertionError("train/dev overlap")
    print(json.dumps({"status":"pass","protocol_id":lock["protocol_id"],"train_n":len(st),"dev_n":len(sd),"confirmation_read":False,
        "dataset_revision":lock["dataset"]["revision"],"model_revision":lock["model"]["revision"],"rvl_revision":lock["sources"]["rvl_revision"]},indent=2))
    return 0


def audit_base(args,lock):
    train=select(rows_from_parquet(args.train_parquet,"train",lock),"train",lock["selection"]["train_n"])
    dev=select(rows_from_parquet(args.dev_parquet,"validation",lock),"validation",lock["selection"]["dev_n"])
    obj=json.loads(args.base_gate.read_text())
    dev_check=check_eval_block(obj["dev_results"],dev,"validation")
    if dev_check["accuracy"]!=obj["dev_accuracy"] or dev_check["parse_rate"]!=obj["dev_parse_rate"]:raise AssertionError("base feasibility metric mismatch")
    rollouts=json.loads((args.base_gate.parent/"training-rollouts.json").read_text())
    if len(rollouts["seed_groups"])!=len(lock["training"]["seeds"]):raise AssertionError("seed count mismatch")
    readout={}
    by_id={str(x["id"]):x for x in train}
    for seed,groups in rollouts["seed_groups"].items():
        if len(groups)!=len(train):raise AssertionError(f"{seed}: group count mismatch")
        positive=mixed=parsefail=0;seen=set()
        for group,row in zip(groups,train):
            if len(group)!=4:raise AssertionError(f"{seed}: incomplete reward group")
            qid=str(row["id"])
            if any(str(x["task_id"])!=qid or str(x["answer"])!=str(row["answerKey"]) for x in group):raise AssertionError("rollout task/gold binding mismatch")
            rewards=[]
            for item in group:
                parsed=parse_independently(item["raw_generation"])
                reward=int(parsed==str(row["answerKey"]))
                if item.get("parsed_label")!=parsed or float(item["reward"])!=float(reward):raise AssertionError("independent reward recomputation mismatch")
                parsefail+=int(parsed is None);positive+=reward;rewards.append(reward)
            mixed+=int(len(set(rewards))>1)
        readout[seed]={"positive_traces":positive,"mixed_groups":mixed,"mixed_group_rate":mixed/len(groups),"parse_failures":parsefail}
        if readout[seed]!=obj["rollout_readiness"][seed]:raise AssertionError(f"{seed}: readiness record mismatch")
    expected=(dev_check["accuracy"]>=lock["base_gate"]["minimum_dev_accuracy"] and dev_check["parse_rate"]>=lock["base_gate"]["minimum_dev_parse_rate"]
        and all(x["positive_traces"]>=lock["base_gate"]["minimum_positive_traces_per_seed"] and x["mixed_group_rate"]>=lock["base_gate"]["minimum_mixed_group_rate"] for x in readout.values()))
    if expected!=bool(obj["base_gate_passed"]):raise AssertionError("base gate decision mismatch")
    print(json.dumps({"status":"pass","dev":dev_check,"rollout_readiness":readout,"gate":expected,"confirmation_read":False},indent=2))
    return 0


def bootstrap_summary(lock,evaluations):
    seeds=[str(s) for s in lock["training"]["seeds"]]
    base=evaluations["base"]["results"]
    base_map={str(x["task_id"]):int(x["exact"]) for x in base};ids=list(base_map)
    g={s:{str(x["task_id"]):int(x["exact"]) for x in evaluations[f"grpo-{s}"]["results"]} for s in seeds}
    sft={s:{str(x["task_id"]):int(x["exact"]) for x in evaluations[f"sft-{s}"]["results"]} for s in seeds}
    if any(set(x)!=set(ids) for x in [*g.values(),*sft.values()]):raise AssertionError("arms use different confirmation IDs")
    differences_s=[];differences_b=[];rng=random.Random(lock["analysis"]["bootstrap_seed"])
    for _ in range(lock["analysis"]["bootstrap_replicates"]):
        picked=[rng.choice(ids) for _ in ids]
        differences_s.append(sum(sum(g[seed][qid]-sft[seed][qid] for qid in picked)/len(picked) for seed in seeds)/len(seeds))
        differences_b.append(sum(sum(g[seed][qid]-base_map[qid] for qid in picked)/len(picked) for seed in seeds)/len(seeds))
    def interval(values):
        values.sort();n=len(values);return [values[math.floor(.025*(n-1))],values[math.floor(.975*(n-1))]]
    per_seed={seed:{"grpo_minus_sft":sum(g[seed][q]-sft[seed][q] for q in ids)/len(ids),"grpo_minus_base":sum(g[seed][q]-base_map[q] for q in ids)/len(ids)} for seed in seeds}
    d_s=sum(sum(g[x].values())/len(ids)-sum(sft[x].values())/len(ids) for x in seeds)/len(seeds)
    d_b=sum(sum(g[x].values())/len(ids)-sum(base_map.values())/len(ids) for x in seeds)/len(seeds)
    ci_s=interval(differences_s);ci_b=interval(differences_b);rules=lock["analysis"]["success_criteria"]
    gate=(d_s>=rules["minimum_grpo_minus_sft"] and ci_s[0]>0 and sum(v["grpo_minus_sft"]>0 for v in per_seed.values())>=rules["minimum_positive_seeds"]
        and d_b>=rules["minimum_grpo_minus_base"] and ci_b[0]>0 and sum(v["grpo_minus_base"]>0 for v in per_seed.values())>=rules["minimum_positive_seeds"])
    return {"n_confirmation":len(ids),"grpo_minus_sft":d_s,"grpo_minus_sft_bootstrap_95":ci_s,
        "grpo_minus_base":d_b,"grpo_minus_base_bootstrap_95":ci_b,"per_seed":per_seed,"success_gate":bool(gate)}


def audit_final(args,lock):
    summary=json.loads(args.summary.read_text())
    test=select(rows_from_parquet(args.test_parquet,"test",lock),"test",lock["selection"]["test_n"])
    if not summary.get("confirmation_opened") or summary.get("confirmation_n")!=len(test):raise AssertionError("confirmation stage was not recorded correctly")
    for name,ev in summary["evaluation"].items():
        check=check_eval_block(ev["results"],test,"test")
        for key in ("accuracy","parse_rate"):
            if not math.isclose(check[key],float(ev[key]),rel_tol=0,abs_tol=1e-12):raise AssertionError(f"{name}: {key} mismatch")
        if check["false_accepts"]!=0:raise AssertionError(f"{name}: verifier false acceptance")
    computed=bootstrap_summary(lock,summary["evaluation"])
    for key in ("grpo_minus_sft","grpo_minus_sft_bootstrap_95","grpo_minus_base","grpo_minus_base_bootstrap_95","success_gate"):
        if computed[key]!=summary["evaluation_summary"][key]:raise AssertionError(f"summary metric {key} mismatch")
    expected_continue=computed["success_gate"] and bool(summary.get("resource_gate"))
    if expected_continue!= (summary["decision"]=="CONTINUE"):raise AssertionError("final decision does not match frozen gates")
    print(json.dumps({"status":"pass","summary":computed,"all_arms_independently_rescored":True,"false_accepts":0},indent=2))
    return 0


def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["protocol","base-gate","final"],required=True)
    p.add_argument("--rvl-source",type=Path,required=True);p.add_argument("--model-path",type=Path,required=True)
    p.add_argument("--train-parquet",type=Path);p.add_argument("--dev-parquet",type=Path);p.add_argument("--test-parquet",type=Path)
    p.add_argument("--base-gate",type=Path);p.add_argument("--summary",type=Path)
    args=p.parse_args();lock=read_lock();check_pinned_sources(lock,args.model_path,args.rvl_source)
    if args.mode=="protocol":return audit_protocol(args,lock)
    if args.mode=="base-gate":return audit_base(args,lock)
    return audit_final(args,lock)


if __name__=="__main__":raise SystemExit(main())
