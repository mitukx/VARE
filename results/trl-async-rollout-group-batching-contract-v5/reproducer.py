"""Source-level contract audit for complete-or-drop AsyncGRPO rollout batching."""
import ast
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class IterableDataset:
    pass


class FiniteStream:
    def __init__(self, rows): self.rows = rows
    def __iter__(self): return iter(self.rows)


def load_batchers():
    source = Path.cwd() / "trl/experimental/async_grpo/async_grpo_trainer.py"
    tree = ast.parse(source.read_text())
    wanted = {"_balance_by_squared_length", "FixedCountBatcher", "TokenBudgetBatcher"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in wanted]
    if {getattr(node, "name", None) for node in nodes} != wanted:
        raise RuntimeError("required production definitions missing")
    namespace = {"Any": Any, "torch": SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(IterableDataset=IterableDataset))),
                 "logger": logging.getLogger("async-rollout-contract-audit")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["FixedCountBatcher"], namespace["TokenBudgetBatcher"]


def row(row_id, rollout_id, rollout_size, tokens):
    return {"row_id": row_id, "rollout_id": rollout_id, "rollout_size": rollout_size,
            "input_ids": list(range(tokens))}


def groups(rows):
    out = defaultdict(list)
    for sample in rows: out[sample["rollout_id"]].append(sample["row_id"])
    return dict(out)


def inspect(rows, batches, metrics, processes, fixed_capacity=None, token_budget=None):
    source_groups = groups(rows)
    emitted = defaultdict(list)
    row_batch_ids = defaultdict(set)
    row_count_by_id = Counter()
    capacity_ok = True
    nonempty = True
    for batch_index, rank_rows in enumerate(batches):
        if len(rank_rows) != processes or any(not rank for rank in rank_rows):
            nonempty = False
            continue
        flat = [sample for rank in rank_rows for sample in rank]
        if fixed_capacity is not None and len(flat) != fixed_capacity:
            capacity_ok = False
        if token_budget is not None and any(sum(len(s["input_ids"]) for s in rank) > token_budget for rank in rank_rows):
            capacity_ok = False
        for sample in flat:
            emitted[sample["rollout_id"]].append(sample["row_id"])
            row_batch_ids[sample["rollout_id"]].add(batch_index)
            row_count_by_id[sample["row_id"]] += 1
    drop_rows = sum(metrics.get("batch/dropped_whole_rollout_samples", []))
    groups_whole = True
    no_group_split = True
    for rollout_id, input_ids in source_groups.items():
        outputs = emitted.get(rollout_id, [])
        if len(outputs) not in (0, len(input_ids)):
            groups_whole = False
        if outputs and len(row_batch_ids[rollout_id]) != 1:
            no_group_split = False
    emitted_ids = [row_id for ids in emitted.values() for row_id in ids]
    missing = len(rows) - len(emitted_ids)
    unique_once = all(row_count_by_id[row_id] == 1 for row_id in emitted_ids) and len(emitted_ids) == len(set(emitted_ids))
    accounting = unique_once and drop_rows == missing
    per_group = {k: {"input_rows": len(v), "emitted_rows": len(emitted.get(k, [])),
                     "emitted_microbatches": sorted(row_batch_ids.get(k, set()))} for k,v in source_groups.items()}
    return {"groups_whole_or_dropped": groups_whole, "no_group_split": no_group_split,
            "accounting_exact_with_drop_metric": accounting, "nonempty_rank_rows": nonempty,
            "capacity_exact_or_respected": capacity_ok, "input_rows": len(rows),
            "emitted_rows": len(emitted_ids), "whole_group_drop_rows": drop_rows,
            "per_group": per_group,
            "contract_pass": groups_whole and no_group_split and accounting and nonempty and capacity_ok}


def run_fixed(FixedCountBatcher, specs):
    rows=[row(*spec) for spec in specs]
    metrics=defaultdict(list)
    batcher=FixedCountBatcher(FiniteStream(rows),2,4)
    batcher.metrics=metrics
    batches=list(batcher)
    return inspect(rows,batches,metrics,2,fixed_capacity=4)


def run_token(TokenBudgetBatcher, specs, budget=6):
    rows=[row(*spec) for spec in specs]
    metrics=defaultdict(list)
    batches=list(TokenBudgetBatcher(FiniteStream(rows),2,budget,metrics))
    return inspect(rows,batches,metrics,2,token_budget=6)


def main():
    FixedCountBatcher, TokenBudgetBatcher=load_batchers()
    cases={
      "fixed_boundary": (lambda: run_fixed(FixedCountBatcher,[
        ("a0","A",3,1),("a1","A",3,1),("a2","A",3,1),
        ("b0","B",2,1),("b1","B",2,1),("c0","C",2,1),("c1","C",2,1),
        ("d0","D",1,1),("e0","E",1,1)]), "A legacy per-row boundary splits B; candidate may reject only whole rollouts"),
      "fixed_oversize": (lambda: run_fixed(FixedCountBatcher,
        [(f"o{i}","O",5,1) for i in range(5)]+[(f"h{i}","H",4,1) for i in range(4)]), "reject O whole, retain H as one full batch"),
      "fixed_infeasible_prefix": (lambda: run_fixed(FixedCountBatcher,
        [("p0","P",1,1)]+[(f"q{i}","Q",4,1) for i in range(4)]), "reject P whole, retain Q as one full batch"),
      "fixed_exact_control": (lambda: run_fixed(FixedCountBatcher,
        [(f"x{i}","X",2,1) for i in range(2)]+[(f"y{i}","Y",2,1) for i in range(2)]), "retain both groups in one exact-size batch"),
      "token_boundary": (lambda: run_token(TokenBudgetBatcher,[
        ("a0","A",2,7),("a1","A",2,3),("b0","B",2,4),("b1","B",2,6),
        ("c0","C",2,2),("c1","C",2,2),("d0","D",2,9),("d1","D",2,1)], budget=10), "retain all four groups at per-row budget 10"),
      "token_exact_fit": (lambda: run_token(TokenBudgetBatcher,
        [(f"t{i}","T",5,n) for i,n in enumerate([3,3,2,2,2])]), "retain exact-fit T using [3,3]/[2,2,2]"),
      "token_infeasible_group": (lambda: run_token(TokenBudgetBatcher,
        [(f"u{i}","U",3,n) for i,n in enumerate([4,4,4])]+[("v0","V",2,3),("v1","V",2,3)]), "reject U whole, retain V"),
      "token_overbudget_member": (lambda: run_token(TokenBudgetBatcher,
        [("w0","W",2,7),("w1","W",2,2),("z0","Z",2,3),("z1","Z",2,3)]), "reject W whole, retain Z")
    }
    out={}
    for name,(fn,expected) in cases.items():
      try: out[name]={"expected":expected,"observed":fn()}
      except Exception as e: out[name]={"expected":expected,"exception":{"type":type(e).__name__,"message":str(e)},"contract_pass":False}
    out["all_cases_pass"]=all(v.get("observed",{}).get("contract_pass",False) for v in out.values())
    print(json.dumps(out,indent=2,sort_keys=True))


if __name__=="__main__": main()
