"""Independent byte/tensor/continuity audit for the CPU v2 experiment."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import statistics


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    """Independent encoding oracle, including dtype/shape, Adam and RNG bytes."""
    import torch
    digest = hashlib.sha256()
    def walk(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((str(item.dtype),list(item.shape))).encode())
            digest.update(item.detach().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item,key=str):
                digest.update(str(key).encode())
                walk(item[key])
        elif isinstance(item,(tuple,list)):
            for member in item:
                walk(member)
        else:
            digest.update(repr(item).encode())
    walk(value)
    return digest.hexdigest()


def checkpoint(case, policy_id, mode):
    if mode == "minimal":
        path = case / f"{policy_id}.pt"
        return path.read_bytes(), sha(path)
    uri = (case / "checkpoints/artifacts.sqlite").resolve().as_uri()+"?mode=ro"
    # The checkpoint-store schema is inspected independently rather than
    # trusting a supplied checkpoint digest or the benchmark's read helper.
    with sqlite3.connect(uri,uri=True) as db:
        row = db.execute("SELECT digest,size FROM artifacts WHERE policy_id=?",(policy_id,)).fetchone()
    need(row is not None,"checkpoint missing")
    raw = (case / "checkpoints/objects" / row[0]).read_bytes()
    need(len(raw)==row[1] and hashlib.sha256(raw).hexdigest()==row[0],"checkpoint integrity mismatch")
    return raw,row[0]


def audit(root, check_tensors=False, model_dir=None):
    root = Path(root)
    sums = read(root / "SHA256SUMS.json")
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!="SHA256SUMS.json"}
    need(actual==set(sums),"artifact inventory differs")
    for name,digest in sums.items():
        need(sha(root/name)==digest,f"changed artifact: {name}")
    manifest,summary,rows=read(root/"manifest.json"),read(root/"summary.json"),read(root/"progress.json")
    for name,digest in manifest["sources"].items():
        need(sha(root/"sources"/name)==digest,"source snapshot differs")
    protocol=read(root/"sources/benchmarks/e2e_rl_continuation_v2/protocol.json")
    args=manifest["arguments"]
    locked=(args["workers"]==protocol["workers"] and args["seeds"]==protocol["seeds"] and
            args["rounds"]==protocol["rounds"] and args["tasks"]==protocol["rollouts"] and
            args["eval_tasks"]==protocol["evaluation_tasks"] and not args["skip_faults"])
    need(manifest["protocol_conformant"]==summary["protocol_conformant"]==locked,"reduced run claims locked protocol")
    expected_id=protocol["schema"] if locked else protocol["schema"]+".diagnostic"
    need(manifest["protocol_id"]==summary["protocol_id"]==expected_id,"protocol identity differs")
    expected_cases={f"normal-w{w}-s{s}-{m}" for w in args["workers"] for s in args["seeds"] for m in ("minimal","guarded")}
    if not args["skip_faults"]:
        expected_cases.update(f"kill-{cut}-s{s}-guarded" for s in args["seeds"] for cut in protocol["cutpoints"])
    need({r["directory"] for r in rows}==expected_cases and len(rows)==len(expected_cases),"campaign case budget differs")
    if check_tensors or model_dir:
        import torch
        torch.set_num_threads(1)
    if model_dir:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        for name,digest in protocol["model_files"].items():
            need(sha(Path(model_dir)/name)==digest,"model revision differs")
        model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True).eval()
        tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
        choices=[tokenizer.encode(" A")[0],tokenizer.encode(" B")[0]]
    references,round_records={},{}
    tensor_states=0
    for row in rows:
        case=root/row["directory"]
        config=read(case/"config.json")
        need(row["protocol_id"]==config["protocol_id"]==expected_id,"case protocol identity differs")
        need(row["protocol_conformant"]==config["protocol_conformant"]==locked,"case conformance differs")
        commits=[read(case/"rounds"/f"r{i:04d}"/"committed.json") for i in range(args["rounds"])]
        need([r["completed_round"] for r in commits]==list(range(args["rounds"])),"round gap or duplicate")
        need(commits[-1]["active_state_sha256"]==row["final_state_sha256"],"final state differs")
        round_records[row["directory"]]=commits
        if row["cutpoint"]=="none":
            references[(row["workers"],row["seed"],row["mode"])]=commits
        attempts=[json.loads(line) for line in (case/"optimizer_attempts.jsonl").read_text().splitlines()]
        expected_attempts=args["rounds"]+int(row["cutpoint"] in ("post_optimizer","candidate_saved","journal_prepared"))
        need(len(attempts)==row["physical_optimizer_attempts"]==expected_attempts,"physical attempt accounting differs")
        for attempt in attempts:
            committed=commits[attempt["round_index"]]
            need(attempt["candidate_state_sha256"]==committed["candidate_state_sha256"] and
                 attempt["candidate_model_sha256"]==committed["update"]["parameters_after_sha256"],
                 "physical optimizer attempt differs from committed candidate")
        trace=[json.loads(line) for line in (case/"trace.jsonl").read_text().splitlines()]
        spans=[r["duration_s"] for r in trace if r["stage"]=="round"]
        need(row["round_wall_s"]==sum(spans),"round wall differs")
        need(row["tasks_s"]==row["tokens_s"]==args["tasks"]*len(spans)/sum(spans),"throughput differs")
        need(row["accepted_updates"]>=2,"no actual continuation")
        previous_pid,previous_step="policy-0",0
        for index,commit in enumerate(commits):
            round_dir=case/"rounds"/f"r{index:04d}"
            train,heldout=read(round_dir/"training_tasks.json"),read(round_dir/"evaluation_tasks.json")
            need(len(train)==args["tasks"] and len(heldout)==args["eval_tasks"],"task budget differs")
            need(not ({r["prompt"] for r in train}&{r["prompt"] for r in heldout}),"optimization/evaluation overlap")
            raw,old_sha=checkpoint(case,previous_pid,row["mode"])
            state=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True) if check_tensors or model_dir else None
            if state:
                need(state["step"]==previous_step,"incumbent Adam step differs")
                incumbent_model=fingerprint(state["model"])
            for file in round_dir.glob('rollouts-attempt*.jsonl'):
                rollouts=[json.loads(line) for line in file.read_text().splitlines()]
                need(len(rollouts)==args["tasks"],"incomplete rollout attempt")
                for rollout in rollouts:
                    worker=rollout["metadata"]["worker"]
                    need(worker["loaded_checkpoint_sha256"]==old_sha and worker["loaded_step"]==previous_step,"stale worker served a new policy label")
                    need(rollout["policy_id"]==previous_pid and rollout["policy_version"]==previous_step,"policy provenance differs")
                    if state:
                        need(worker["loaded_model_sha256"]==incumbent_model,"worker tensor acknowledgment differs")
            candidate=commit["candidate_id"]
            candidate_raw,candidate_sha=checkpoint(case,candidate,row["mode"])
            update=commit["update"]
            need(update["batch_size"]==args["tasks"] and update["parameters_before_sha256"]!=update["parameters_after_sha256"],"mock/no-op/wrong-budget update")
            for path in round_dir.glob('batch-attempt*.json'):
                batch=read(path)
                need(batch["incumbent_id"]==previous_pid and batch["checkpoint_sha256"]==old_sha and batch["step"]==previous_step,"training batch incumbent differs")
                need(len(batch["members"])==args["tasks"] and len({m["created_step"] for m in batch["members"]})==args["tasks"],"duplicate or incomplete optimizer batch")
                need(all(m["policy_id"]==previous_pid and m["policy_version"]==previous_step and m["loaded_checkpoint_sha256"]==old_sha for m in batch["members"]),"stale evidence reached optimizer")
                if state:
                    need(batch["model_sha256"]==incumbent_model,"training batch tensor hash differs")
            if state:
                candidate_state=torch.load(io.BytesIO(candidate_raw),map_location="cpu",weights_only=True)
                need(fingerprint(candidate_state)==commit["candidate_state_sha256"],"candidate model/Adam/RNG fingerprint differs")
                need(candidate_state["step"]==previous_step+1,"candidate step not exactly next")
                need(fingerprint(candidate_state["model"])==update["parameters_after_sha256"] and incumbent_model==update["parameters_before_sha256"],"update tensors do not match retained checkpoint")
                need(fingerprint(candidate_state["optimizer"])==update["optimizer_state_sha256"],"Adam moments differ")
                tensor_states+=1
            expected_active=candidate if commit["accepted"] else previous_pid
            expected_step=previous_step+int(commit["accepted"])
            need(commit["active_policy"]==expected_active and commit["accepted_updates"]==expected_step,"double-applied or missing committed update")
            if state:
                expected_state=candidate_state if commit["accepted"] else state
                need(fingerprint(expected_state)==commit["active_state_sha256"],"active state differs after promotion/reject")
            need(commit["stale_replay_rejected"]==int(previous_step>0),"stale replay canary was consumed")
            for path in round_dir.glob('evaluation-*.json'):
                record=read(path)
                need(len(record["scores"])==args["eval_tasks"] and statistics.fmean(record["scores"].values())==record["report"]["primary"],"independent evaluation arithmetic differs")
                evaluated_raw,evaluated_sha=checkpoint(case,record["policy_id"],row["mode"])
                need(record["checkpoint_sha256"]==evaluated_sha,"evaluation checkpoint differs")
                if model_dir:
                    evaluated_state=torch.load(io.BytesIO(evaluated_raw),map_location="cpu",weights_only=True)
                    model.load_state_dict(evaluated_state["model"])
                    with torch.inference_mode():
                        for task in heldout:
                            score=float(model(torch.tensor([tokenizer.encode(task["prompt"])] )).logits[0,-1,choices].softmax(-1)[0])
                            need(abs(score-record["scores"][task["id"]])<1e-7,"independent checkpoint score does not replay")
            previous_pid,previous_step=expected_active,expected_step
        if row["mode"]=="guarded":
            need(row["final_journal"]["pending_tx_id"] is None and row["final_journal"]["round_owner_id"] is None,"journal unresolved")
            with sqlite3.connect((case/"promotion.sqlite").resolve().as_uri()+"?mode=ro",uri=True) as db:
                active=db.execute("SELECT policy_id,checkpoint_sha256 FROM active").fetchone()
                duplicate=db.execute("SELECT candidate_id FROM transactions WHERE state='committed' GROUP BY candidate_id HAVING COUNT(*)>1").fetchall()
                pending=db.execute("SELECT COUNT(*) FROM transactions WHERE state='prepared'").fetchone()[0]
                owners=db.execute("SELECT COUNT(*) FROM runtime_round").fetchone()[0]
            need(not duplicate and pending==owners==0,"duplicate commit or unresolved durable owner")
            need(active==(commits[-1]["active_policy"],commits[-1]["active_checkpoint_sha256"]),"actual journal pointer differs")
        if row["cutpoint"]!="none":
            killed,progress=read(case/"kill_record.json"),read(case/"post_restart_update.json")
            need(killed["returncode"]==-9 and killed["old_descendants"] and killed["live_old_descendants"]==0,"learner/descendant SIGKILL not established")
            need(row["recovery_to_next_update_s"]==progress["time"]-killed["killed_at"]>0,"recovery clock differs")
            expected_round=protocol["kill_round"] if row["cutpoint"] in ("post_optimizer","candidate_saved","journal_prepared") else protocol["kill_round"]+1
            need(progress["completed_round"]==expected_round,"recovery did not execute the next logical optimizer round")
    for row in rows:
        records=round_records[row["directory"]]
        if row["cutpoint"]=="none":
            other="minimal" if row["mode"]=="guarded" else "guarded"
            reference=references[(row["workers"],row["seed"],other)]
        else:
            reference=references[(2,row["seed"],"guarded")]
        for observed,expected in zip(records,reference,strict=True):
            need(observed["active_state_sha256"]==expected["active_state_sha256"] and observed["candidate_state_sha256"]==expected["candidate_state_sha256"] and observed["accepted"]==expected["accepted"],"continuity reference mismatch")
            def update(r):
                return {k:v for k,v in r["update"].items() if k!="optimizer_steps"}
            need(update(observed)==update(expected),"next numerical optimizer update differs")
    need(summary["cases"]==len(rows) and summary["completed_rounds"]==sum(r["completed_rounds"] for r in rows),"summary budget differs")
    need(summary["physical_optimizer_attempts"]==sum(r["physical_optimizer_attempts"] for r in rows),"summary attempts differ")
    for group in summary["normal"]:
        values=[r["tasks_s"] for r in rows if r["cutpoint"]=="none" and r["workers"]==group["workers"] and r["mode"]==group["mode"]]
        need(group["tasks_s_mean"]==statistics.fmean(values) and group["n"]==len(values),"mean differs")
        if len(values)>1:
            need(group["tasks_s_sd"]==statistics.stdev(values),"seed variance differs")
    return {"pass":True,"cases":len(rows),"protocol_conformant":locked,"artifacts_hashed":len(sums),
            "candidate_tensor_states_verified":tensor_states,"checkpoint_scores_replayed":bool(model_dir)}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--check-tensors",action="store_true")
    parser.add_argument("--model-dir",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.directory,args.check_tensors,args.model_dir),sort_keys=True))
