#!/usr/bin/env python3
"""Execute one frozen, resource-bounded VARE Sprint IX coding-agent campaign."""
from __future__ import annotations
import hashlib, json, os, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import psutil

ROOT=Path(r"/Users/user/Documents/Codex/2026-10-26/vare-flagship-research-sprint-real-coding")
VARE=ROOT/"work/VARE"
EVIDENCE=ROOT/"work/VARE-sprint-iv-evidence-public/results/sprint_ix_20261028/mlx_lm_1777_agentic_repair"
CANDIDATES=ROOT/"work/sprint-ix-mlx1777-candidates"
PROTOCOL=EVIDENCE/"protocol-v3.json"
MODEL=Path("/Users/user/.cache/huggingface/hub/models--mlx-community--Qwen2.5-Coder-7B-Instruct-4bit/snapshots/019cc73c45c770444708a6dd8690c66243cc5c80")

def sha_bytes(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b""):h.update(block)
 return h.hexdigest()
def canonical(value)->bytes:return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def durable(path:Path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 encoded=(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
 with path.open("xb") as stream:
  stream.write(encoded);stream.flush();os.fsync(stream.fileno())
 fd=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def event_chain(path:Path):
 previous="0"*64; events=[]
 for sequence,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
  rec=json.loads(line); event=rec["event"]
  digest=sha_bytes(canonical(event))
  if event.get("sequence")!=sequence or rec.get("previous_event_sha256")!=previous or rec.get("event_sha256")!=digest:
   raise ValueError("event journal integrity failure")
  previous=digest;events.append(event)
 return events,previous
def snapshot():
 memory=psutil.virtual_memory(); swap=psutil.swap_memory()
 disk=__import__("shutil").disk_usage(EVIDENCE)
 return {
  "timestamp_utc":datetime.now(timezone.utc).isoformat(),
  "available_memory_bytes":memory.available,"total_memory_bytes":memory.total,
  "swap_used_bytes":swap.used,"swap_free_bytes":swap.free,"swap_total_bytes":swap.total,
  "disk_free_bytes":disk.free,
 }
def preflight(protocol):
 sample=snapshot(); lim=protocol["resources"]
 checks={
  "minimum_available_memory":sample["available_memory_bytes"]>=lim["minimum_available_memory_bytes"],
  "minimum_free_swap":sample["swap_free_bytes"]>=lim["minimum_free_swap_bytes"],
  "minimum_disk_free":sample["disk_free_bytes"]>=lim["minimum_disk_free_bytes"],
 }
 sample["checks"]=checks;sample["passed"]=all(checks.values())
 return sample
def process_tree_rss(pid:int)->int:
 try:
  parent=psutil.Process(pid)
  return sum(p.memory_info().rss for p in [parent,*parent.children(recursive=True)] if p.is_running())
 except (psutil.NoSuchProcess,psutil.AccessDenied):
  return 0
def kill_group(proc):
 try:os.killpg(proc.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:proc.wait(timeout=5)
 except subprocess.TimeoutExpired:
  try:os.killpg(proc.pid,signal.SIGKILL)
  except ProcessLookupError:pass
def make_command(arm,cell,protocol):
 model=protocol["model"]; budget=protocol["budgets"]; task=protocol["task"]
 out=cell/"agent.json"; event=cell/"events.jsonl"; workspace=CANDIDATES/arm
 interface="lease_bound_transactional_localized_v1" if arm=="C_localized" else "lease_bound_transactional_navigation_v1"
 cmd=[str(protocol["runtime"]["execution_python"]),str(VARE/"scripts/run_local_agent.py"),
  "--workspace",str(workspace),"--model-path",str(MODEL),
  "--model-repository",model["repository"],"--backend","mlx","--mlx-device","metal",
  "--task-root",str(EVIDENCE/"task"),"--diff-base","HEAD",
  "--output",str(out),"--event-journal",str(event),
  "--instruction",protocol["task"]["instruction"],"--seed",str(protocol["sampling"]["seed"]),
  "--max-turns",str(budget["max_turns"]),
  "--max-no-tool-retries",str(budget["max_no_tool_retries"]),
  "--max-new-tokens-per-turn",str(budget["max_new_tokens_per_turn"]),
  "--max-total-new-tokens",str(budget["max_total_new_tokens"]),
  "--max-context-tokens",str(budget["max_context_tokens"]),
  "--wall-seconds",str(budget["agent_wall_seconds"]),
  "--temperature",str(protocol["sampling"]["temperature"]),
  "--top-p",str(protocol["sampling"]["top_p"]),
  "--top-k",str(protocol["sampling"]["top_k"]),
  "--no-do-sample","--edit-format","structured_replace",
  "--generation-protocol",budget["generation_protocol"],
  "--tool-interface",interface,
  "--require-source-read-before-edit","--post-edit-navigation-gate","--post-edit-completion-guidance",
  "--grader-entrypoint",str(EVIDENCE/"task/evaluator/grade_hidden.py"),
  "--public-test-image",protocol["grading"]["image"],
  "--public-test-timeout-s",str(budget["public_test_timeout_seconds"]),
  "--public-test-memory-mb",str(budget["public_test_memory_mib"]),
  "--max-public-test-runs",str(budget["max_public_test_runs"]),
  "--require-split","development"]
 if arm!="C_localized":cmd += ["--navigation-guidance","v2"]
 if arm!="A_autonomous":cmd += ["--no-progress-guard"]
 if arm=="C_localized":cmd += ["--localization-record",str(EVIDENCE/"localization-v3.json")]
 return cmd
def grade(workspace:Path,protocol):
 grader=EVIDENCE/"task/evaluator/grade_hidden.py"; image=protocol["grading"]["image"]
 argv=["docker","run","--rm","--pull=never","--network=none","--read-only","--cap-drop=ALL",
  "--security-opt=no-new-privileges:true","--pids-limit","64","--memory","1024m","--cpus","1",
  "--user","65534:65534","--tmpfs","/tmp:rw,nosuid,nodev,size=16m",
  "--mount",f"type=bind,source={workspace.resolve()},target=/candidate,readonly",
  "--mount",f"type=bind,source={grader.resolve()},target=/grader.py,readonly",
  image,"python","/grader.py","/candidate/mlx_lm/generate.py"]
 env={**os.environ,"DOCKER_CONTEXT":"colima"}
 try:
  proc=subprocess.run(argv,capture_output=True,text=True,timeout=120,env=env,check=False)
 except Exception as exc:
  return {"argv":argv,"error":type(exc).__name__+": "+str(exc),"grader_sha256":sha(grader),
          "candidate_sha256":sha(workspace/"mlx_lm/generate.py"),"image":image,
          "network":"none","root_filesystem":"read-only","candidate_mount":"read-only"}
 try: parsed=json.loads(proc.stdout.strip()) if proc.stdout.strip() else None
 except json.JSONDecodeError:parsed=None
 return {"argv":argv,"exit_code":proc.returncode,"stdout":proc.stdout,"stderr":proc.stderr,
         "parsed":parsed,"grader_sha256":sha(grader),"candidate_sha256":sha(workspace/"mlx_lm/generate.py"),
         "image":image,"network":"none","root_filesystem":"read-only","candidate_mount":"read-only"}
def validate(protocol):
 if protocol.get("status")!="FROZEN_BEFORE_MODEL_INFERENCE":raise ValueError("protocol not frozen")
 expected=protocol.get("protocol_sha256")
 payload=dict(protocol);payload.pop("protocol_sha256",None)
 if sha_bytes(canonical(payload))!=expected:raise ValueError("protocol hash mismatch")
 for relative,digest in protocol["asset_hashes"].items():
  path=EVIDENCE/relative
  if not path.is_file() or sha(path)!=digest:raise ValueError("frozen asset drift: "+relative)
 for relative,digest in protocol["runtime_source_hashes"].items():
  path=VARE/relative
  if not path.is_file() or sha(path)!=digest:raise ValueError("controller source drift: "+relative)
 for arm,path in protocol["candidate_workspaces"].items():
  src=Path(path)/"mlx_lm/generate.py"
  if not src.is_file() or sha(src)!=protocol["task"]["baseline_source_sha256"]:raise ValueError("candidate baseline drift: "+arm)
 model_files={}
 for p in sorted(MODEL.iterdir()):
  if p.is_file():model_files[p.name]=sha(p)
 if model_files!=protocol["model"]["files_sha256"]:raise ValueError("cached model snapshot drift")
def run_one(arm,protocol):
 arm_spec=protocol["arms"][arm]
 cell=EVIDENCE/"attempts-v3"/arm
 cell.mkdir(parents=True,exist_ok=False)
 gate=preflight(protocol);durable(cell/"preflight.json",gate)
 if not gate["passed"]:
  durable(cell/"status.json",{"arm":arm,"status":"PREFLIGHT_REJECTED","model_invocations":0,"preflight":gate})
  return {"arm":arm,"status":"PREFLIGHT_REJECTED","resource_abort":False}
 workspace=Path(protocol["candidate_workspaces"][arm])
 before_source=sha(workspace/"mlx_lm/generate.py")
 before_swap=psutil.swap_memory().used
 cmd=make_command(arm,cell,protocol)
 env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
  "TOKENIZERS_PARALLELISM":"false","HF_HUB_DISABLE_TELEMETRY":"1",
  "GIT_CONFIG_NOSYSTEM":"1","GIT_CONFIG_GLOBAL":os.devnull,
  "GIT_NO_REPLACE_OBJECTS":"1","DOCKER_CONTEXT":"colima"}
 durable(cell/"invocation.json",{"arm":arm,"protocol_sha256":protocol["protocol_sha256"],
  "created_at_utc":datetime.now(timezone.utc).isoformat(),"command":cmd,
  "env_allowlist":{k:env[k] for k in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","TOKENIZERS_PARALLELISM","HF_HUB_DISABLE_TELEMETRY","GIT_CONFIG_NOSYSTEM","GIT_CONFIG_GLOBAL","GIT_NO_REPLACE_OBJECTS","DOCKER_CONTEXT")},
  "candidate_source_sha256_before":before_source,"candidate_workspace":str(workspace),
  "arm_spec":arm_spec})
 start=time.monotonic(); samples=[]; abort=None
 with (cell/"stdout.log").open("xb") as stdout,(cell/"stderr.log").open("xb") as stderr:
  proc=subprocess.Popen(cmd,cwd=VARE,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
  while proc.poll() is None:
   sample=snapshot();sample["elapsed_seconds"]=round(time.monotonic()-start,3)
   sample["process_tree_rss_bytes"]=process_tree_rss(proc.pid)
   sample["swap_growth_bytes"]=sample["swap_used_bytes"]-before_swap
   samples.append(sample)
   checks=protocol["resources"]
   if sample["elapsed_seconds"]>checks["max_outer_wall_seconds"]:abort="outer_wall_limit"
   elif sample["process_tree_rss_bytes"]>checks["max_process_tree_rss_bytes"]:abort="process_rss_limit"
   elif sample["swap_growth_bytes"]>checks["max_swap_growth_bytes"]:abort="host_swap_growth_limit"
   elif sample["swap_free_bytes"]<checks["minimum_free_swap_bytes"]:abort="minimum_free_swap_violation"
   elif sample["available_memory_bytes"]<checks["minimum_available_memory_bytes"]:abort="minimum_available_memory_violation"
   elif sample["disk_free_bytes"]<checks["minimum_disk_free_bytes"]:abort="minimum_disk_free_violation"
   if abort:
    kill_group(proc);break
   time.sleep(checks["sample_interval_seconds"])
  returncode=proc.wait()
 final=snapshot();final["elapsed_seconds"]=round(time.monotonic()-start,3)
 final["process_tree_rss_bytes"]=process_tree_rss(proc.pid);final["swap_growth_bytes"]=final["swap_used_bytes"]-before_swap
 samples.append(final)
 if abort is None:
  checks=protocol["resources"]
  if final["swap_growth_bytes"]>checks["max_swap_growth_bytes"]:abort="host_swap_growth_limit_at_exit"
  elif final["swap_free_bytes"]<checks["minimum_free_swap_bytes"]:abort="minimum_free_swap_violation_at_exit"
  elif final["available_memory_bytes"]<checks["minimum_available_memory_bytes"]:abort="minimum_available_memory_violation_at_exit"
  elif final["disk_free_bytes"]<checks["minimum_disk_free_bytes"]:abort="minimum_disk_free_violation_at_exit"
 with (cell/"resource-samples.jsonl").open("xb") as stream:
  for sample in samples:
   stream.write((json.dumps(sample,sort_keys=True)+"\n").encode());stream.flush();os.fsync(stream.fileno())
 event_path=cell/"events.jsonl"
 events=[];chain_tail=None
 if event_path.is_file():
  try:events,chain_tail=event_chain(event_path);chain_status="VALID"
  except Exception as exc:chain_status=f"INVALID:{type(exc).__name__}:{exc}"
 else:chain_status="MISSING"
 agent_path=cell/"agent.json"; agent=None
 if agent_path.is_file():
  try:agent=json.loads(agent_path.read_text())
  except Exception:pass
 source_after=sha(workspace/"mlx_lm/generate.py")
 diff=subprocess.run(["git","-C",str(workspace),"diff","--binary","HEAD","--","mlx_lm/generate.py"],capture_output=True,check=True).stdout
 (cell/"candidate.patch").write_bytes(diff)
 changed=subprocess.run(["git","-C",str(workspace),"status","--short"],capture_output=True,text=True,check=True).stdout.splitlines()
 patch_sha=sha_bytes(diff)
 grade_result=(grade(workspace,protocol) if abort is None else {"status":"SKIPPED_AFTER_RESOURCE_ABORT"})
 durable(cell/"independent-grader.json",grade_result)
 public=(agent or {}).get("public_test_result")
 grade_pass=bool(grade_result.get("exit_code")==0 and (grade_result.get("parsed") or {}).get("passed") is True)
 completed=bool(agent and agent.get("stop_reason")=="agent_finished" and returncode==0 and abort is None)
 verified=bool(completed and changed and source_after!=before_source and public and public.get("passed") is True and grade_pass and chain_status=="VALID")
 status="VERIFIED_REPAIR" if verified else ("RESOURCE_ABORT" if abort else ("AGENT_COMPLETED_UNGRADED_OR_INCORRECT" if completed else "INCOMPLETE_OR_RUNTIME_FAILURE"))
 result={"arm":arm,"status":status,"exit_code":returncode,"resource_abort":abort,
  "elapsed_seconds":round(time.monotonic()-start,3),"peak_sampled_process_tree_rss_bytes":max((x["process_tree_rss_bytes"] for x in samples),default=0),
  "swap_growth_bytes":max((x["swap_growth_bytes"] for x in samples),default=0),
  "candidate_source_sha256_before":before_source,"candidate_source_sha256_after":source_after,
  "candidate_changed":source_after!=before_source,"candidate_patch_sha256":patch_sha,"candidate_patch_bytes":len(diff),
  "candidate_git_status":changed,"event_chain_status":chain_status,"event_chain_tail":chain_tail,
  "event_count":len(events),"event_types":[x.get("event_type") for x in events],
  "agent_stop_reason":(agent or {}).get("stop_reason"),"generated_tokens":(agent or {}).get("generated_tokens"),
  "public_test_result":public,"independent_grade_passed":grade_pass,"independent_grade":grade_result.get("parsed"),
  "verified_repair":verified,"claim_unit":"one task family; one attempt per arm; exploratory descriptive only"}
 durable(cell/"attempt-summary.json",result)
 return result

def main():
 protocol=json.loads(PROTOCOL.read_text())
 validate(protocol)
 summary={"schema_version":1,"experiment_id":protocol["experiment_id"],"protocol_sha256":protocol["protocol_sha256"],
  "status":"RUNNING","started_at_utc":datetime.now(timezone.utc).isoformat(),"attempts":[]}
 for arm in protocol["attempt_order"]:
  name=arm["arm"]
  result=run_one(name,protocol);summary["attempts"].append(result)
  if result["resource_abort"] or result["status"]=="PREFLIGHT_REJECTED":
   for skipped in [x["arm"] for x in protocol["attempt_order"][len(summary["attempts"]):]]:
    durable(EVIDENCE/"attempts-v3"/(skipped+".not-run.json"),{"arm":skipped,"status":"NOT_RUN_AFTER_RESOURCE_OR_PREFLIGHT_GATE","model_invocations":0})
   summary["status"]="STOPPED_BY_RESOURCE_OR_PREFLIGHT_GATE";break
 else:
  summary["status"]="ALL_PREREGISTERED_ARMS_COMPLETED"
 summary["completed_at_utc"]=datetime.now(timezone.utc).isoformat()
 out=EVIDENCE/"campaign-summary-v3.json"
 durable(out,summary)
 print(json.dumps({"status":summary["status"],"attempts":[{"arm":x["arm"],"status":x["status"],"generated_tokens":x.get("generated_tokens"),"verified_repair":x.get("verified_repair"),"abort":x.get("resource_abort")} for x in summary["attempts"]],"summary_path":str(out)},sort_keys=True))
if __name__=="__main__":main()

