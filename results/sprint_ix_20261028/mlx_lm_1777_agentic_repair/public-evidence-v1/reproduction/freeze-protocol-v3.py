#!/usr/bin/env python3
from __future__ import annotations
import hashlib,importlib.util,json,os
from datetime import datetime,timezone
from pathlib import Path
import psutil
E=Path(r"/Users/user/Documents/Codex/2026-10-26/vare-flagship-research-sprint-real-coding/work/VARE-sprint-iv-evidence-public/results/sprint_ix_20261028/mlx_lm_1777_agentic_repair")
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def require(v,msg):
 if not v:raise SystemExit(msg)
if any((E/n).exists() for n in ("protocol-v3.json","protocol-v3.sha256","protocol-v3-freeze-record.json")):raise SystemExit("v3 already frozen")
v2=json.loads((E/"protocol-v2.json").read_text());v2hash=v2["protocol_sha256"]
payload=dict(v2);payload.pop("protocol_sha256",None)
require(sha_bytes(canon(payload))==v2hash,"v2 protocol integrity failure")
report=json.loads((E/"preflight/v3-launcher-contracts/report.json").read_text())
runner_path=E/"reproduction/run-agent-campaign-v3.py"
require(report["status"]=="PASS_NO_MODEL_LOADED" and report["model_process_started"] is False and report["generation_started"] is False and report["model_weights_loaded"] is False,"v3 no-model preflight failed")
require(report["runner_sha256"]==sha(runner_path),"v3 runner drift")
# Frozen compute limits carry through unchanged.
mem=psutil.virtual_memory();swap=psutil.swap_memory();disk=psutil.disk_usage(E)
limits=v2["resources"]
resource={"captured_at_utc":datetime.now(timezone.utc).isoformat(),"available_memory_bytes":mem.available,"memory_total_bytes":mem.total,"swap_free_bytes":swap.free,"swap_used_bytes":swap.used,"swap_total_bytes":swap.total,"disk_free_bytes":disk.free,
 "checks":{"minimum_available_memory":mem.available>=limits["minimum_available_memory_bytes"],"minimum_free_swap":swap.free>=limits["minimum_free_swap_bytes"],"minimum_disk_free":disk.free>=limits["minimum_disk_free_bytes"]}}
require(all(resource["checks"].values()),f"v3 resource preflight failed: {resource}")
p={**v2}
p["protocol_id"]="vare-sprint-ix-progress-aware-coding-agent-v3"
p["experiment_id"]="mlx-lm-prefill-history-development-repair-a-b-c-20261028-v3"
p["status"]="FROZEN_BEFORE_MODEL_INFERENCE"
p["frozen_at_utc"]=datetime.now(timezone.utc).isoformat()
p["supersedes_protocol_sha256"]=v2hash
p["supersession_reason"]="V2's Python invocation failed before child process start because the launcher used the Python version string instead of the executable path. V3 now uses runtime.execution_python and includes an executable version probe; all arm commands were constructed and validated before inference."
p["invalidation_records"]=["protocol-v1-invalidation.json","protocol-v1-invalidation-complete.json","protocol-v2-invalidation.json","campaign-summary-v2-preinference-failure.json"]
p["runtime"]={**p["runtime"],"campaign_runner_sha256":sha(runner_path)}
p["resource_preflight_at_freeze_v3"]=resource
assets=dict(p["asset_hashes"])
extra=[
 "reproduction/run-agent-campaign-v3.py",
 "reproduction/preflight-v3-runner.py",
 "reproduction/freeze-protocol-v3.py",
 "protocol-v2-invalidation.json",
 "attempts-v2/A_autonomous/launcher-failure.json",
 "attempts-v2/A_autonomous/invocation.json",
 "attempts-v2/A_autonomous/preflight.json",
 "attempts-v2/A_autonomous/stdout.log",
 "attempts-v2/A_autonomous/stderr.log",
 "attempts-v2/B_progress_aware.not-run.json",
 "attempts-v2/C_localized.not-run.json",
 "campaign-summary-v2-preinference-failure.json",
 "preflight/v3-launcher-contracts/report.json",
]
for rel in extra:assets[rel]=sha(E/rel)
p["asset_hashes"]=dict(sorted(assets.items()))
p["protocol_sha256"]=sha_bytes(canon({k:v for k,v in p.items() if k!="protocol_sha256"}))
spec=importlib.util.spec_from_file_location("vare_sprint_ix_runner_v3",runner_path)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.validate(p)
commands=[]
for arm_record in p["attempt_order"]:
 arm=arm_record["arm"]
 command=runner.make_command(arm,E/"preflight/v3-launcher-contracts"/arm,p)
 require(command[0]==p["runtime"]["execution_python"],"wrong executable bound in final command")
 require(Path(command[0]).is_file(),"final executable unavailable")
 if arm=="C_localized":require("--navigation-guidance" not in command,"localized arm guidance contract failed")
 commands.append({"arm":arm,"command_sha256":sha_bytes(json.dumps(command,separators=(",",":")).encode()),"executable":command[0]})
freeze={"schema_version":1,"status":"FROZEN_BEFORE_MODEL_INFERENCE","protocol_sha256":p["protocol_sha256"],"supersedes_protocol_sha256":v2hash,"created_at_utc":datetime.now(timezone.utc).isoformat(),"model_process_started":False,"model_weights_loaded":False,"generation_started":False,"validation":{"v2_integrity":True,"v3_asset_hashes":True,"model_checkpoint":True,"controller_sources":True,"candidate_baselines":True,"python_executable":True,"python_probe":True,"A_B_C_command_construction":True,"localized_guidance_contract":True,"resource_preflight":resource["checks"]},"commands":commands}
for name,data in (
 ("protocol-v3.json",(json.dumps(p,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()),
 ("protocol-v3.sha256",(p["protocol_sha256"]+"  canonical-json-payload\n").encode()),
 ("protocol-v3-freeze-record.json",(json.dumps(freeze,indent=2,sort_keys=True)+"\n").encode())):
 path=E/name
 with path.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
fd=os.open(E,os.O_RDONLY)
try:os.fsync(fd)
finally:os.close(fd)
print(json.dumps({"status":p["status"],"protocol_sha256":p["protocol_sha256"],"superseded_v2":v2hash,"resource_gates":resource["checks"],"runner_sha256":p["runtime"]["campaign_runner_sha256"],"commands":commands},sort_keys=True))

