import hashlib,importlib.util,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
e=Path(r"/Users/user/Documents/Codex/2026-10-26/vare-flagship-research-sprint-real-coding/work/VARE-sprint-iv-evidence-public/results/sprint_ix_20261028/mlx_lm_1777_agentic_repair")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
runner_path=e/"reproduction/run-agent-campaign-v3.py"
spec=importlib.util.spec_from_file_location("vare_sprint_ix_runner_v3",runner_path)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
v2=json.loads((e/"protocol-v2.json").read_text())
p={**v2}
p["protocol_id"]="vare-sprint-ix-progress-aware-coding-agent-v3"
p["experiment_id"]="mlx-lm-prefill-history-development-repair-a-b-c-20261028-v3"
p["supersedes_protocol_sha256"]=v2["protocol_sha256"]
p["supersession_reason"]="V2 launcher failed before process start because runtime.python held a version label. V3 launches runtime.execution_python and checks the executable with a no-model probe. No model was loaded or generation started."
p["runtime"]={**p["runtime"],"campaign_runner_sha256":sha(runner_path)}
assets=dict(p["asset_hashes"])
extra=[
 "reproduction/run-agent-campaign-v3.py",
 "protocol-v2-invalidation.json",
 "attempts-v2/A_autonomous/launcher-failure.json",
 "attempts-v2/A_autonomous/invocation.json",
 "attempts-v2/A_autonomous/preflight.json",
 "attempts-v2/A_autonomous/stdout.log",
 "attempts-v2/A_autonomous/stderr.log",
 "attempts-v2/B_progress_aware.not-run.json",
 "attempts-v2/C_localized.not-run.json",
 "campaign-summary-v2-preinference-failure.json",
]
for rel in extra:assets[rel]=sha(e/rel)
p["asset_hashes"]=dict(sorted(assets.items()))
p["protocol_sha256"]=hashlib.sha256(canon({k:v for k,v in p.items() if k!="protocol_sha256"})).hexdigest()
runner.validate(p)
arms=[]
for arm in ("A_autonomous","B_progress_aware","C_localized"):
 command=runner.make_command(arm,e/"preflight/v3-launcher-contracts"/arm,p)
 executable=Path(command[0])
 assert executable.is_file(),f"interpreter missing: {executable}"
 assert command[0]==p["runtime"]["execution_python"]
 probe=subprocess.run([command[0],"-c","import sys; assert sys.version_info[:3] == (3, 12, 12); print(sys.executable)"],capture_output=True,text=True,check=True,timeout=10)
 if arm=="C_localized":assert "--navigation-guidance" not in command and "--localization-record" in command
 else:assert "--navigation-guidance" in command and command[command.index("--navigation-guidance")+1]=="v2"
 arms.append({"arm":arm,"executable":command[0],"python_probe_stdout":probe.stdout.strip(),"command_sha256":hashlib.sha256(json.dumps(command,separators=(",",":")).encode()).hexdigest(),"tool_interface":command[command.index("--tool-interface")+1],"localized_record":"--localization-record" in command,"navigation_guidance_v2":"--navigation-guidance" in command})
report={"schema_version":1,"status":"PASS_NO_MODEL_LOADED","created_at_utc":datetime.now(timezone.utc).isoformat(),"superseded_v2_protocol_sha256":v2["protocol_sha256"],"prospective_v3_protocol_sha256_before_freeze_assets":p["protocol_sha256"],"runner_sha256":sha(runner_path),"model_weights_loaded":False,"generation_started":False,"model_process_started":False,"checks":{"v2_protocol_valid":True,"v3_frozen_assets_valid":True,"model_snapshot_sha256_valid":True,"controller_hashes_valid":True,"candidate_baselines_valid":True,"python_executable_exists":True,"python_version_probe":True,"three_commands_constructed":True,"localized_guidance_v2_omitted":True},"arms":arms}
out=e/"preflight/v3-launcher-contracts/report.json"
with out.open("xb") as f:f.write((json.dumps(report,indent=2,sort_keys=True)+"\n").encode())
print(json.dumps({"status":report["status"],"arms":arms,"runner_sha256":report["runner_sha256"],"protocol_v2":report["superseded_v2_protocol_sha256"]},sort_keys=True))

