#!/usr/bin/env python3
"""Run one frozen, offline Qwen3-4B Development edit attempt."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

import psutil

PROFILE="(version 1) (allow default) (deny network*)"
ROOT=Path(__file__).resolve().parents[2]

def h(data: bytes) -> str:return hashlib.sha256(data).hexdigest()
def hf(path: Path) -> str:return h(path.read_bytes())
def write_new(path: Path, obj) -> None:
    b=(json.dumps(obj,sort_keys=True,indent=2)+"\n").encode()
    with path.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
def snapshot(path: Path) -> dict:
    v=psutil.virtual_memory();s=psutil.swap_memory()
    return {"timestamp_utc":datetime.now(timezone.utc).isoformat(),"available_memory_bytes":v.available,
      "total_memory_bytes":v.total,"free_swap_bytes":s.free,"swap_used_bytes":s.used,
      "disk_free_bytes":shutil.disk_usage(path).free}
def process_rss(pid: int) -> int:
    try:
      p=psutil.Process(pid);return sum(x.memory_info().rss for x in [p,*p.children(recursive=True)] if x.is_running())
    except (psutil.NoSuchProcess,psutil.AccessDenied):return 0
def manifest(root: Path) -> None:
    rows=[]
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.name!="manifest.json"):
      rows.append({"path":p.relative_to(root).as_posix(),"size_bytes":p.stat().st_size,"sha256":hf(p)})
    write_new(root/"manifest.json",{"schema_version":1,"algorithm":"sha256","files":rows})
def git(repo: Path,*args: str,**kwargs):
    return subprocess.run(["git","-C",str(repo),*args],check=True,capture_output=True,**kwargs)
def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--protocol-sha256',required=True)
    ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--lock-sha256',required=True);ap.add_argument('--preflight-only',action='store_true')
    args=ap.parse_args(); pp=args.protocol.resolve();lp=args.lock.resolve()
    protocol=json.loads(pp.read_text());lock=json.loads(lp.read_text())
    if hf(pp)!=args.protocol_sha256 or lock.get('protocol_sha256')!=args.protocol_sha256:raise RuntimeError('protocol hash mismatch')
    if hf(lp)!=args.lock_sha256:raise RuntimeError('lock hash mismatch')
    if hf(Path(__file__).resolve())!=lock['model_runner_sha256']:raise RuntimeError('runner hash mismatch')
    impl=protocol['runtime']['implementation_sha256']
    for rel,expected in impl.items():
      if hf(ROOT/rel)!=expected:raise RuntimeError('implementation changed: '+rel)
    task_root=(ROOT/protocol['task']['task_root']).resolve()
    for rel,expected in protocol['task']['input_sha256'].items():
      if hf(task_root/rel)!=expected:raise RuntimeError('task input changed: '+rel)
    if (task_root/'evaluator').exists() or list(task_root.glob('test*.py')):
      raise RuntimeError('grader or test file is present in model-visible task root')
    model=Path(protocol['model']['snapshot_path']).expanduser().resolve()
    actual={p.name:hf(p) for p in model.iterdir() if p.is_file()}
    if actual!=protocol['model']['files_sha256']:raise RuntimeError('model snapshot file inventory/hash mismatch')
    py=Path(protocol['runtime']['model_python']).expanduser().absolute()
    monpy=Path(protocol['runtime']['monitor_python']).expanduser().absolute()
    freeze=subprocess.check_output([str(py),'-m','pip','freeze','--all'],text=True).encode()
    if h(freeze)!=protocol['runtime']['model_venv_freeze_sha256']:raise RuntimeError('model environment freeze mismatch')
    repo=(ROOT/protocol['source']['repo_path']).resolve();base=protocol['source']['base_commit']
    if git(repo,'rev-parse',f'{base}^{{commit}}',text=True).stdout.strip()!=base:raise RuntimeError('base revision missing')
    workspace=(ROOT/protocol['execution']['candidate_workspace']).resolve()
    output=(ROOT/protocol['execution']['output_root']).resolve()
    if workspace.exists() or output.exists():raise RuntimeError('refusing to overwrite workspace or output')
    # No-generation validation path: it resolves every pinned input, checks host
    # resources and model/runner CLI imports, but never creates a candidate workspace.
    if args.preflight_only:
      pf=(ROOT/protocol['execution']['preflight_output']).resolve()
      if pf.exists():raise RuntimeError('refusing to overwrite preflight record')
      lim=protocol['resources'];state=snapshot(ROOT)
      if platform.system()!='Darwin' or platform.machine()!='arm64':raise RuntimeError('host is not pinned Darwin arm64')
      for key in ('available_memory_bytes','free_swap_bytes','disk_free_bytes'):
        if state[key]<lim['minimum_'+key]:raise RuntimeError('resource preflight failed: '+key)
      base_bytes=git(repo,'show',f"{base}:{protocol['source']['target_path']}").stdout
      if h(base_bytes)!=protocol['source']['baseline_target_sha256']:raise RuntimeError('pinned base target hash mismatch')
      import_check=subprocess.run([str(py),'-c','import mlx,mlx_lm,transformers,tokenizers'],capture_output=True,text=True)
      if import_check.returncode:raise RuntimeError('model runtime import preflight failed: '+import_check.stderr[-2000:])
      cli_check=subprocess.run([str(py),str(ROOT/'scripts/run_local_agent.py'),'--help'],capture_output=True,text=True)
      if cli_check.returncode or 'lease_bound_lines_v1' not in cli_check.stdout:raise RuntimeError('agent CLI tool-interface preflight failed')
      pf.mkdir(parents=True)
      write_new(pf/'preflight.json',{'protocol_sha256':args.protocol_sha256,'lock_sha256':args.lock_sha256,
        'runner_sha256':lock['model_runner_sha256'],'status':'NO_GENERATION_PREFLIGHT_PASS',
        'model_process_started':False,'candidate_workspace_created':False,'model_output_events':0,
        'resource_snapshot':state,'baseline_target_sha256':h(base_bytes),
        'model_runtime_import_returncode':import_check.returncode,'agent_cli_returncode':cli_check.returncode,
        'agent_cli_tool_interface_visible':True})
      manifest(pf)
      return 0
    output.mkdir(parents=True)
    status='EXECUTION_ABORTED';proc_record={};agent=None
    try:
      lim=protocol['resources'];before=snapshot(ROOT)
      if platform.system()!='Darwin' or platform.machine()!='arm64':raise RuntimeError('host is not pinned Darwin arm64')
      for key in ('available_memory_bytes','free_swap_bytes','disk_free_bytes'):
        if before[key]<lim['minimum_'+key]:raise RuntimeError('resource preflight failed: '+key)
      write_new(output/'preflight.json',before)
      add=git(repo,'worktree','add','--detach',str(workspace),base,text=True)
      base_target=workspace/protocol['source']['target_path']
      if hf(base_target)!=protocol['source']['baseline_target_sha256']:raise RuntimeError('candidate base target hash mismatch')
      (output/'workspace-created.txt').write_text(add.stdout,encoding='utf-8')
      instruction=protocol['task']['instruction']
      argv=["/usr/bin/sandbox-exec","-p",PROFILE,str(py),str(ROOT/'scripts/run_local_agent.py'),
        '--workspace',str(workspace),'--model-path',str(model),'--model-repository',protocol['model']['repository'],
        '--backend','mlx','--task-root',str(task_root),'--output',str(output/'agent.json'),
        '--event-journal',str(output/'events.jsonl'),'--instruction',instruction,
        '--seed',str(protocol['budget']['seed']),'--max-turns',str(protocol['budget']['max_turns']),
        '--max-no-tool-retries',str(protocol['budget']['max_no_tool_retries']),
        '--max-new-tokens-per-turn',str(protocol['budget']['max_new_tokens_per_turn']),
        '--max-total-new-tokens',str(protocol['budget']['max_total_new_tokens']),
        '--max-context-tokens',str(protocol['budget']['max_context_tokens']),
        '--wall-seconds',str(protocol['budget']['wall_seconds']),'--temperature',str(protocol['budget']['temperature']),
        '--top-p',str(protocol['budget']['top_p']),'--top-k',str(protocol['budget']['top_k']),
        '--no-do-sample','--edit-format','structured_replace','--tool-interface','lease_bound_lines_v1','--require-split','development']
      env=os.environ.copy();env.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONPATH':str(ROOT/'src')})
      env.pop('HF_TOKEN',None);env.pop('HUGGINGFACE_HUB_TOKEN',None)
      stdout=(output/'stdout.log').open('xb');stderr=(output/'stderr.log').open('xb')
      samples=(output/'resources.jsonl').open('x',encoding='utf-8')
      started=time.monotonic();peak=0;abort=None
      proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
      while proc.poll() is None:
        now=snapshot(ROOT);rss=process_rss(proc.pid);peak=max(peak,rss)
        samples.write(json.dumps({**now,'process_tree_rss_bytes':rss},sort_keys=True)+'\n');samples.flush()
        if now['available_memory_bytes']<lim['minimum_available_memory_bytes']:abort='available_memory_below_floor'
        elif now['free_swap_bytes']<lim['minimum_free_swap_bytes']:abort='free_swap_below_floor'
        elif now['swap_used_bytes']-before['swap_used_bytes']>lim['max_swap_growth_bytes']:abort='swap_growth_above_limit'
        elif now['disk_free_bytes']<lim['minimum_disk_free_bytes']:abort='disk_free_below_floor'
        elif rss>lim['max_model_process_tree_rss_bytes']:abort='model_process_rss_above_limit'
        elif time.monotonic()-started>protocol['budget']['wall_seconds']+protocol['budget']['startup_grace_seconds']:abort='wall_limit'
        if abort:
          try:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
          except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
          break
        time.sleep(lim['sample_interval_seconds'])
      code=proc.wait();stdout.close();stderr.close();samples.flush();os.fsync(samples.fileno());samples.close()
      proc_record={'argv':argv,'network_profile':PROFILE,'exit_code':code,'abort_reason':abort,
        'elapsed_seconds':round(time.monotonic()-started,3),'peak_process_tree_rss_bytes':peak,
        'stdout_sha256':hf(output/'stdout.log'),'stderr_sha256':hf(output/'stderr.log'),
        'resource_samples_sha256':hf(output/'resources.jsonl')}
      write_new(output/'process.json',proc_record)
      if (output/'agent.json').is_file():agent=json.loads((output/'agent.json').read_text())
      diff=git(workspace,'diff','--binary',base,'--',protocol['source']['target_path']).stdout
      with (output/'candidate-diff.patch').open('xb') as f:f.write(diff);f.flush();os.fsync(f.fileno())
      final_hash=hf(base_target)
      event_records=[]
      ep=output/'events.jsonl'
      if ep.exists():event_records=[json.loads(x) for x in ep.read_text().splitlines()]
      generated=sum(x.get('event',{}).get('generated_tokens',0) for x in event_records if x.get('event',{}).get('event_type')=='model_output_completed')
      edit_nonempty=bool(diff) and final_hash!=protocol['source']['baseline_target_sha256']
      if abort:status='RESOURCE_OR_TIME_ABORT'
      elif code!=0 or agent is None:status='MODEL_RUNNER_FAILED_BEFORE_FINAL_RECORD'
      elif not edit_nonempty:status='NO_SOURCE_EDIT'
      else:status='EDIT_APPLIED_AWAITING_INDEPENDENT_GRADING'
      result={'protocol_id':protocol['protocol_id'],'protocol_sha256':args.protocol_sha256,
       'lock_sha256':args.lock_sha256,'runner_sha256':lock['model_runner_sha256'],'task_id':protocol['task']['task_id'],
       'role':'DEVELOPMENT_ONLY','model':protocol['model']['repository'],'model_revision':protocol['model']['revision'],
       'workspace_revision':base,'baseline_target_sha256':protocol['source']['baseline_target_sha256'],
       'candidate_target_sha256':final_hash,'candidate_diff_sha256':hf(output/'candidate-diff.patch'),
       'candidate_diff_nonempty':bool(diff),'source_edit_applied':edit_nonempty,
       'agent_record_present':agent is not None,'agent_outcome':agent.get('agent_outcome') if agent else None,
       'stop_reason':agent.get('stop_reason') if agent else None,
       'generated_tokens_from_event_journal':generated,'model_output_events':sum(1 for x in event_records if x.get('event',{}).get('event_type')=='model_output_completed'),
       'process':proc_record,'status':status,'training_updates':0,'heldout_evaluations':0,
       'claim_limit':protocol['claim_limit']}
      write_new(output/'summary.json',result)
    except BaseException as exc:
      if not (output/'partial-failure.json').exists():write_new(output/'partial-failure.json',{
        'status':'EXECUTION_ABORTED','exception_type':type(exc).__name__,'message':str(exc),
        'time_utc':datetime.now(timezone.utc).isoformat()})
      raise
    finally:
      if not (output/'manifest.json').exists():manifest(output)
    return 0 if status in {'NO_SOURCE_EDIT','EDIT_APPLIED_AWAITING_INDEPENDENT_GRADING'} else 2
if __name__=='__main__':raise SystemExit(main())
