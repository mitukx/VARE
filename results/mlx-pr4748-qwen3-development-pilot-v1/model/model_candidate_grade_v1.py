#!/usr/bin/env python3
"""Build and independently grade a one-attempt #4748 Development candidate."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
import psutil
PROFILE="(version 1) (allow default) (deny network*)"
ROOT=Path(__file__).resolve().parents[2]
def h(b):return hashlib.sha256(b).hexdigest()
def hf(p):return h(p.read_bytes())
def write_new(p,obj):
 b=(json.dumps(obj,sort_keys=True,indent=2)+'\n').encode()
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
def snap(path):
 v=psutil.virtual_memory();s=psutil.swap_memory()
 return {'timestamp_utc':datetime.now(timezone.utc).isoformat(),'available_memory_bytes':v.available,
  'free_swap_bytes':s.free,'swap_used_bytes':s.used,'disk_free_bytes':shutil.disk_usage(path).free}
def proc_rss(pid):
 try:
  p=psutil.Process(pid);return sum(x.memory_info().rss for x in [p,*p.children(recursive=True)] if x.is_running())
 except (psutil.NoSuchProcess,psutil.AccessDenied):return 0
def run(cmd,cwd,env,out,label,limits,before):
 lp=out/(label+'.stdout.log');sp=out/(label+'.resources.jsonl');start=time.monotonic();peak=0;abort=None
 with lp.open('xb') as log,sp.open('x',encoding='utf-8') as samples:
  p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while p.poll() is None:
   q=snap(cwd);rss=proc_rss(p.pid);peak=max(peak,rss)
   samples.write(json.dumps({**q,'process_tree_rss_bytes':rss},sort_keys=True)+'\n');samples.flush()
   if q['available_memory_bytes']<limits['minimum_available_memory_bytes']:abort='available_memory_below_floor'
   elif q['free_swap_bytes']<limits['minimum_free_swap_bytes']:abort='free_swap_below_floor'
   elif q['swap_used_bytes']-before['swap_used_bytes']>limits['max_swap_growth_bytes']:abort='swap_growth_above_limit'
   elif q['disk_free_bytes']<limits['minimum_disk_free_bytes']:abort='disk_free_below_floor'
   elif rss>limits['max_model_process_tree_rss_bytes']:abort='process_tree_rss_above_limit'
   elif time.monotonic()-start>limits['max_phase_seconds']:abort='phase_wall_limit'
   if abort:
    import signal
    try:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
    break
   time.sleep(limits['sample_interval_seconds'])
  code=p.wait();log.flush();os.fsync(log.fileno());samples.flush();os.fsync(samples.fileno())
 return {'argv':cmd,'network_profile':PROFILE,'exit_code':code,'abort_reason':abort,
  'elapsed_seconds':round(time.monotonic()-start,3),'peak_process_tree_rss_bytes':peak,
  'stdout_sha256':hf(lp),'samples_sha256':hf(sp)}
def manifest(root):
 rows=[]
 for p in sorted(x for x in root.rglob('*') if x.is_file() and x.name!='manifest.json'):
  rows.append({'path':p.relative_to(root).as_posix(),'size_bytes':p.stat().st_size,'sha256':hf(p)})
 write_new(root/'manifest.json',{'schema_version':1,'algorithm':'sha256','files':rows})
def git(repo,*args,binary=False):
 r=subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True,text=not binary);return r.stdout
def tree(root):
 d=hashlib.sha256()
 for p in sorted(x for x in root.rglob('*') if x.is_file() and '.git' not in x.relative_to(root).parts):
  rel=p.relative_to(root).as_posix().encode();d.update(len(rel).to_bytes(4,'big'));d.update(rel);d.update(bytes.fromhex(hf(p)))
 return d.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--protocol-sha256',required=True)
 ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--lock-sha256',required=True);ap.add_argument('--model-run',type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 pth=a.protocol.resolve();lockp=a.lock.resolve();protocol=json.loads(pth.read_text());lock=json.loads(lockp.read_text())
 if hf(pth)!=a.protocol_sha256 or lock['protocol_sha256']!=a.protocol_sha256 or hf(lockp)!=a.lock_sha256:raise RuntimeError('protocol/lock hash mismatch')
 if hf(Path(__file__).resolve())!=lock['grader_runner_sha256']:raise RuntimeError('grader runner source hash mismatch')
 runroot=a.model_run.resolve();model_summary=json.loads((runroot/'summary.json').read_text())
 out=a.output.resolve();
 if out.exists():raise RuntimeError('refusing to overwrite grade output')
 out.mkdir(parents=True)
 try:
  task=protocol['task']; source=protocol['source']; lim=protocol['grade_resources']; workspace=(ROOT/protocol['execution']['candidate_workspace']).resolve()
  diff=git(workspace,'diff','--binary',source['base_commit'],'--',source['target_path'],binary=True)
  candidate=workspace/source['target_path']; candidate_hash=hf(candidate)
  if h(diff)!=model_summary['candidate_diff_sha256'] or candidate_hash!=model_summary['candidate_target_sha256']:
   raise RuntimeError('model candidate changed after its durable run record')
  if not diff or model_summary['source_edit_applied'] is not True:
   result={'protocol_sha256':a.protocol_sha256,'model_run_summary_sha256':hf(runroot/'summary.json'),
    'candidate_target_sha256':candidate_hash,'candidate_diff_sha256':h(diff),'status':'NO_EDIT_NO_GRADE',
    'independently_verified_success':False,'reason':'No nonempty source diff was produced; no grader call is justified.'}
   write_new(out/'grade.json',result);return 0
  changed=git(workspace,'diff','--name-only',source['base_commit']).splitlines()
  if changed!=[source['target_path']]:raise RuntimeError('candidate modified files outside the frozen allowlist')
  dependencies=(ROOT/protocol['execution']['dependencies']).resolve();py=Path(protocol['execution']['build_python']).expanduser().absolute()
  for name,spec in protocol['execution']['dependency_sources'].items():
   p=dependencies/spec['path']
   if tree(p)!=spec['tree_sha256']:raise RuntimeError('dependency hash mismatch: '+name)
  before=snap(ROOT)
  for key in ('available_memory_bytes','free_swap_bytes','disk_free_bytes'):
   if before[key]<lim['minimum_'+key]:raise RuntimeError('grade preflight failed: '+key)
  write_new(out/'preflight.json',before)
  buildenv=os.environ.copy();buildenv.update({'PYPI_RELEASE':'1','CMAKE_BUILD_PARALLEL_LEVEL':str(lim['build_parallelism']),
   'PATH':f"{py.parent}:{os.environ.get('PATH','')}",'PYTHONPATH':f"{dependencies/'nanobind'/'src'}:{workspace/'python'}"})
  cmake=['-DMLX_BUILD_METAL=OFF','-DFETCHCONTENT_FULLY_DISCONNECTED=ON']
  for _,spec in sorted(protocol['execution']['dependency_sources'].items()):cmake.append(f"-DFETCHCONTENT_SOURCE_DIR_{spec['fetchcontent_name'].upper()}={dependencies/spec['path']}")
  buildenv['CMAKE_ARGS']=' '.join(cmake)
  br=run(['/usr/bin/sandbox-exec','-p',PROFILE,str(py),'setup.py','build_ext','--inplace'],workspace,buildenv,out,'build',lim,before)
  if br['exit_code']!=0 or br['abort_reason']:
   result={'protocol_sha256':a.protocol_sha256,'candidate_target_sha256':candidate_hash,'candidate_diff_sha256':h(diff),
    'build':br,'independently_verified_success':False,'status':'CANDIDATE_BUILD_FAILED'}
   write_new(out/'grade.json',result);return 0
  grader=(ROOT/protocol['execution']['grader_path']).resolve()
  for rel,expected in protocol['execution']['grader_file_hashes'].items():
   if hf(grader/rel)!=expected:raise RuntimeError('grader input hash mismatch: '+rel)
  runtime=os.environ.copy();runtime.update({'PYTHONPATH':f"{grader}:{workspace/'python'}",'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','DEVICE':'cpu','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
  check=run(['/usr/bin/sandbox-exec','-p',PROFILE,str(py),'-c',
    'import json, mlx.core as mx; print(json.dumps({"module":mx.__file__,"device":str(mx.default_device())}))'],workspace,runtime,out,'import-check',lim,snap(ROOT))
  import_row=json.loads((out/'import-check.stdout.log').read_text().splitlines()[-1]) if check['exit_code']==0 else {}
  if not check['exit_code']==0 or not str(import_row.get('module','')).startswith(str(workspace/'python')):raise RuntimeError('candidate grading imported MLX outside exact workspace')
  oracle=run(['/usr/bin/sandbox-exec','-p',PROFILE,str(py),str(grader/'independent_oracle.py')],workspace,runtime,out,'independent-oracle',lim,snap(ROOT))
  oracle_obj=json.loads((out/'independent-oracle.stdout.log').read_text().splitlines()[-1]) if oracle['exit_code'] in (0,1) else {}
  test=run(['/usr/bin/sandbox-exec','-p',PROFILE,str(py),'-m','unittest','-v',task['upstream_test_selector']],grader,runtime,out,'upstream-test',lim,snap(ROOT))
  all_oracle=bool(oracle_obj.get('passed')) and len(oracle_obj.get('cases',{}))==6
  success=all_oracle and oracle['exit_code']==0 and test['exit_code']==0 and all(x['abort_reason'] is None for x in (oracle,test))
  result={'protocol_sha256':a.protocol_sha256,'model_run_summary_sha256':hf(runroot/'summary.json'),
   'candidate_target_sha256':candidate_hash,'candidate_diff_sha256':h(diff),'candidate_changed_paths':changed,
   'candidate_build':br,'import_provenance':import_row,'import_check':check,
   'independent_oracle':oracle,'oracle_result':oracle_obj,'upstream_test':test,
   'independently_verified_success':success,'status':'INDEPENDENT_TASK_SUCCESS' if success else 'EDIT_NOT_VERIFIED'}
  write_new(out/'grade.json',result);return 0
 except BaseException as e:
  if not (out/'failure.json').exists():write_new(out/'failure.json',{'status':'GRADE_EXECUTION_FAILED','exception_type':type(e).__name__,'message':str(e),'time_utc':datetime.now(timezone.utc).isoformat()})
  raise
 finally:
  if not (out/'manifest.json').exists():manifest(out)
if __name__=='__main__':raise SystemExit(main())
