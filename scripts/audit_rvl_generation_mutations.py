#!/usr/bin/env python3
"""Compare retained RVL v3 and current v4 graders on inherited generation settings."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,platform,subprocess,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TASK_ROOT=ROOT/'benchmarks/historical/rvl_behavior_policy_parity'
V3_ROOT=ROOT/'results/rvl-hf-behavior-policy-parity-v1/cpu-calibration-v3/protocol_snapshot'
REL='src/rvl_systems/hf_backend.py'

def run(args,cwd=None): return subprocess.run(args,cwd=cwd,check=True,text=True,capture_output=True).stdout.strip()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load(path,name):
 sys.dont_write_bytecode=True; spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
def mutate(source):
 path=source/REL; text=path.read_text(encoding='utf-8'); needle='            top_p=1.0,\n'; replacement=needle+'            suppress_tokens=special.suppress_tokens,\n            no_repeat_ngram_size=special.no_repeat_ngram_size,\n'
 if text.count(needle)!=1: raise RuntimeError('expected unique fixed neutral top_p field')
 path.write_text(text.replace(needle,replacement),encoding='utf-8')
def checkout(task,path):
 path.mkdir(); run(['git','init','--quiet'],path); run(['git','remote','add','origin',task['source']['repository']],path)
 base=task['source']['base_revision']; fixed=task['source']['calibration_revision']
 run(['git','fetch','--quiet','--depth=1','--filter=blob:none','origin',base],path); run(['git','fetch','--quiet','--depth=1','--filter=blob:none','origin',fixed],path)
 run(['git','sparse-checkout','init','--no-cone'],path); (path/'.git/info/sparse-checkout').write_text(''.join('/'+f+'\n' for f in task['source']['files']))
 run(['git','checkout','--quiet','--detach',fixed],path)
 if run(['git','rev-parse','HEAD'],path)!=fixed: raise RuntimeError('fixed checkout revision mismatch')
def main():
 parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path); args=parser.parse_args()
 out=(args.output or ROOT/'results/rvl-hf-behavior-policy-parity-v1/generation-config-mutation-v4').expanduser().resolve()
 if out.exists(): parser.error(f'output already exists: {out}')
 if not V3_ROOT.is_dir(): parser.error(f'retained v3 snapshot missing: {V3_ROOT}')
 task=json.loads((TASK_ROOT/'task.json').read_text()); old=load(V3_ROOT/'evaluator/grade.py','vare_rvl_v3'); new=load(TASK_ROOT/'evaluator/grade.py','vare_rvl_v4')
 out.mkdir(parents=True); (out/'mutations').mkdir(); outcomes={}
 with tempfile.TemporaryDirectory(prefix='vare-rvl-generation-mutation-') as temp:
  source=Path(temp)/'candidate'; checkout(task,source)
  fixed_rev=task['source']['calibration_revision']; base=task['source']['base_revision']
  for label,do_mutate in [('control',False),('inherited_suppression',True)]:
   if do_mutate: mutate(source)
   old_result=old.grade(source,V3_ROOT); new_result=new.grade(source,TASK_ROOT)
   outcomes[label]={'v3':{'passed':old_result['passed'],'failures':old_result['failures']},'v4':{'passed':new_result['passed'],'failures':new_result['failures']}}
   write(out/f'{label}.v3.grade.json',old_result); write(out/f'{label}.v4.grade.json',new_result)
   if do_mutate:
    patch=run(['git','diff','--binary',base,'--',*task['source']['files']],source)
    (out/'mutations'/f'{label}.patch').write_text(patch,encoding='utf-8')
    run(['git','checkout','--quiet','--',*task['source']['files']],source)
 expected={'control':(True,True),'inherited_suppression':(True,False)}
 passed=all((outcomes[k]['v3']['passed'],outcomes[k]['v4']['passed'])==v for k,v in expected.items())
 summary={'schema_version':1,'study_id':'rvl-inherited-generation-settings-v4','source_repository':task['source']['repository'],'source_revision':fixed_rev,'base_revision':base,'protocols':{'comparison':'retained-v3','candidate':'current-v4'},'expected_outcomes':{k:{'v3_passed':v[0],'v4_passed':v[1]} for k,v in expected.items()},'observed_outcomes':outcomes,'acceptance_passed':passed,'mutation_description':'Propagate the pretrained model suppress_tokens and no_repeat_ngram_size into the fresh generation config. The v4 fixture starts with token 2 suppressed, which invalidates the first expected response token unless neutralized.','resources':{'model_weights_downloaded':False,'third_party_python_packages':0,'accelerator_hours':0,'paid_api_calls':0,'external_compute_usd':0},'claim_limit':'One mutation of two inherited decoding settings on one pinned backend revision; not comprehensive Transformers GenerationConfig coverage or model capability evidence.','runtime':{'created_at_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform()}}
 write(out/'summary.json',summary)
 files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
 write(out/'manifest.json',{'schema_version':1,'files':files,'v3_evaluator_sha256':sha(V3_ROOT/'evaluator/grade.py'),'v4_evaluator_sha256':sha(TASK_ROOT/'evaluator/grade.py'),'task_json_sha256':sha(TASK_ROOT/'task.json'),'protocol_lock_sha256':sha(TASK_ROOT/'protocol.lock.json'),'mutation_script_sha256':sha(Path(__file__).resolve())})
 print(json.dumps({'acceptance_passed':passed,'output':str(out),'outcomes':{k:{'v3':v['v3']['passed'],'v4':v['v4']['passed']} for k,v in outcomes.items()}},indent=2))
 return 0 if passed else 2
if __name__=='__main__': raise SystemExit(main())
