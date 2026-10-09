#!/usr/bin/env python3
"""Independent same-host audit of the frozen v3 feasibility bundle."""
from __future__ import annotations
import hashlib,json,math,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'protocols/rvl_cpu_real_model_update_path_v3.lock.json'
RUNNER=ROOT/'scripts/run_rvl_cpu_real_model_update_path_v3.py'
ADAPTER=ROOT/'src/vare/integrations/rvl_grpo.py'
RVL=Path('/Users/user/Documents/Codex/2026-10-08/https-github-com-mitukx-vare-https/work/rvl-cpu-update-smoke')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def reward(text,key):
 m=re.match(r'^\s*([A-D])(?=$|[\s).,:;])',text)
 return float(bool(m and m.group(1)==key))
def main():
 bundle=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/rvl-cpu-real-model-update-path-v3/run-1'
 p=json.loads(PROTOCOL.read_text()); s=json.loads((bundle/'summary.json').read_text()); checks={}
 def ck(k,v): checks[k]=bool(v); assert v,k
 ck('protocol_hash',s['protocol_sha256']==sha(PROTOCOL))
 ck('runner_hash',sha(RUNNER)==p['runner_sha256']==s['runner_sha256'])
 ck('adapter_hash',sha(ADAPTER)==p['vare_adapter_sha256']==s['vare_adapter_sha256'])
 ck('rvl_source_hashes',{x.relative_to(RVL).as_posix():sha(x) for x in sorted((RVL/'src/rvl_systems').rglob('*.py'))}==p['rvl_source_files_sha256'])
 model=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots'/p['model']['revision']
 ck('model_hashes',{n:sha(model/n) for n in p['model']['files_sha256']}==p['model']['files_sha256'])
 ck('terminal_status',s['status']=='passed' and json.loads((bundle/'progress.json').read_text())['status']=='passed')
 groups=s['responses']; tasks=p['task']['groups']; ck('groups_are_prefix',[g['prompt_id'] for g in groups]==[t['prompt_id'] for t in tasks[:len(groups)]])
 replay=[]
 for g,t in zip(groups,tasks):
  vals=[reward(row['text'],t['answer']) for row in g['responses']]
  ck(t['prompt_id']+'_reward',vals==g['rewards'] and g['question']==t['question'] and g['answer_key']==t['answer'])
  replay.append(vals)
  if len(set(vals))>1: break
 idx=next(i for i,x in enumerate(replay) if len(set(x))>1)
 ck('first_nonconstant_group',s['selected_prompt_id']==tasks[idx]['prompt_id'] and s['selected_rewards']==replay[idx] and all(len(set(x))==1 for x in replay[:idx]))
 vals=replay[idx]; mean=sum(vals)/4; var=sum((x-mean)**2 for x in vals)/4; eps=p['update']['advantage_epsilon']; clip=p['update']['advantage_clip']
 expected=[max(-clip,min(clip,(x-mean)/math.sqrt(var+eps))) for x in vals]
 ck('advantages',len(s['received_advantages'])==4 and all(abs(a-b)<1e-10 for a,b in zip(expected,s['received_advantages'],strict=True)))
 ck('one_update_finite_nonzero_gradient',s['optimizer_step_calls']==1 and s['finite_gradient_tensor_count']>0 and s['nonzero_gradient_tensor_count']>0)
 ck('candidate_changed',s['changed_parameter_tensor_count']>0 and s['max_abs_parameter_delta']>0)
 ck('incumbent_transaction_restore',s['incumbent_restored_exact'] and s['incumbent_rng_restored_exact'] and s['incumbent_training_mode_restored_exact'])
 ck('tokenizer_prompt_roundtrip',s['tokenizer_prompt_roundtrip_exact'])
 ck('candidate_weight_roundtrip',s['roundtrip_exact'] and bool(s['reloaded_parameter_fingerprint']))
 ck('cpu_and_caps',s['device']=='cpu' and s['mps_available_but_disabled'] and s['peak_rss_bytes']<=p['runtime']['max_peak_rss_bytes'] and s['wall_seconds']<=p['runtime']['max_wall_seconds'])
 result={'protocol_id':p['protocol_id'],'protocol_sha256':sha(PROTOCOL),'summary_sha256':sha(bundle/'summary.json'),'audit_type':'separate same-host author replay, not external reproduction','checks':checks,'checks_passed':len(checks),'checks_total':len(checks),'result':'passed','limits':['The optimizer is not independently rerun.','Temporary candidate checkpoint bytes are not retained; weight and tokenizer round-trip claims are checked from runner-recorded fingerprints and prompt IDs.','Execution feasibility only; no task-quality or capability claim.']}
 (bundle/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
