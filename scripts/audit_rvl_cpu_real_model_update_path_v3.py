#!/usr/bin/env python3
"""Audit v3 measurement gates separately from its post-gate runner error."""
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
 b=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/rvl-cpu-real-model-update-path-v3/run-1'
 p=json.loads(PROTOCOL.read_text()); s=json.loads((b/'summary.json').read_text()); checks={}
 def ck(k,v): checks[k]=bool(v); assert v,k
 ck('protocol_hash',s['protocol_sha256']==sha(PROTOCOL))
 ck('runner_hash',sha(RUNNER)==p['runner_sha256']==s['runner_sha256'])
 ck('adapter_hash',sha(ADAPTER)==p['vare_adapter_sha256']==s['vare_adapter_sha256'])
 ck('rvl_source_hashes',{x.relative_to(RVL).as_posix():sha(x) for x in sorted((RVL/'src/rvl_systems').rglob('*.py'))}==p['rvl_source_files_sha256'])
 m=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots'/p['model']['revision']
 ck('model_hashes',{n:sha(m/n) for n in p['model']['files_sha256']}==p['model']['files_sha256'])
 ck('retained_post_gate_cleanup_failure',s['status']=='failed' and s['error_type']=='UnboundLocalError' and 'loaded_tokenizer' in s['error'])
 groups=s['responses']; tasks=p['task']['groups']; ck('groups_are_protocol_prefix',[g['prompt_id'] for g in groups]==[t['prompt_id'] for t in tasks[:len(groups)]])
 replay=[]
 for g,t in zip(groups,tasks):
  vals=[reward(row['text'],t['answer']) for row in g['responses']]
  ck(t['prompt_id']+'_reward_replay',vals==g['rewards'] and g['question']==t['question'] and g['answer_key']==t['answer'])
  replay.append(vals)
  if len(set(vals))>1: break
 i=next(i for i,v in enumerate(replay) if len(set(v))>1)
 ck('first_mixed_group_selected',s['selected_prompt_id']==tasks[i]['prompt_id'] and s['selected_rewards']==replay[i] and all(len(set(x))==1 for x in replay[:i]))
 vals=replay[i]; avg=sum(vals)/4; var=sum((x-avg)**2 for x in vals)/4; eps=p['update']['advantage_epsilon']; clip=p['update']['advantage_clip']; adv=[max(-clip,min(clip,(x-avg)/math.sqrt(var+eps))) for x in vals]
 ck('advantages_recomputed',all(abs(x-y)<1e-10 for x,y in zip(adv,s['received_advantages'],strict=True)))
 ck('one_optimizer_step_and_finite_nonzero_gradients',s['optimizer_step_calls']==1 and s['finite_gradient_tensor_count']>0 and s['nonzero_gradient_tensor_count']>0)
 ck('candidate_weights_changed',s['changed_parameter_tensor_count']>0 and s['max_abs_parameter_delta']>0)
 ck('incumbent_model_optimizer_rng_mode_restored',s['incumbent_restored_exact'] and s['incumbent_rng_restored_exact'] and s['incumbent_training_mode_restored_exact'])
 ck('tokenizer_ids_match_on_frozen_prompts',s['tokenizer_prompt_roundtrip_exact'])
 ck('saved_model_parameter_fingerprint_matches',s['roundtrip_exact'] and bool(s['reloaded_parameter_fingerprint']))
 ck('cpu_resource_gates',s['device']=='cpu' and s['mps_available_but_disabled'] and s['wall_seconds']<=p['runtime']['max_wall_seconds'] and s['peak_rss_bytes']<=p['runtime']['max_peak_rss_bytes'])
 result={'protocol_id':p['protocol_id'],'protocol_sha256':sha(PROTOCOL),'summary_sha256':sha(b/'summary.json'),'checks':checks,'checks_passed':len(checks),'checks_total':len(checks),'measurement_gate_evidence':'all declared update, restore, tokenizer/model round-trip, and resource fields are present and satisfy their frozen thresholds','runner_terminal_status':'failed after these fields were recorded, during redundant cleanup of an already deleted tokenizer variable','audit_type':'separate same-host author replay, not external reproduction','limits':['This audit verifies retained claims and hashes; it does not rerun the optimizer or independently access the deleted temporary checkpoint.','This is execution feasibility only; no task-success or model-capability effect is measured.']}
 (b/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
