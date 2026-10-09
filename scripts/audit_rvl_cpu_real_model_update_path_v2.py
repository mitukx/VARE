#!/usr/bin/env python3
"""Independently replay retained v2 rewards and record its pre-reload failure."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'protocols/rvl_cpu_real_model_update_path_v2.lock.json'
RUNNER=ROOT/'scripts/run_rvl_cpu_real_model_update_path_v2.py'
ADAPTER=ROOT/'src/vare/integrations/rvl_grpo.py'
RVL=Path('/Users/user/Documents/Codex/2026-10-08/https-github-com-mitukx-vare-https/work/rvl-cpu-update-smoke')
BUNDLE=ROOT/'results/rvl-cpu-real-model-update-path-v2/run-1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def reward(text,key):
 m=re.match(r'^\s*([A-D])(?=$|[\s).,:;])',text)
 return float(bool(m and m.group(1)==key))
def main():
 p=json.loads(PROTOCOL.read_text()); s=json.loads((BUNDLE/'summary.json').read_text()); checks={}
 def check(k,v): checks[k]=bool(v); assert v,k
 check('protocol_runner_adapter_hashes',s['protocol_sha256']==sha(PROTOCOL) and s['runner_sha256']==sha(RUNNER)==p['runner_sha256'] and s['vare_adapter_sha256']==sha(ADAPTER)==p['vare_adapter_sha256'])
 check('pinned_rvl_sources',{x.relative_to(RVL).as_posix():sha(x) for x in sorted((RVL/'src/rvl_systems').rglob('*.py'))}==p['rvl_source_files_sha256'])
 check('terminal_failure_retained',s['status']=='failed' and (BUNDLE/'progress.json').is_file())
 check('failure_is_tokenizer_variable_lifetime',s['error_type']=='NameError' and 'free variable \'tokenizer\'' in s['error'])
 check('one_update_and_incumbent_restore',s['optimizer_step_calls']==1 and s['incumbent_restored_exact'] and s['incumbent_rng_restored_exact'] and s['incumbent_training_mode_restored_exact'])
 check('declared_resource_caps_passed',s['wall_seconds']<=p['runtime']['max_wall_seconds'] and s['peak_rss_bytes']<=p['runtime']['max_peak_rss_bytes'])
 replay=[]
 for g,t in zip(s['responses'],p['task']['groups']):
  scores=[reward(x['text'],t['answer']) for x in g['responses']]
  replay.append({'prompt_id':g['prompt_id'],'scores':scores,'matches':scores==g['rewards']})
  if len(set(scores))>1: break
 check('reward_labels_replay',all(x['matches'] for x in replay))
 check('first_nonconstant_group_matches',s['selected_prompt_id']==replay[-1]['prompt_id'] and len(set(replay[-1]['scores']))>1 and all(len(set(x['scores']))==1 for x in replay[:-1]))
 result={'protocol_id':p['protocol_id'],'protocol_sha256':sha(PROTOCOL),'summary_sha256':sha(BUNDLE/'summary.json'),'checks':checks,'groups':replay,'audit_result':'passed for faithful failure reconstruction','experiment_result':'failed before tokenizer or model reload; do not count update-path gate as passed','limits':['Same-host author audit; not external reproduction.','Candidate checkpoint was written to a temporary directory, then discarded when the tokenizer variable-lifetime error interrupted reload.','No task efficacy or capability was measured.']}
 (BUNDLE/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
