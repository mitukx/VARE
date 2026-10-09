#!/usr/bin/env python3
"""Independent post-run replay of the frozen v2 feasibility bundle."""
from __future__ import annotations
import hashlib, json, math, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'protocols/rvl_cpu_real_model_update_path_v2.lock.json'
RUNNER=ROOT/'scripts/run_rvl_cpu_real_model_update_path_v2.py'
ADAPTER=ROOT/'src/vare/integrations/rvl_grpo.py'
RVL=Path('/Users/user/Documents/Codex/2026-10-08/https-github-com-mitukx-vare-https/work/rvl-cpu-update-smoke')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def reward(text, answer):
    m=re.match(r'^\s*([A-D])(?=$|[\s).,:;])',text)
    return float(bool(m and m.group(1)==answer))
def main():
    bundle=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/rvl-cpu-real-model-update-path-v2/run-1'
    protocol=json.loads(PROTOCOL.read_text()); summary=json.loads((bundle/'summary.json').read_text()); progress=json.loads((bundle/'progress.json').read_text())
    checks={}
    def check(name,ok):
        checks[name]=bool(ok)
        if not ok: raise AssertionError('audit failed: '+name)
    check('protocol_hash',summary['protocol_sha256']==sha(PROTOCOL))
    check('runner_hash',sha(RUNNER)==protocol['runner_sha256']==summary['runner_sha256'])
    check('adapter_hash',sha(ADAPTER)==protocol['vare_adapter_sha256']==summary['vare_adapter_sha256'])
    check('pinned_rvl_sources',{p.relative_to(RVL).as_posix():sha(p) for p in sorted((RVL/'src/rvl_systems').rglob('*.py'))}==protocol['rvl_source_files_sha256'])
    model=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots'/protocol['model']['revision']
    check('cached_model_files',{n:sha(model/n) for n in protocol['model']['files_sha256']}==protocol['model']['files_sha256'])
    check('terminal_pass',summary['status']=='passed' and progress['status']=='passed')
    groups=summary['responses']; tasks=protocol['task']['groups']
    check('groups_are_protocol_prefix',[g['prompt_id'] for g in groups]==[t['prompt_id'] for t in tasks[:len(groups)]])
    recomputed=[]
    for g,t in zip(groups,tasks):
        check(t['prompt_id']+'_four_samples',len(g['responses'])==4)
        check(t['prompt_id']+'_question_answer',g['question']==t['question'] and g['answer_key']==t['answer'])
        scores=[reward(row['text'],t['answer']) for row in g['responses']]
        check(t['prompt_id']+'_reward_replay',scores==g['rewards'])
        recomputed.append(scores)
        if len(set(scores))>1: break
    selected=next(i for i,s in enumerate(recomputed) if len(set(s))>1)
    check('first_nonconstant_group',summary['selected_prompt_id']==tasks[selected]['prompt_id'] and summary['selected_rewards']==recomputed[selected])
    scores=recomputed[selected]; mean=sum(scores)/4; variance=sum((x-mean)**2 for x in scores)/4
    eps=protocol['update']['advantage_epsilon']; clip=protocol['update']['advantage_clip']
    expected=[max(-clip,min(clip,(x-mean)/math.sqrt(variance+eps))) for x in scores]
    check('group_advantages',all(abs(x-y)<1e-10 for x,y in zip(expected,summary['received_advantages'],strict=True)))
    check('single_finite_nonzero_gradient_update',summary['optimizer_step_calls']==1 and summary['finite_gradient_tensor_count']>0 and summary['nonzero_gradient_tensor_count']>0)
    check('changed_candidate',summary['changed_parameter_tensor_count']>0 and summary['max_abs_parameter_delta']>0)
    check('incumbent_restored',summary['incumbent_restored_exact'] and summary['incumbent_rng_restored_exact'] and summary['incumbent_training_mode_restored_exact'])
    check('tokenizer_prompt_roundtrip',summary['tokenizer_prompt_roundtrip_exact'])
    check('model_parameter_roundtrip',summary['roundtrip_exact'] and bool(summary['reloaded_parameter_fingerprint']))
    check('cpu_and_resource_caps',summary['device']=='cpu' and summary['mps_available_but_disabled'] and summary['peak_rss_bytes']<=protocol['runtime']['max_peak_rss_bytes'] and summary['wall_seconds']<=protocol['runtime']['max_wall_seconds'])
    result={'protocol_id':protocol['protocol_id'],'protocol_sha256':sha(PROTOCOL),'summary_sha256':sha(bundle/'summary.json'),'audit_type':'separate same-host author replay; not outside reproduction','checks':checks,'checks_passed':len(checks),'checks_total':len(checks),'result':'passed','limits':['The optimizer is not independently rerun.','The temporary candidate checkpoint is not retained; exact tensor and tokenizer round trips are checked from runner-recorded digests/results.','This is execution feasibility only, not task efficacy or model capability.']}
    (bundle/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0
if __name__=='__main__': raise SystemExit(main())
