"""Audit retained preflight records, without inference or rerunning opened tasks."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('packet',type=Path)
    args=parser.parse_args();root=args.packet
    raw=json.loads((root/'run-1/raw.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    checks={}
    checks['frozen_protocol_hash']=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()==raw['protocol_sha256']
    checks['runner_hash']=hashlib.sha256((root/'preflight.py').read_bytes()).hexdigest()==raw['runner_sha256']
    checks['unchanged_rvl_head']=raw['rvl_commit']==protocol['rvl_commit']
    checks['fixed_two_member_receipt']=raw.get('donor_receipt',{}).get('member_ids')==['donor-0','donor-1']
    checks['no_learner_consumption']=raw.get('donor_receipt',{}).get('consumed_by_learner') is False
    checks['no_donor_selection']=raw.get('donor_receipt',{}).get('selected_by_reward_length_or_completion') is False
    checks['no_paid_calls']=raw['paid_spend_usd']==0
    checks['bounded_runtime']=raw['total_wall_s']<=protocol['max_runtime_s']+5
    episodes={}
    for path in sorted((root/'run-1').glob('*-journal.json')):
        data=json.loads(path.read_text())
        name=path.stem.removesuffix('-journal')
        rows=data['generations'];history=data['history']
        valid=0;tools={}
        for i,g in enumerate(rows):
            m=g['metadata'];tokens=m['response_token_ids'];logps=m['response_token_logprobs']
            checks[f'{name}/{i}/token_metadata']=len(tokens)==len(logps)==g['token_count'] and all(math.isfinite(x) and x<=0 for x in logps) and abs(sum(logps)-g['logprob'])<1e-8
            checks[f'{name}/{i}/state_context_hash']=m['state_snapshot_sha256']==hashlib.sha256(g['prompt'].encode()).hexdigest()
            checks[f'{name}/{i}/artifact_ids']=len(m['feedback_artifact_sha256'])==len(m['policy_artifact_sha256'])==64
            obs=history[i]['observation']
            if 'tool_error' not in obs:
                valid+=1
                tool=json.loads(g['response'])['tool'];tools[tool]=tools.get(tool,0)+1
        episodes[name]={'turns':len(rows),'valid_actions':valid,'tools':tools,'finished':data['done'],
                        'source_bytes':len(data['source'].encode()),'hidden_reward':raw['hidden_rewards'].get(name)}
    recon=[d for ds in raw['diagnostics'].values() for d in ds if 'sampled_logprob_reconstruction_max_abs_error' in d]
    costs={phase:sum(v for k,v in raw['episode_wall_s'].items() if k.startswith(phase+'-')) for phase in protocol['ordering']}
    if not any(k.startswith('repair-') for k in raw['episode_wall_s']):
        costs['repair']=None
    costs['recorded_partial_repair_operation_wall_s']=sum(e['elapsed_s'] for e in raw['events'] if e.get('phase')=='repair')
    costs['model_load_s']=raw.get('model_load_s')
    costs['total_measured_process_wall_s']=raw['total_wall_s']
    costs['note']='Synchronized Metal/host wall time, not pure GPU-seconds. Incomplete repair is a lower bound, not a speedup measurement.'
    result={'checks':checks,'audit_pass':all(checks.values()),'run_status':raw['status'],
            'run_error':raw.get('error'),'episodes':episodes,'reconstruction':recon,'costs':costs,
            'research_gate_passed':False,'reason':'This one-task fixed-weight diagnostic cannot establish >=15% CUDA collector savings or <=1pp task non-inferiority.',
            'scope':'Same-host record consistency, not independent reproduction or exact sampling-law proof.'}
    result['reconstruction_note']='mean_token_tv compares reconstructed T=1 and T=.8 laws; it is not the TV error of old-policy reconstruction. Full logit vectors were not retained.'
    print(json.dumps(result,indent=2))
    return 0 if result['audit_pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
