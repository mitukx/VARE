"""Read-only evidence reconstruction, intentionally independent of runner imports."""
import hashlib
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parent
base=json.loads((root/'baseline.json').read_text())
fixed=json.loads((root/'patched.json').read_text())
checks={}
for arm,data in [('base',base),('fixed',fixed)]:
    violations=0
    for row in data['cases']:
        events=row['events'];health=next(e for e in events if e['event']=='health');reward=next(e for e in events if e['event']=='reward');expected=row['expected']
        checks[f'{arm}_{row["case"]}_recorded_sha_check']=row['checks']['active_sha_matches_intended']==(health['active_artifact_sha256']==expected['sha256'])
        checks[f'{arm}_{row["case"]}_recorded_reward_check']=row['checks']['reward_matches_intended']==(reward['reward']==expected['reward'])
        violations+=sum(not v for v in row['checks'].values())
        if row['case']!='async_control':
            order={e['event']:e['index'] for e in events}
            checks[f'{arm}_{row["case"]}_ordering']=order['old_loader_entered'] < order['new_prepare_ack'] < order['old_loader_released']
    checks[f'{arm}_violation_count']=violations==data['violations']
checks['primary_baseline_fails']=base['failed_cases']==4 and base['violations']==21
checks['primary_fixed_passes']=fixed['failed_cases']==0 and fixed['violations']==0
for arm in ['baseline','patched']:
    row=json.loads((root/f'trainer-{arm}.json').read_text())
    r0,r1=row['rewards'];oracle=-(r0-r1)/(2*math.sqrt(((r0-r1)/2)**2+1e-6))
    checks[f'{arm}_actual_gradient_oracle']=abs(row['loss_gradient']-oracle)<=1e-6
    checks[f'{arm}_sgd']=abs(row['theta_after_sgd']+.1*row['loss_gradient'])<=1e-7
    checks[f'{arm}_zero_policy_mismatch']=row['metrics']['max_abs_log_ratio']<=1e-6
    checks[f'{arm}_worker_hash_matches_primary']=row['worker_source_sha256']==(base if arm=='baseline' else fixed)['worker_source_sha256']
    checks[f'{arm}_expected_direction']=row['loss_gradient']>0 if arm=='baseline' else row['loss_gradient']<0
    if arm=='patched': checks['patched_intended_gradient_matches']=row['gradient_error_to_intended']<=1e-6
checks['same_trainer_source']=json.loads((root/'trainer-baseline.json').read_text())['trainer_source_sha256']==json.loads((root/'trainer-patched.json').read_text())['trainer_source_sha256']
for name in ['protocol','trainer-protocol']:
    checks[f'{name}_hash']=hashlib.sha256((root/f'{name}.json').read_bytes()).hexdigest()==(root/f'{name}.sha256').read_text().split()[0]
checks['final_suite_22_passes']='Ran 22 tests' in (root/'final-tests.log').read_text() and (root/'final-tests.log').read_text().rstrip().endswith('OK')
result={'checks':checks,'passed':all(checks.values()),'failed':[k for k,v in checks.items() if not v],'note':'Same-author, same-host scripted audit. It is not external or human review and does not independently rerun TCP schedules.'}
(root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'checks':len(checks),'failed':result['failed']}))
raise SystemExit(not result['passed'])
