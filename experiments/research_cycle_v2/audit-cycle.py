"""Same-host record consistency audit; does not constitute outside reproduction."""
from pathlib import Path
import hashlib
import json
import sys
root=Path(sys.argv[1]).resolve()
review=root/'verifier-same-epoch-review-v2'
research=root/'interactive-rollout-reuse-v1'
if not review.exists(): review=root/'verifier_same_epoch_review_v2'
if not research.exists(): research=root/'interactive_rollout_reuse_v1'
checks={}
def check(name,value): checks[name]=bool(value)
for packet in [review,research]:
 for sha in packet.glob('*.sha256'):
  recorded=sha.read_text().split()[0]
  payload=packet/Path(sha.read_text().split()[1]).name
  check(f'{packet.name}/{sha.name}',hashlib.sha256(payload.read_bytes()).hexdigest()==recorded)
b=json.loads((review/'baseline.json').read_text());p=json.loads((review/'patched.json').read_text())
check('base_wrong_success',b[0]['wrong_artifact_success'] and b[0]['activation_success'])
check('base_reward_mismatch',b[0]['reward']==.1)
check('base_identical_control',b[1]['fixed_contract_pass'])
check('fixed_conflict_rejects',not p[0]['activation_success'] and not p[0]['wrong_artifact_success'])
check('fixed_recovery',all(x['fixed_contract_pass'] and x['recovery_error'] is None for x in p))
ci=json.loads((review/'github-ci.json').read_text())
check('ci_exact_head',ci['headRefOid']=='ba94a9e513fdaf882cde091766007a88b9edaeee')
check('ci_all_success',len(ci['statusCheckRollup'])==4 and all(x['conclusion']=='SUCCESS' for x in ci['statusCheckRollup']))
check('ready_unmerged',ci['state']=='OPEN' and not ci['isDraft'])
r=json.loads((research/'raw.json').read_text());t=json.loads((research/'trainer-raw.json').read_text())
check('primary_checks',r['pass'] and all(r['checks'].values()))
check('exact_naive_tv',r['exogenous']['naive_tv']['fraction']=='2/25')
check('exact_consumed_bias',all(x['consumed_tv']['fraction']=='1/10' and x['consumed_group_update']['fraction']=='1/20' and x['fresh_group_update']['fraction']=='9/100' for x in r['adaptive']))
check('donor_control',all(x['independent_tv']['fraction']=='0' for x in r['adaptive']))
check('feedback_control',r['feedback']['repair_tv']['fraction']=='0' and r['feedback']['relabel_tv']['fraction']=='1/2')
check('retained_failed_attempt',not json.loads((research/'attempt-1-raw.json').read_text())['pass'])
check('trainer_checks',t['pass'] and all(t['checks'].values()))
for i,s in enumerate(t['summaries']):
 check(f'trainer_oracle_{i}',abs(s['consumed']-s['consumed_oracle'])<1e-6 and abs(s['fresh']-s['fresh_oracle'])<1e-6)
result={'checks':checks,'pass':all(checks.values()),'scope':'Same-author record/hash and oracle consistency, not external reproduction.'}
print(json.dumps(result,indent=2))
assert all(checks.values()),checks
