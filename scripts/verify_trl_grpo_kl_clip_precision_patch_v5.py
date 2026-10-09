#!/usr/bin/env python3
"""Independent standard-library audit for frozen TRL KL precision patch v5 outputs."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
HEAD='0aaea03f2fa449bc7a91f1973e7940da11da65da'
PATCH='0f6c60de679c3c8b467b2bc8b77b01e06f890486cd567322940e5ce8a2137468'
FILES={'base_f16':'pr-head-f16-token.json','candidate_f16':'candidate-f16-token.json','base_f32':'pr-head-f32-token.json','candidate_f32':'candidate-f32-token.json'}
REL=.005; ABS=.1

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b): return isinstance(a,(int,float)) and math.isfinite(float(a)) and math.isclose(float(a),b,rel_tol=REL,abs_tol=ABS)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); root=a.run_dir; errors=[]
 protocol=json.loads((root/'protocol.snapshot.json').read_text()); f=protocol['fixture']; art=protocol['artifacts']
 protocol_hash=sha(root/'protocol.snapshot.json'); patch_hash=sha(root/'candidate.patch')
 if protocol['protocol_id']!='trl_grpo_kl_clip_precision_patch_v5': errors.append('protocol id mismatch')
 if patch_hash!=PATCH or protocol['source']['patch_sha256']!=PATCH: errors.append('candidate patch digest mismatch')
 checker_hash=sha(Path(__file__))
 if checker_hash!=art['checker_sha256']: errors.append('checker hash differs from lock')
 runner_hash=sha(root/'runner.py')
 if runner_hash!=art['runner_sha256']: errors.append('retained runner hash differs from lock')
 rows={k:json.loads((root/name).read_text()) for k,name in FILES.items()}
 for k,r in rows.items():
  if r.get('protocol_id')!=protocol['protocol_id']: errors.append(f'{k}: protocol identity mismatch')
  if r.get('protocol_sha256')!=protocol_hash: errors.append(f'{k}: protocol digest mismatch')
  if r.get('source',{}).get('revision')!=HEAD: errors.append(f'{k}: source revision mismatch')
  if r.get('status')!='complete': errors.append(f'{k}: method execution incomplete')
  if r.get('importance_sampling_level')!=f['importance_sampling_level']: errors.append(f'{k}: IS mode mismatch')
  expected_dtype='float16' if k.endswith('f16') else 'float32'
  if r.get('dtype')!=expected_dtype: errors.append(f'{k}: dtype mismatch')
  should_patch=k.startswith('candidate')
  if r.get('source',{}).get('patched') is not should_patch: errors.append(f'{k}: patched-source marker mismatch')
  expected_trainer_sha=protocol['source']['candidate_trainer_sha256'] if should_patch else protocol['source']['base_trainer_sha256']
  if r.get('source',{}).get('trainer_sha256')!=expected_trainer_sha: errors.append(f'{k}: trainer file digest mismatch')
  if k.startswith('candidate') and r.get('source',{}).get('patch_sha256')!=PATCH: errors.append(f'{k}: source patch mismatch')
  fixture_digest=hashlib.sha256(json.dumps(f,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  if r.get('fixture_sha256')!=fixture_digest: errors.append(f'{k}: fixture digest mismatch')
 beta=f['beta']; B=len(f['completion_mask']); c=f['clip']; mask=f['completion_mask']; xs=f['kl_log_ratios']; z=f['policy_logps']; old=f['old_logps']
 T=f['sequence_length']
 if f['batch_size']!=B or not (len(mask)==len(xs)==len(z)==len(old)==B): errors.append('fixture batch dimensions differ')
 for name,matrix in [('completion_mask',mask),('kl_log_ratios',xs),('policy_logps',z),('old_logps',old)]:
  if any(len(row)!=T for row in matrix): errors.append(f'fixture {name} sequence dimensions differ')
 if f['use_bias_correction_kl'] is not True: errors.append('fixture does not use the frozen bias-correction estimand')
 active_count=0; expected_loss=0.; expected_metric_num=0.; expected_grad=[]
 for i,rowmask in enumerate(mask):
  n=sum(rowmask)
  if n<=0: errors.append(f'fixture row {i} has no active tokens'); expected_grad.append([0.]*len(rowmask)); continue
  row_grad=[]; row_obj=0.
  for j,m in enumerate(rowmask):
   if m:
    ratio=math.exp(z[i][j]-old[i][j]); clipped=min(xs[i][j],c); kval=math.exp(clipped)-clipped-1
    row_obj += kval*ratio/n; expected_metric_num += kval*ratio; active_count += 1
    row_grad.append(-beta*ratio*clipped/(B*n))
   else: row_grad.append(0.)
  expected_loss += beta*row_obj/B; expected_grad.append(row_grad)
 expected_metric=expected_metric_num/active_count
 base=rows['base_f16']; cand=rows['candidate_f16']
 base_overflow=base.get('kl_metric') in ('Infinity','-Infinity','NaN')
 if not base_overflow: errors.append('unmodified fp16 baseline did not overflow its aggregate metric')
 base_gradient_bad=False
 for i,row in enumerate(expected_grad):
  for j,target in enumerate(row):
   if mask[i][j] and not close(base['gradient'][i][j],target): base_gradient_bad=True
 if not base_gradient_bad: errors.append('unmodified fp16 baseline gradient matches analytic target unexpectedly')
 cand_finite=(bool(cand['loss_finite']) and math.isfinite(cand['loss']) and bool(cand['gradient_finite']) and isinstance(cand['kl_metric'],(int,float)) and math.isfinite(cand['kl_metric']))
 if not cand_finite: errors.append('candidate fp16 output has a non-finite loss, metric, or gradient')
 if not close(cand['loss'],expected_loss): errors.append('candidate fp16 loss outside analytic tolerance')
 if not close(cand['kl_metric'],expected_metric): errors.append('candidate fp16 KL metric outside analytic tolerance')
 for i,row in enumerate(expected_grad):
  for j,target in enumerate(row):
   if not close(cand['gradient'][i][j],target): errors.append(f'candidate fp16 gradient[{i},{j}] outside analytic tolerance')
   if not mask[i][j] and cand['gradient'][i][j]!=0: errors.append(f'candidate padding gradient[{i},{j}] is nonzero')
 b32=rows['base_f32']; c32=rows['candidate_f32']; f32=(b32['loss']==c32['loss'] and b32['gradient']==c32['gradient'] and b32['kl_metric']==c32['kl_metric'])
 if not f32: errors.append('fp32 control is not bit-identical')
 out={'protocol_id':protocol['protocol_id'],'protocol_sha256':protocol_hash,'status':'pass' if not errors else 'fail','errors':errors,'baseline_fp16_metric_overflow':base_overflow,'baseline_fp16_gradient_mismatch':base_gradient_bad,'candidate_fp16_finite':cand_finite,'analytic_expected':{'loss':expected_loss,'kl_metric':expected_metric,'gradient':expected_grad},'candidate_fp16_observed':{'loss':cand['loss'],'kl_metric':cand['kl_metric'],'gradient':cand['gradient']},'baseline_fp16_observed':{'loss':base['loss'],'kl_metric':base['kl_metric'],'gradient':base['gradient']},'fp32_bit_identical':f32,'source_imports':'none; standard-library math and retained JSON only','tolerance':{'relative':REL,'absolute':ABS}}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n'); print(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)); return 0 if out['status']=='pass' else 1
if __name__=='__main__': raise SystemExit(main())
