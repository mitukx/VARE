#!/usr/bin/env python3
"""Independently check the frozen multi-token token-level KL precision experiment."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
HEAD='0aaea03f2fa449bc7a91f1973e7940da11da65da'
PATCH='0f6c60de679c3c8b467b2bc8b77b01e06f890486cd567322940e5ce8a2137468'
FILES={'base_f16':'pr-head-f16-token.json','candidate_f16':'candidate-f16-token.json','base_f32':'pr-head-f32-token.json','candidate_f32':'candidate-f32-token.json'}
BETA=.1; B=2; C=10.; R=math.exp(.25); LENGTHS=(2,3); REL=.005; ABS=.1
K=math.exp(C)-C-1
LOSS=BETA*R*K
EXPECTED_GRAD=[[-BETA*R*C/B/n]*n+[0.]*(3-n) for n in LENGTHS]

def close(a,b): return isinstance(a,(int,float)) and math.isfinite(float(a)) and math.isclose(float(a),b,rel_tol=REL,abs_tol=ABS)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); root=a.run_dir; errors=[]
    patch_hash=hashlib.sha256((root/'candidate.patch').read_bytes()).hexdigest()
    rows={k:json.loads((root/f).read_text()) for k,f in FILES.items()}
    for k,r in rows.items():
        if r.get('protocol_id')!='trl_grpo_kl_clip_precision_patch_v4': errors.append(f'{k}: protocol mismatch')
        if r.get('source',{}).get('revision')!=HEAD: errors.append(f'{k}: source revision mismatch')
        if r.get('status')!='complete': errors.append(f'{k}: method execution incomplete')
        if k.startswith('candidate') and r.get('source',{}).get('patch_sha256')!=PATCH: errors.append(f'{k}: patch digest mismatch')
    base=rows['base_f16']; candidate=rows['candidate_f16']
    metric_overflow=base.get('kl_metric') in ('Infinity','-Infinity','NaN')
    if not metric_overflow: errors.append('unmodified fp16 baseline did not reproduce non-finite aggregate KL metric')
    base_bad_grad=False
    for i,row in enumerate(EXPECTED_GRAD):
        for j,target in enumerate(row):
            actual=base.get('gradient',[[math.nan]*3]*2)[i][j]
            if isinstance(actual,(int,float)) and math.isfinite(float(actual)) and not close(actual,target): base_bad_grad=True
    if not base_bad_grad: errors.append('unmodified fp16 baseline gradients unexpectedly match analytic target')
    candidate_finite=(bool(candidate.get('loss_finite')) and math.isfinite(candidate.get('loss',math.nan)) and bool(candidate.get('gradient_finite')) and isinstance(candidate.get('kl_metric'),(int,float)) and math.isfinite(float(candidate['kl_metric'])))
    if not candidate_finite: errors.append('candidate fp16 path has non-finite loss, metric, or gradient')
    if not close(candidate.get('loss'),LOSS): errors.append('candidate fp16 loss outside analytic tolerance')
    for i,row in enumerate(EXPECTED_GRAD):
        for j,target in enumerate(row):
            actual=candidate.get('gradient',[[math.nan]*3]*2)[i][j]
            if not close(actual,target): errors.append(f'candidate fp16 gradient[{i},{j}] outside analytic tolerance')
            if j>=LENGTHS[i] and actual!=0: errors.append(f'candidate padding gradient[{i},{j}] is nonzero')
    b32=rows['base_f32']; c32=rows['candidate_f32']
    fp32_equal=(b32.get('loss')==c32.get('loss') and b32.get('gradient')==c32.get('gradient') and b32.get('kl_metric')==c32.get('kl_metric'))
    if not fp32_equal: errors.append('candidate fp32 control changed')
    out={'protocol_id':'trl_grpo_kl_clip_precision_patch_v4','status':'pass' if not errors else 'fail','errors':errors,'baseline_fp16_metric_overflow':metric_overflow,'baseline_fp16_gradient_mismatch':base_bad_grad,'candidate_fp16_finite':candidate_finite,'analytic_expected':{'loss':LOSS,'gradient':EXPECTED_GRAD,'kl_metric':R*K},'candidate_fp16_observed':{'loss':candidate.get('loss'),'gradient':candidate.get('gradient'),'kl_metric':candidate.get('kl_metric')},'baseline_fp16_observed':{'loss':base.get('loss'),'gradient':base.get('gradient'),'kl_metric':base.get('kl_metric')},'fp32_bit_identical':fp32_equal,'patch_sha256':patch_hash,'tolerance':{'relative':REL,'absolute':ABS},'source_imports':'none; standard-library math plus retained JSON and patch bytes'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n'); print(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)); return 0 if out['status']=='pass' else 1
if __name__=='__main__': raise SystemExit(main())
