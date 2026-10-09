#!/usr/bin/env python3
"""Check the corrected frozen sequence-level KL precision protocol without importing TRL/PyTorch."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
HEAD='0aaea03f2fa449bc7a91f1973e7940da11da65da'
PATCH='0f6c60de679c3c8b467b2bc8b77b01e06f890486cd567322940e5ce8a2137468'
FILES={'baseline_f16':'pr-head-f16-sequence.json','candidate_f16':'candidate-f16-sequence.json','baseline_f32':'pr-head-f32-sequence.json','candidate_f32':'candidate-f32-sequence.json'}
BETA=.1; B=2; C=8.; Q=math.exp(.125); X=((16.,3.),(16.,16.,3.)); REL=.005; ABS=.1

def targets():
    loss=0.; gradients=[]
    for xs in X:
        n=len(xs); ys=[min(x,C) for x in xs]; ks=[math.exp(y)-y-1 for y in ys]
        loss += BETA*Q*sum(ks)/n/B
        gradients.append([BETA*Q/n/B*(sum(ks)/n+1-math.exp(y)) for y in ys]+[0.]*(3-n))
    return loss,gradients

def close(a,b): return isinstance(a,(int,float)) and math.isfinite(float(a)) and math.isclose(float(a),b,rel_tol=REL,abs_tol=ABS)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); root=a.run_dir; errors=[]
    patch_hash=hashlib.sha256((root/'candidate.patch').read_bytes()).hexdigest()
    rows={k:json.loads((root/f).read_text()) for k,f in FILES.items()}
    for k,r in rows.items():
        if r.get('protocol_id')!='trl_grpo_kl_clip_precision_patch_v3': errors.append(f'{k}: protocol mismatch')
        if r.get('source',{}).get('revision')!=HEAD: errors.append(f'{k}: source revision mismatch')
        if r.get('status')!='complete': errors.append(f'{k}: method execution incomplete')
        if k.startswith('candidate') and r.get('source',{}).get('patch_sha256')!=PATCH: errors.append(f'{k}: patch digest mismatch')
    base16=rows['baseline_f16']; cand16=rows['candidate_f16']; el,eg=targets()
    base_defect=base16.get('kl_metric')=='Infinity'
    if not base_defect: errors.append('unmodified fp16 baseline did not reproduce non-finite KL metric')
    candidate_finite=bool(cand16.get('loss_finite')) and math.isfinite(cand16.get('loss',math.nan)) and bool(cand16.get('gradient_finite')) and isinstance(cand16.get('kl_metric'),(int,float)) and math.isfinite(float(cand16['kl_metric']))
    if not candidate_finite: errors.append('candidate fp16 sequence output contains non-finite values')
    if not close(cand16.get('loss'),el): errors.append('candidate fp16 loss outside analytic tolerance')
    for i,row in enumerate(eg):
        for j,target in enumerate(row):
            actual=cand16.get('gradient',[[math.nan]*3]*2)[i][j]
            if not close(actual,target): errors.append(f'candidate fp16 gradient[{i},{j}] outside analytic tolerance')
            if j>=len(X[i]) and actual!=0: errors.append(f'candidate fp16 padding gradient[{i},{j}] is nonzero')
    b32=rows['baseline_f32']; c32=rows['candidate_f32']
    fp32_equal=(b32.get('loss')==c32.get('loss') and b32.get('gradient')==c32.get('gradient') and b32.get('kl_metric')==c32.get('kl_metric'))
    if not fp32_equal: errors.append('candidate fp32 control changed')
    out={'protocol_id':'trl_grpo_kl_clip_precision_patch_v3','status':'pass' if not errors else 'fail','errors':errors,'baseline_fp16_expected_defect_reproduced':base_defect,'candidate_fp16_finite':candidate_finite,'analytic_expected':{'loss':el,'gradient':eg},'candidate_fp16_observed':{'loss':cand16.get('loss'),'gradient':cand16.get('gradient'),'kl_metric':cand16.get('kl_metric')},'fp32_bit_identical':fp32_equal,'patch_sha256':patch_hash,'tolerance':{'relative':REL,'absolute':ABS},'source_imports':'none; standard-library math plus retained JSON and patch bytes'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n'); print(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)); return 0 if out['status']=='pass' else 1
if __name__=='__main__': raise SystemExit(main())
