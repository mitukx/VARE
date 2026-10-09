#!/usr/bin/env python3
"""Independently evaluate the frozen multi-token TRL KL precision source-path records."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path

HEAD="0aaea03f2fa449bc7a91f1973e7940da11da65da"
PATCH="0f6c60de679c3c8b467b2bc8b77b01e06f890486cd567322940e5ce8a2137468"
NAMES=("pr-head-f16-sequence","candidate-f16-sequence","pr-head-f32-sequence","candidate-f32-sequence","candidate-f16-token")
BETA=.1; C=10.; Q=math.exp(.25); X=((20.,4.),(20.,20.,4.)); REL=.005; ABS=.1

def close(a,b): return math.isclose(float(a),float(b),rel_tol=REL,abs_tol=ABS)
def expected(level):
    loss=0.; grads=[]
    for xs in X:
        n=len(xs); clipped=[min(x,C) for x in xs]
        ks=[math.exp(v)-v-1 for v in clipped]
        loss += BETA*Q*sum(ks)/n/len(X)
        if level=="sequence":
            row=[BETA*Q/n/len(X)*(sum(ks)/n+1-math.exp(v)) for v in clipped]
        else:
            row=[-BETA*Q*v/n/len(X) for v in clipped]
        grads.append(row+[0.]*(3-n))
    return loss,grads

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    root=a.run_dir; errors=[]; checks={}
    patch_sha=hashlib.sha256((root/'candidate.patch').read_bytes()).hexdigest()
    rows={n:json.loads((root/f'{n}.json').read_text()) for n in NAMES}
    for n,r in rows.items():
        if r.get('protocol_id')!='trl_grpo_kl_clip_precision_patch_v2': errors.append(f'{n}: protocol id mismatch')
        if r.get('source',{}).get('revision')!=HEAD: errors.append(f'{n}: revision mismatch')
        if r.get('status')!='complete': errors.append(f'{n}: method incomplete')
        if n.startswith('candidate-') and r.get('source',{}).get('patch_sha256')!=PATCH: errors.append(f'{n}: patch mismatch')
    for name,level in [('candidate-f16-sequence','sequence'),('candidate-f16-token','token')]:
        r=rows[name]; el,eg=expected(level); ok=bool(r.get('loss_finite')) and close(r['loss'],el)
        for i,row in enumerate(eg):
            for j,target in enumerate(row):
                actual=r['gradient'][i][j]
                ok &= math.isfinite(actual) and close(actual,target)
                if j >= len(X[i]) and actual != 0: ok=False
        checks[name]={"candidate_matches_analytic":bool(ok),"expected_loss":el,"actual_loss":r['loss'],"expected_gradient":eg,"actual_gradient":r['gradient']}
        if not ok: errors.append(f'{name}: candidate differs from analytic masked objective')
    base=rows['pr-head-f32-sequence']; cand=rows['candidate-f32-sequence']
    f32same=(base['loss']==cand['loss'] and base['gradient']==cand['gradient'] and base['kl_metric']==cand['kl_metric'])
    checks['fp32_control']={"bit_identical":f32same}
    if not f32same: errors.append('fp32 candidate control changed')
    baseline_f16=rows['pr-head-f16-sequence']
    baseline_has_inf=not math.isfinite(baseline_f16.get('kl_metric',math.nan))
    protocol_nonpass=baseline_has_inf # Frozen acceptance said all fixture outputs must be finite, without exempting the known-bad baseline.
    out={"protocol_id":"trl_grpo_kl_clip_precision_patch_v2","status":"candidate_checks_pass_protocol_nonpass" if not errors and protocol_nonpass else ("pass" if not errors else "fail"),"errors":errors,"candidate_checks":checks,"pr_head_fp16_sequence_baseline":{"kl_metric":baseline_f16.get('kl_metric'),"contains_nonfinite":baseline_has_inf},"frozen_acceptance_adjudication":"non-pass: the broad all-fixture-finite gate fails on the unpatched PR-head baseline; preserve as a protocol design error, not a candidate failure","source_imports":"none; reads retained JSON, protocol lock and patch bytes","patch_sha256":patch_sha,"tolerance":{"relative":REL,"absolute":ABS}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2,sort_keys=True)); return 0 if out['status']=='pass' else 1
if __name__=='__main__': raise SystemExit(main())
