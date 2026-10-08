#!/usr/bin/env python3
"""Verify the synthetic DPO v2 training-only budget development bundle."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,statistics,sys
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'protocols/synthetic_dpo_cpu_v2_development.json'; LOCK=ROOT/'protocols/synthetic_dpo_cpu_v2_development.lock.json'; RUNNER=ROOT/'scripts/run_synthetic_dpo_v2.py'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def close(a,b,label):
 if not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12): raise ValueError(f'{label} mismatch: {a!r} vs {b!r}')
def load(p):
 sys.dont_write_bytecode=True
 spec=importlib.util.spec_from_file_location('vare_dev_audit_runner',p); mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod
def audit(bundle):
 spec=json.loads(SPEC.read_text()); lock=json.loads(LOCK.read_text()); body=dict(lock); digest=body.pop('sha256',None)
 if body!=spec or digest!=hashlib.sha256(canon(spec)).hexdigest(): raise ValueError('development lock mismatch')
 if json.loads((bundle/'protocol.json').read_text())!=spec or json.loads((bundle/'protocol.lock.json').read_text())!=lock: raise ValueError('bundle protocol snapshot mismatch')
 manifest=json.loads((bundle/'manifest.json').read_text()); expected=manifest['files']; actual={}
 for n,d in expected.items():
  p=bundle.joinpath(*PurePosixPath(n).parts)
  if not p.resolve().is_relative_to(bundle.resolve()) or not p.is_file() or sha(p)!=d: raise ValueError(f'missing/invalid manifest file: {n}')
  actual[n]=d
 inventory={p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file() and p.name!='manifest.json'}
 if set(expected)!=inventory: raise ValueError('manifest inventory mismatch')
 runner_path=bundle/'source/run_synthetic_dpo_v2.py'
 dev_runner_path=bundle/'source/develop_synthetic_dpo_budget.py'
 if sha(runner_path)!=sha(RUNNER): raise ValueError('learner runner source snapshot mismatch')
 if sha(dev_runner_path)!=sha(ROOT/'scripts/develop_synthetic_dpo_budget.py'): raise ValueError('development runner source snapshot mismatch')
 runner=load(runner_path); raw_rows={u:[] for u in spec['candidate_updates']}
 for seed in spec['seeds']:
  raw=json.loads((bundle/'seeds'/f'seed-{seed}.json').read_text()); examples=raw.get('training_examples',[])
  if raw.get('seed')!=seed or raw.get('heldout_examples_generated') is not False or len(examples)!=512: raise ValueError(f'invalid training-only record: {seed}')
  if any(key.lower() in {'heldout_examples','heldout_metrics','heldout_data'} for key in raw): raise ValueError(f'heldout field present in development record: {seed}')
  if examples!=runner.generate_examples(runner.random.Random(seed),64,8): raise ValueError(f'development seed data regeneration mismatch: {seed}')
  base=[[0.0]*runner.DIM for _ in range(runner.N_ACTIONS)]
  for updates in spec['candidate_updates']:
   theta=runner.train(base,examples,spec['beta'],spec['learning_rate'],updates)
   contexts=list(dict.fromkeys(tuple(item['x']) for item in examples)); kvals=[]
   for x in contexts:
    probs=runner.softmax(runner.logits(theta,list(x))); kvals.append(sum(p*math.log(p*runner.N_ACTIONS) for p in probs if p>0))
   calc={'seed':seed,'training_preference_nll':runner.objective(theta,examples,spec['beta']),'mean_training_context_kl_to_uniform':statistics.fmean(kvals),'parameter_l2_delta':runner.parameter_delta(theta)}
   recorded=raw['candidate_metrics'][str(updates)]
   for k,v in calc.items(): close(recorded.get(k),v,f'seed {seed} updates {updates} {k}')
   raw_rows[updates].append(calc)
 candidates=[]
 thresholds=spec['selection_thresholds']
 for updates,rows in raw_rows.items():
  mean_kl=statistics.fmean(r['mean_training_context_kl_to_uniform'] for r in rows); max_kl=max(r['mean_training_context_kl_to_uniform'] for r in rows)
  candidates.append({'updates':updates,'mean_training_nll':statistics.fmean(r['training_preference_nll'] for r in rows),'mean_training_context_kl':mean_kl,'max_seed_training_context_kl':max_kl,'eligible':mean_kl<=thresholds['maximum_mean_training_context_kl'] and max_kl<=thresholds['maximum_per_seed_training_context_kl']})
 selected=max([r['updates'] for r in candidates if r['eligible']],default=None)
 summary=json.loads((bundle/'summary.json').read_text())
 if summary.get('candidate_results')!=candidates or summary.get('selected_updates')!=selected or summary.get('selection_passed')!=(selected==100) or summary.get('heldout_data_generated') is not False: raise ValueError('development summary does not reconstruct')
 return {'status':'verified','protocol_id':spec['protocol_id'],'seed_count':len(spec['seeds']),'candidate_count':len(candidates),'selected_updates':selected,'heldout_data_generated':False,'manifest_file_count':len(expected)}
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('bundle',type=Path); a=p.parse_args()
 try: result=audit(a.bundle.expanduser().resolve())
 except Exception as e: print(f'audit failed: {type(e).__name__}: {e}',file=sys.stderr); return 1
 print(json.dumps(result,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
