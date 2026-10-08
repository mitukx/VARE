#!/usr/bin/env python3
"""Run the frozen training-only development budget selector for synthetic DPO v2."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, platform, statistics, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'protocols/synthetic_dpo_cpu_v2_development.json'
LOCK=ROOT/'protocols/synthetic_dpo_cpu_v2_development.lock.json'
RUNNER=ROOT/'scripts/run_synthetic_dpo_v2.py'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canon(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def write(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def load_runner():
    spec=importlib.util.spec_from_file_location('vare_dpo_v2_dev_runner',RUNNER)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); return module

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',required=True,type=Path); args=parser.parse_args()
    out=args.output.expanduser().resolve()
    if out.exists(): parser.error(f'output already exists: {out}')
    protocol=json.loads(SPEC.read_text()); lock=json.loads(LOCK.read_text()); locked=dict(lock); digest=locked.pop('sha256',None)
    if digest!=hashlib.sha256(canon(protocol)).hexdigest() or locked!=protocol: parser.error('development protocol lock mismatch')
    learner=load_runner(); out.mkdir(parents=True); (out/'seeds').mkdir()
    (out/'protocol.json').write_bytes(SPEC.read_bytes()); (out/'protocol.lock.json').write_bytes(LOCK.read_bytes())
    (out/'source').mkdir(); (out/'source/develop_synthetic_dpo_budget.py').write_bytes(Path(__file__).read_bytes()); (out/'source/run_synthetic_dpo_v2.py').write_bytes(RUNNER.read_bytes())
    started=time.perf_counter(); rows={u:[] for u in protocol['candidate_updates']}
    acceptance_mean_kl=protocol["selection_thresholds"]["maximum_mean_training_context_kl"]; per_seed_ceiling=protocol["selection_thresholds"]["maximum_per_seed_training_context_kl"]
    for seed in protocol['seeds']:
        examples=learner.generate_examples(learner.random.Random(seed),64,8)
        base=[[0.0]*learner.DIM for _ in range(learner.N_ACTIONS)]
        for updates in protocol['candidate_updates']:
            theta=learner.train(base,examples,protocol['beta'],protocol['learning_rate'],updates)
            contexts=list(dict.fromkeys(tuple(item['x']) for item in examples))
            kl_values=[]
            for context in contexts:
                probs=learner.softmax(learner.logits(theta,list(context)))
                kl_values.append(sum(p*__import__('math').log(p*learner.N_ACTIONS) for p in probs if p>0))
            rows[updates].append({'seed':seed,'training_preference_nll':learner.objective(theta,examples,protocol['beta']),'mean_training_context_kl_to_uniform':statistics.fmean(kl_values),'parameter_l2_delta':learner.parameter_delta(theta)})
        write(out/'seeds'/f'seed-{seed}.json',{'seed':seed,'training_examples':examples,'candidate_metrics':{str(u):next(row for row in rows[u] if row['seed']==seed) for u in protocol['candidate_updates']},'heldout_examples_generated':False})
    candidates=[]
    for updates in protocol['candidate_updates']:
        vals=rows[updates]; mean_kl=statistics.fmean(r['mean_training_context_kl_to_uniform'] for r in vals); max_kl=max(r['mean_training_context_kl_to_uniform'] for r in vals)
        candidates.append({'updates':updates,'mean_training_nll':statistics.fmean(r['training_preference_nll'] for r in vals),'mean_training_context_kl':mean_kl,'max_seed_training_context_kl':max_kl,'eligible':mean_kl<=acceptance_mean_kl and max_kl<=per_seed_ceiling})
    eligible=[r['updates'] for r in candidates if r['eligible']]
    selected=max(eligible) if eligible else None
    summary={'schema_version':1,'protocol_id':protocol['protocol_id'],'protocol_sha256':sha(SPEC),'lock_sha256':sha(LOCK),'runner_sha256':sha(RUNNER),'selection_rule':protocol['selection_rule'],'candidate_results':candidates,'selected_updates':selected,'selection_passed':selected==100,'heldout_data_generated':False,'confirmation_protocol_updates':100,'wall_seconds':time.perf_counter()-started,'python_version':sys.version,'platform':platform.platform(),'claim_limit':protocol['claim_boundary']}
    write(out/'summary.json',summary)
    files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write(out/'manifest.json',{'schema_version':1,'files':files})
    print(json.dumps({'selection_passed':summary['selection_passed'],'selected_updates':selected,'output':str(out),'candidates':candidates},indent=2))
    return 0 if summary['selection_passed'] else 2
if __name__=='__main__': raise SystemExit(main())
