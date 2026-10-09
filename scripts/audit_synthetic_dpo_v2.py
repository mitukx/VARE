#!/usr/bin/env python3
"""Reconstruct and verify the frozen synthetic DPO v2 confirmation bundle."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, random, statistics, sys
from pathlib import Path, PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'protocols/synthetic_dpo_cpu_v2.json'
LOCK=ROOT/'protocols/synthetic_dpo_cpu_v2.lock.json'
RUNNER=ROOT/'scripts/run_synthetic_dpo_v2.py'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def load_runner(path):
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('vare_dpo_v2_audit_runner',path)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); return module
def close(a,b,label):
    if not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError(f'reconstruction mismatch for {label}: recorded={a!r} reconstructed={b!r}')
def verify_manifest(bundle):
    manifest=json.loads((bundle/'manifest.json').read_text()); files=manifest.get('files')
    if not isinstance(files,dict): raise ValueError('manifest files field is not a mapping')
    expected=set()
    for name,digest in files.items():
        rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name: raise ValueError(f'unsafe manifest path: {name}')
        path=bundle.joinpath(*rel.parts)
        if not path.resolve().is_relative_to(bundle.resolve()) or not path.is_file(): raise ValueError(f'missing/escaping file: {name}')
        if sha(path)!=digest: raise ValueError(f'hash mismatch: {name}')
        expected.add(rel.as_posix())
    actual={p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file() and p.name!='manifest.json'}
    if actual!=expected: raise ValueError(f'manifest inventory mismatch: missing={sorted(expected-actual)}, unlisted={sorted(actual-expected)}')
    return manifest

def verify(bundle):
    spec=json.loads(SPEC.read_text()); lock=json.loads(LOCK.read_text()); lock_body=dict(lock); lock_hash=lock_body.pop('sha256',None)
    if lock_body!=spec or lock_hash!=hashlib.sha256(canonical(spec)).hexdigest(): raise ValueError('checked-in protocol and lock disagree')
    if json.loads((bundle/'protocol.json').read_text())!=spec or json.loads((bundle/'protocol.lock.json').read_text())!=lock: raise ValueError('bundle protocol snapshots differ from frozen protocol')
    runner_snapshot=bundle/'source/run_synthetic_dpo_v2.py'
    if not runner_snapshot.is_file() or sha(runner_snapshot)!=sha(RUNNER): raise ValueError('runner source snapshot differs from checked-in runner')
    summary=json.loads((bundle/'summary.json').read_text())
    for key,value in [('protocol_id',spec['protocol_id']),('protocol_canonical_sha256',hashlib.sha256(canonical(spec)).hexdigest()),('protocol_file_sha256',sha(bundle/'protocol.json')),('protocol_lock_sha256',sha(bundle/'protocol.lock.json')),('source_script_sha256',sha(runner_snapshot))]:
        if summary.get(key)!=value: raise ValueError(f'summary {key} mismatch')
    verify_manifest(bundle)
    runner=load_runner(runner_snapshot); beta=spec['learner']['beta']; lr=spec['learner']['learning_rate']; updates=spec['learner']['updates']
    all_metrics={arm:[] for arm in spec['arms']}; improvements=[]; invalid_shuffle_counts=[]
    for seed in spec['seeds']:
        raw=json.loads((bundle/'seeds'/f'seed-{seed}.json').read_text())
        train=raw.get('train_examples',[]); held=raw.get('heldout_examples',[])
        if raw.get('seed')!=seed or len(train)!=512 or len(held)!=256: raise ValueError(f'seed record/count mismatch: {seed}')
        rng=random.Random(seed)
        if train!=runner.generate_examples(rng,64,8) or held!=runner.generate_examples(rng,128,2):
            raise ValueError(f'seed/split data regeneration mismatch: {seed}')
        if {tuple(item['x']) for item in train} & {tuple(item['x']) for item in held}: raise ValueError(f'train/heldout context overlap: {seed}')
        base=[[0.0]*runner.DIM for _ in range(runner.N_ACTIONS)]
        reconstructed={'reference':{**runner.evaluate(base,held),'parameter_l2_delta':0.0,'updates':0}}
        labels_by_arm={'clean_dpo':[item['chosen'] for item in train]}
        noisy=labels_by_arm['clean_dpo'][:]; noise_rng=random.Random(seed+200000)
        for i,item in enumerate(train):
            if noise_rng.random()<0.2: noisy[i]=item['rejected'] if noisy[i]==item['chosen'] else item['chosen']
        shuffled=labels_by_arm['clean_dpo'][:]; random.Random(seed+100000).shuffle(shuffled)
        invalid_shuffle_counts.append(sum(label not in (item['a'],item['b']) for item,label in zip(train,shuffled)))
        labels_by_arm.update({'flip_20pct':noisy,'shuffle_labels':shuffled})
        for arm,labels in labels_by_arm.items():
            examples=[{**item,'chosen':label,'rejected':item['b'] if label==item['a'] else item['a']} for item,label in zip(train,labels)]
            theta=runner.train(base,examples,beta,lr,updates)
            reconstructed[arm]={**runner.evaluate(theta,held),'parameter_l2_delta':runner.parameter_delta(theta),'updates':updates}
        recorded=raw.get('metrics',{})
        if set(recorded)!=set(reconstructed): raise ValueError(f'arm inventory mismatch for seed {seed}')
        for arm,metrics in reconstructed.items():
            for metric,value in metrics.items(): close(recorded[arm].get(metric),value,f'seed {seed} {arm}.{metric}')
            all_metrics[arm].append({'seed':seed,**metrics})
        improvements.append(reconstructed['reference']['heldout_preference_nll']-reconstructed['clean_dpo']['heldout_preference_nll'])
    mean=statistics.fmean(improvements); bootstrap=spec['metrics']['paired_bootstrap']; ci=runner.bootstrap_interval(improvements,bootstrap['resamples'],bootstrap['seed'])
    mean_kl=statistics.fmean(r['mean_kl_to_uniform'] for r in all_metrics['clean_dpo']); improved=sum(v>0 for v in improvements); acc=spec['metrics']['acceptance']
    accepted=(mean>=acc['mean_clean_nll_improvement_at_least_nats_per_pair'] and ci[0]>acc['paired_bootstrap_lower_bound_above'] and improved>=acc['minimum_seeds_with_clean_nll_improvement'] and mean_kl<=acc['maximum_mean_clean_policy_kl'])
    close(summary.get('clean_dpo_vs_reference_nll_improvement_mean'),mean,'summary mean NLL improvement')
    if summary.get('clean_dpo_vs_reference_nll_improvement_per_seed')!=improvements: raise ValueError('per-seed NLL improvements mismatch')
    if summary.get('clean_dpo_vs_reference_nll_improvement_paired_bootstrap_95pct')!=ci: raise ValueError('bootstrap interval mismatch')
    close(summary.get('clean_dpo_mean_kl_to_uniform'),mean_kl,'summary mean KL')
    if summary.get('clean_dpo_improved_seed_count')!=improved or summary.get('acceptance_passed')!=accepted: raise ValueError('acceptance summary does not reconstruct')
    invalid_shuffle_seed_count=sum(count>0 for count in invalid_shuffle_counts)
    return {'status':'verified','protocol_id':spec['protocol_id'],'seed_count':len(spec['seeds']),'reconstructed_arm_seed_metrics':sum(map(len,all_metrics.values())),'manifest_file_count':len(json.loads((bundle/'manifest.json').read_text())['files']),'acceptance_passed':accepted,'mean_clean_nll_improvement':mean,'paired_bootstrap_95pct':ci,'mean_clean_policy_kl':mean_kl,'invalid_shuffle_control_seed_count':invalid_shuffle_seed_count,'shuffle_control_warning':'The v2 runner shuffles absolute chosen action IDs rather than pairwise orientation; exclude this arm from inference.' if invalid_shuffle_seed_count else None}

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('bundle',type=Path); args=parser.parse_args()
    try: result=verify(args.bundle.expanduser().resolve())
    except Exception as exc: print(f'audit failed: {type(exc).__name__}: {exc}',file=sys.stderr); return 1
    print(json.dumps(result,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
