#!/usr/bin/env python3
"""Recover the aggregate summary after the frozen runner's post-training KeyError."""
import argparse, json, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--bundle',type=Path,default=ROOT/'results/cpu-hh-length-normalized-dpo-v1/development/run-1')
args=parser.parse_args()
OUT=args.bundle.resolve()
if not (OUT/'runner.snapshot.py').is_file() or not (OUT/'auditor.snapshot.py').is_file():
    raise FileNotFoundError('expected frozen run snapshots in bundle')
sys.path.insert(0,str(ROOT/'scripts'))
import audit_cpu_hh_length_normalized_dpo_development_v1 as aud
import run_cpu_hh_length_normalized_dpo_development_v1 as runner
spec, protocol_hash=aud.locked_protocol()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):
    import os; os.environ[key]='1'
runtime=runner.verify_assets(spec)
torch.set_num_threads(spec['compute_limits']['threads'])
tokenizer=AutoTokenizer.from_pretrained(str(aud.MODEL_DIR),local_files_only=True,use_fast=True)
train,dev,counts=aud.independently_select(spec,tokenizer)
model=AutoModelForCausalLM.from_pretrained(str(aud.MODEL_DIR),local_files_only=True,torch_dtype=torch.float32,low_cpu_mem_usage=True).to(torch.device('cpu')).eval()
for p in model.parameters(): p.requires_grad_(False)
features=aud.encode_independently(dev,tokenizer,model,torch,spec['compute_limits']['max_sequence_tokens'])
chunk=spec['compute_limits']['logit_token_chunk_size']
cache_path=ROOT/'work/hh_run1_replay_cache.json'
cache_path.parent.mkdir(parents=True,exist_ok=True)
cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
if 'base' in cache:
    base=cache['base']
else:
    base=aud.evaluate_independently(features,model,None,torch,chunk)
    cache['base']=base; cache_path.write_text(json.dumps(cache,allow_nan=False))
results={'base':base}
for method in spec['learner']['methods']:
    results[method]={}
    for seed in spec['learner']['seeds']:
        path=OUT/'adapters'/f'{method}-seed-{seed}.pt'
        adapter=torch.load(path,map_location='cpu',weights_only=True)
        cache_key=f'{method}:{seed}'
        if cache_key in cache:
            metric=cache[cache_key]
        else:
            metric=aud.evaluate_independently(features,model,adapter,torch,chunk)
            cache[cache_key]=metric; cache_path.write_text(json.dumps(cache,allow_nan=False))
        results[method][str(seed)]={'metrics':metric,'adapter_file':path.relative_to(OUT).as_posix(),'adapter_sha256':aud.file_sha(path)}
        print('replayed',method,seed,metric['pair_accuracy'],flush=True)
seeds=spec['learner']['seeds']; n=len(dev); cfg=spec['metrics']['bootstrap']; gate=spec['metrics']['development_gate']; candidate='length_normalized_dpo'
acc=lambda m,s,i: results[m][str(s)]['metrics']['per_prompt'][i]['accuracy'] if m!='base' else results['base']['per_prompt'][i]['accuracy']
means={m:statistics.fmean([results[m][str(s)]['metrics']['pair_accuracy'] for s in seeds]) for m in spec['learner']['methods']}
base_rows=[acc('base',None,i) for i in range(n)]
by_seed={m:[[acc(m,s,i) for s in seeds] for i in range(n)] for m in spec['learner']['methods']}
agg_rows={m:[statistics.fmean(by_seed[m][i]) for i in range(n)] for m in spec['learner']['methods']}
ci_base=aud.paired_bootstrap(agg_rows[candidate],base_rows,cfg['resamples'],cfg['seed'])
ci_dpo=aud.paired_bootstrap(agg_rows[candidate],agg_rows['dpo'],cfg['resamples'],cfg['seed']+1)
ci_sft=aud.paired_bootstrap(agg_rows[candidate],agg_rows['chosen_sft'],cfg['resamples'],cfg['seed']+2)
base_acc=statistics.fmean(base_rows); cand_acc=statistics.fmean(agg_rows[candidate]); dpo_acc=statistics.fmean(agg_rows['dpo']); sft_acc=statistics.fmean(agg_rows['chosen_sft'])
gains_base=[results[candidate][str(s)]['metrics']['pair_accuracy']-base_acc for s in seeds]
gains_dpo=[results[candidate][str(s)]['metrics']['pair_accuracy']-results['dpo'][str(s)]['metrics']['pair_accuracy'] for s in seeds]
kl=[results[candidate][str(s)]['metrics']['mean_full_vocab_token_kl_to_base'] for s in seeds]
nll=statistics.fmean([results[candidate][str(s)]['metrics']['pair_nll'] for s in seeds])
decision={
'base_floor_pass':base_acc>=gate['minimum_base_pair_accuracy'],
'candidate_gain_vs_base_pass':cand_acc-base_acc>=gate['minimum_length_normalized_dpo_gain_vs_base'] and ci_base[0]>0,
'candidate_gain_vs_standard_dpo_pass':cand_acc-dpo_acc>=gate['minimum_length_normalized_dpo_gain_vs_standard_dpo'] and ci_dpo[0]>0,
'candidate_seed_consistency_vs_base_pass':sum(x>0 for x in gains_base)>=gate['minimum_seed_gains_vs_base'],
'candidate_seed_consistency_vs_standard_dpo_pass':sum(x>0 for x in gains_dpo)>=gate['minimum_seed_gains_vs_standard_dpo'],
'candidate_sft_noninferiority_pass':cand_acc-sft_acc>=-gate['maximum_pair_accuracy_regression_vs_chosen_sft'] and ci_sft[0]>-gate['maximum_pair_accuracy_regression_vs_chosen_sft'],
'kl_pass':all(x<=gate['maximum_mean_full_vocab_token_kl_per_seed'] for x in kl),
'nll_guard_pass':nll<=results['base']['pair_nll']+gate['nll_guard_vs_base_nats']}
decision['pass']=all(decision.values())
summary={'protocol_sha256':protocol_hash,'status':'completed','decision':decision,'selection_counts':counts,
 'train_selection':[{k:r[k] for k in ('source_index','context_hash','chosen_tokens','rejected_tokens')} for r in train],
 'development_selection':[{k:r[k] for k in ('source_index','context_hash','chosen_tokens','rejected_tokens')} for r in dev],
 'base_metrics':{k:v for k,v in results['base'].items() if k!='per_prompt'},'base_per_prompt':results['base']['per_prompt'],
 'arms':{m:{str(s):{'metrics':{k:v for k,v in results[m][str(s)]['metrics'].items() if k!='per_prompt'},'per_prompt':results[m][str(s)]['metrics']['per_prompt'],'adapter_file':results[m][str(s)]['adapter_file'],'adapter_sha256':results[m][str(s)]['adapter_sha256']} for s in seeds} for m in spec['learner']['methods']},
 'aggregate':{'base_pair_accuracy':base_acc,'standard_dpo_pair_accuracy':dpo_acc,'length_normalized_dpo_pair_accuracy':cand_acc,'chosen_sft_pair_accuracy':sft_acc,'length_normalized_dpo_minus_base_accuracy':cand_acc-base_acc,'length_normalized_dpo_minus_base_accuracy_prompt_bootstrap_95':ci_base,'length_normalized_dpo_minus_standard_dpo_accuracy':cand_acc-dpo_acc,'length_normalized_dpo_minus_standard_dpo_accuracy_prompt_bootstrap_95':ci_dpo,'length_normalized_dpo_minus_sft_accuracy':cand_acc-sft_acc,'length_normalized_dpo_minus_sft_accuracy_prompt_bootstrap_95':ci_sft,'length_normalized_dpo_seed_gains_vs_base':gains_base,'length_normalized_dpo_seed_gains_vs_standard_dpo':gains_dpo,'length_normalized_dpo_mean_pair_nll':nll,'standard_dpo_mean_pair_nll':statistics.fmean([results['dpo'][str(s)]['metrics']['pair_nll'] for s in seeds]),'chosen_sft_mean_pair_nll':statistics.fmean([results['chosen_sft'][str(s)]['metrics']['pair_nll'] for s in seeds]),'length_normalized_dpo_mean_token_kl_by_seed':kl,'base_length_only_pair_accuracy':results['base']['length_only_pair_accuracy'],'length_normalized_dpo_mean_token_pair_accuracy':statistics.fmean([results[candidate][str(s)]['metrics']['mean_token_normalized_pair_accuracy'] for s in seeds]),'length_normalized_dpo_length_only_pair_accuracy_by_seed':[results[candidate][str(s)]['metrics']['length_only_pair_accuracy'] for s in seeds]},
 'inputs_sha256':{'train.jsonl.gz':spec['dataset']['file_sha256_at_lock']['helpful-base/train.jsonl.gz'],'test.jsonl.gz_opaque_only':spec['dataset']['file_sha256_at_lock']['helpful-base/test.jsonl.gz'],'model_files':spec['model']['files_sha256_at_lock']},
 'runtime':runtime,'wall_seconds':None,'peak_rss_bytes':None,
 'recovery_note':'All 9 frozen-runner adapter files were saved and emitted dev metrics. The runner then raised KeyError while assembling aggregate output due to a missing in-memory metric key. This summary was reconstructed via the separately implemented independent auditor scoring path; audit.json records same-host score replay.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
sys.path.insert(0,str(ROOT/'scripts'))
from hh_reward_task import write_manifest
write_manifest(OUT)
print(json.dumps({'decision':decision,'aggregate':summary['aggregate']},indent=2),flush=True)
