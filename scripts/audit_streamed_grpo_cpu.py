"""Offline replay of the existing GRPO equation against streamed model artifacts."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
from audit_grpo_e2e_cpu import digest, require
from validate_grpo_e2e_cpu import reference_update

ROOT=Path(__file__).resolve().parents[1]


def tensor_error(actual, expected):
    require(actual.keys()==expected.keys(),"numerical tensor coverage mismatch")
    require(all(actual[name].shape==value.shape and actual[name].dtype==value.dtype for name,value in expected.items()),"numerical tensor shape/dtype mismatch")
    error=max(float((actual[name]-value).abs().max()) for name,value in expected.items())
    require(math.isfinite(error) and error<2e-6,"numerical tensor mismatch")
    return error


def audit(root, model_dir, rvl_source, expected_manifest):
    import torch
    require(not expected_manifest.resolve().is_relative_to(root.resolve()),"external manifest must be outside artifact")
    manifest=json.loads((root/'SHA256.json').read_text())
    require(manifest==json.loads(expected_manifest.read_text()),"external artifact anchor mismatch")
    for name,expected in manifest.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,"unsafe artifact path")
        require(digest(root/name)==expected,"artifact bytes differ: "+name)
    protocol=json.loads((root/'protocol.json').read_text())
    protocols={f'vare.streamed-grpo-systems.v{i}':ROOT/f'benchmarks/streamed_grpo_cpu_v{i}/protocol.json' for i in (1,2,3)}
    require(protocol.get('schema') in protocols,"unsupported protocol")
    require(protocol==json.loads(protocols[protocol['schema']].read_text()),"frozen protocol mismatch")
    summary=json.loads((root/'summary.json').read_text())
    for name,expected in summary['source_files'].items():
        require(digest(root/'source'/name)==expected,"source snapshot mismatch")
    require(digest(root/'source/scripts/validate_streamed_grpo_cpu.py')==protocol['runner_sha256'],"runner mismatch")
    require(digest(Path(__file__).with_name('validate_grpo_e2e_cpu.py'))==summary['source_files']['scripts/validate_grpo_e2e_cpu.py'],"reference source mismatch")
    for name,expected in protocol['model_files'].items():
        require(digest(model_dir/name)==expected,"reference model mismatch")
    require(subprocess.check_output(['git','-C',str(rvl_source),'rev-parse','HEAD'],text=True).strip()==protocol['rvl_revision'],"RVL revision mismatch")
    require(subprocess.check_output(['git','-C',str(rvl_source),'status','--porcelain'],text=True).strip()=='M src/rvl_systems/hf_backend.py',"unexpected RVL edits")
    require(digest(rvl_source/'src/rvl_systems/hf_backend.py')==protocol['rvl_patched_backend_sha256'],"RVL patch mismatch")
    sys.path.insert(0,str(ROOT/'src'))
    sys.path.insert(0,str(rvl_source/'src'))
    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer,HFTTrainerConfig
    from vare.trajectory_store import audit_trajectory_store
    torch.set_num_threads(1)
    backend=HFLocalBackend(str(model_dir),device='cpu',precision='fp32')
    model=backend.model
    for name,parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('transformer.wpe.') or name.startswith('transformer.ln_f.'))
    config=HFTTrainerConfig(learning_rate=1e-4)
    HFCausalLMGRPOTrainer(model,config=config)  # Reuse the trainer's dropout setup.
    updates=json.loads((root/'updates.json').read_text())
    require(len(updates)==summary['optimizer_calls']==4,"actual update coverage mismatch")
    results=[]
    for update in updates:
        batch=[SimpleNamespace(attempt=SimpleNamespace(**row['attempt']),verification=SimpleNamespace(**row['verification'])) for row in update['batch']]
        for i,row in enumerate(batch):
            row.attempt.created_step=i
        require(len(batch)==protocol['group_size'],"group coverage mismatch")
        before=torch.load(root/(batch[0].attempt.policy_id+'.pt'),weights_only=True)
        candidate=torch.load(root/(update['candidate']+'.pt'),weights_only=True)
        saved=torch.load(root/(update['candidate']+'-gradients.pt'),weights_only=True)
        captured={}
        original=torch.nn.utils.clip_grad_norm_
        def clip(parameters,*args,**kwargs):
            captured['vectors']={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
            captured['norm']=float(original(parameters,*args,**kwargs))
            return captured['norm']
        with patch.object(torch.nn.utils,'clip_grad_norm_',clip):
            ref=reference_update(model,before,batch,config)
        norm=update['metrics']['grad_norm']
        require(math.isclose(norm,captured['norm'],rel_tol=2e-6,abs_tol=2e-6),"unclipped norm mismatch")
        coefficient=(config.max_grad_norm/(torch.tensor(norm,dtype=torch.float32)+1e-6)).clamp(max=1)
        reconstructed={n:v/coefficient for n,v in saved['actual_gradients'].items()}
        errors={'preclip_gradient':tensor_error(reconstructed,captured['vectors']),
                'clipped_gradient':tensor_error(saved['actual_gradients'],ref['grads']),
                'parameter':tensor_error(candidate['model'],ref['model'])}
        require(abs(update['metrics']['loss']-ref['loss'])<2e-6,"loss mismatch")
        require(torch.equal(before['cpu_rng'],candidate['cpu_rng']),"RNG mismatch")
        require(candidate['optimizer']['param_groups']==ref['optimizer']['param_groups'],"optimizer configuration mismatch")
        require(candidate['optimizer']['state'].keys()==ref['optimizer']['state'].keys(),"optimizer coverage mismatch")
        for key,state in candidate['optimizer']['state'].items():
            require(tensor_error(state,ref['optimizer']['state'][key])<2e-6,"optimizer state mismatch")
        require(ref['max_behavior_alignment_error']<1e-4,"behavior alignment mismatch")
        results.append({'candidate':update['candidate'],'errors':errors,'preclip_norm':norm})
    require(audit_trajectory_store(root/'snapshot.db')==summary['audit'],"trajectory audit differs")
    rounds=json.loads((root/'rounds.json').read_text())
    require(rounds[1]['result']['decision']['accepted'] is False,"forced rejection missing")
    require(rounds[2]['ack_interrupted'] and rounds[2]['active']['policy_id']==updates[3]['batch'][0]['attempt']['policy_id'],"recovered state not used for next update")
    return {'pass':True,'updates':len(results),'numerics':results,'scope':'Externally anchored bytes and independent full-graph/preclip/Adam/RNG replay. Live checkpoint decoding and logical recovery are separately retained; no task-capability or learner-SIGKILL claim.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--rvl-source',type=Path,required=True)
    parser.add_argument('--expected-manifest',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.output,args.model_dir,args.rvl_source,args.expected_manifest),indent=2))
