#!/usr/bin/env python3
"""Execute the protocol-driven multi-token fp16 GRPO KL precision fixture on pinned TRL source."""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess, sys
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace
import torch

HEAD='0aaea03f2fa449bc7a91f1973e7940da11da65da'
PATCH='0f6c60de679c3c8b467b2bc8b77b01e06f890486cd567322940e5ce8a2137468'
SOURCE='trl/trainer/grpo_trainer.py'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def source_state(root, patched):
    root=root.resolve(); rev=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if rev!=HEAD: raise ValueError(f'source revision mismatch: {rev}')
    dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=all'],text=True).strip()
    patch_hash=None
    if patched:
        changed=subprocess.check_output(['git','-C',str(root),'diff','--name-only','HEAD'],text=True).splitlines()
        untracked=subprocess.check_output(['git','-C',str(root),'ls-files','--others','--exclude-standard'],text=True).splitlines()
        patch=subprocess.check_output(['git','-C',str(root),'diff','--binary','HEAD'])
        patch_hash=hashlib.sha256(patch).hexdigest()
        if changed!=[SOURCE] or untracked or patch_hash!=PATCH: raise ValueError('candidate patch/source path identity mismatch')
    elif dirty: raise ValueError('unmodified source checkout is dirty')
    return {'revision':rev,'patched':patched,'patch_sha256':patch_hash,'trainer_sha256':sha(root/SOURCE)}

class Accelerator:
    num_processes=1
    @staticmethod
    def reduce(x,reduction='sum'):
        if reduction!='sum': raise ValueError(reduction)
        return x
    @staticmethod
    def gather(x): return x
class DummyModel(torch.nn.Module):
    def __init__(self,dtype):
        super().__init__(); self.anchor=torch.nn.Parameter(torch.zeros((),dtype=dtype))

def safe(value):
    if isinstance(value,float) and not math.isfinite(value): return 'Infinity' if value>0 else ('-Infinity' if value<0 else 'NaN')
    if isinstance(value,dict): return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe(v) for v in value]
    return value

def execute(source_root,protocol,patched,dtype_name):
    prov=source_state(source_root,patched)
    root=str(source_root.resolve())
    if root not in sys.path: sys.path.insert(0,root)
    from trl.trainer.grpo_config import GRPOConfig
    from trl.trainer.grpo_trainer import GRPOTrainer
    f=protocol['fixture']; dtype={'float16':torch.float16,'float32':torch.float32}[dtype_name]
    policy=torch.tensor(f['policy_logps'],dtype=dtype,requires_grad=True)
    old=torch.tensor(f['old_logps'],dtype=dtype)
    x=torch.tensor(f['kl_log_ratios'],dtype=dtype)
    ref=policy.detach()+x
    mask=torch.tensor(f['completion_mask'],dtype=dtype)
    entropies=torch.zeros_like(policy)
    model=DummyModel(dtype); tr=GRPOTrainer.__new__(GRPOTrainer); tr.model=model
    tr.args=SimpleNamespace(use_bias_correction_kl=f['use_bias_correction_kl'],delta=None,kl_log_ratio_clip=f['clip'],steps_per_generation=1)
    tr.accelerator=Accelerator(); tr.aux_loss_enabled=False; tr.beta=f['beta']; tr.current_gradient_accumulation_steps=1
    tr.epsilon_low=tr.epsilon_high=.2; tr.importance_sampling_level=f['importance_sampling_level']; tr.loss_type='grpo'
    tr.off_policy_mask_threshold=None; tr.top_entropy_quantile=1.; tr.use_vllm=False; tr.vllm_importance_sampling_correction=False
    tr._entropy_bonus_enabled=False; tr._metrics={'train':defaultdict(list),'eval':defaultdict(list)}
    def get_logps(self,*args,**kwargs): return policy,entropies,None
    tr._get_per_token_logps_and_entropies=MethodType(get_logps,tr)
    inputs={'prompt_ids':torch.ones((mask.shape[0],1),dtype=torch.long),
      'prompt_mask':torch.ones((mask.shape[0],1),dtype=torch.long),
      'completion_ids':torch.ones(mask.shape,dtype=torch.long),'completion_mask':mask,
      'advantages':torch.tensor(f['advantages'],dtype=dtype),'old_per_token_logps':old,'ref_per_token_logps':ref}
    GRPOConfig(output_dir='/tmp/vare-kl-precision-v5',kl_log_ratio_clip=f['clip'])
    loss=GRPOTrainer._compute_loss(tr,model,inputs); loss.backward()
    return {'protocol_id':protocol['protocol_id'],'source':prov,'dtype':dtype_name,
      'importance_sampling_level':f['importance_sampling_level'],'status':'complete',
      'loss':float(loss.detach().float().item()),'loss_finite':bool(torch.isfinite(loss).item()),
      'gradient':policy.grad.detach().float().tolist(),'gradient_finite':bool(torch.isfinite(policy.grad).all().item()),
      'kl_metric':float(tr._metrics['train']['kl'][-1]),'fixture_sha256':hashlib.sha256(json.dumps(f,sort_keys=True,separators=(',',':')).encode()).hexdigest()}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--source-root',type=Path,required=True); p.add_argument('--protocol',type=Path,required=True); p.add_argument('--patched',action='store_true'); p.add_argument('--dtype',choices=['float16','float32'],required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    protocol=json.loads(a.protocol.read_text()); protocol_hash=sha(a.protocol)
    if protocol['protocol_id']!='trl_grpo_kl_clip_precision_patch_v5': raise ValueError('protocol ID mismatch')
    if protocol['source']['candidate_revision']!=HEAD or protocol['source']['patch_sha256']!=PATCH: raise ValueError('locked source identity mismatch')
    if sha(Path(__file__))!=protocol['artifacts']['runner_sha256']: raise ValueError('runner hash differs from frozen protocol')
    result=execute(a.source_root,protocol,a.patched,a.dtype); result['protocol_sha256']=protocol_hash
    a.output.parent.mkdir(parents=True,exist_ok=True); rendered=json.dumps(safe(result),indent=2,sort_keys=True,allow_nan=False)+'\n'; a.output.write_text(rendered); print(rendered,end='')
if __name__=='__main__': main()
