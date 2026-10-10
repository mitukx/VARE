import hashlib
from itertools import product
import json
import math
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
import torch
repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
from src.rvl_systems.types import Generation, VerifiedGeneration

torch.set_num_threads(2)
class BinaryPolicy(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(math.log(p/(1-p))))
    def forward(self, input_ids):
        logits = torch.stack([self.theta * 0, self.theta])
        return SimpleNamespace(logits=logits.expand(*input_ids.shape, 2))

rows=[]
for p in [.1, .9]:
    for group in product((0,1), repeat=2):
        model=BinaryPolicy(p)
        initial=float(model.theta.detach())
        logps=torch.log_softmax(model(torch.zeros(1,1,dtype=torch.long)).logits[0,0],-1).detach().tolist()
        samples=[]
        for a in group:
            metadata={'prompt_token_ids':[0], 'response_token_ids':[a], 'response_token_logprobs':[logps[a]], 'sampling_temperature':1.}
            generation=Generation('group','prompt',str(a),logps[a],1,0,metadata)
            samples.append(VerifiedGeneration(generation,float(a),0,1))
        trainer=HFCausalLMGRPOTrainer(model,config=HFTTrainerConfig(max_grad_norm=100))
        trainer.optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        metrics=trainer.train_step(samples)
        rows.append({'p':p,'group':group,'ascent_gradient':-float(model.theta.grad),'theta_before':initial,'theta_after':float(model.theta.detach()),'metrics':metrics})

summaries=[]
for p in [.1,.9]:
    fresh=consumed=0.
    selected=int(p>.5)
    for row in [r for r in rows if r['p']==p]:
        a,b=row['group']
        target=lambda x:p if x else 1-p
        fresh+=target(a)*target(b)*row['ascent_gradient']
        consumed+=(target(b) if a==selected else 0.)*row['ascent_gradient']
    summaries.append({'p':p,'fresh':fresh,'consumed':consumed,'independent_donor':fresh,'fresh_oracle':.09/math.sqrt(1+4e-6),'consumed_oracle':.05/math.sqrt(1+4e-6)})
checks={'gradient_matches_finite_law':all(abs(r['fresh']-r['fresh_oracle'])<1e-6 and abs(r['consumed']-r['consumed_oracle'])<1e-6 for r in summaries),
    'zero_behavior_mismatch':all(abs(r['metrics']['max_abs_log_ratio'])<1e-6 and abs(r['metrics']['behavior_kl_estimate'])<1e-6 for r in rows)}
print(json.dumps({'rows':rows,'summaries':summaries,'checks':checks,'pass':all(checks.values()),'source_sha256':hashlib.sha256((repo/'src/rvl_systems/hf_trainer.py').read_bytes()).hexdigest(),'torch':torch.__version__,'python':platform.python_version(),'note':'Controlled tensor policy in actual train_step. Group probabilities analytically assigned, not an actual adaptive pretrained LM training run.'},indent=2))
assert all(checks.values()),checks
