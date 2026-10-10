"""Controlled TCP verifier -> existing RVL GRPO train_step connection."""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

async def run():
    import torch
    from src.rvl_systems.types import Generation
    from src.rvl_systems.verifier import FunctionalVerifier
    from src.rvl_systems.verifier_deployment import VerifierDeploymentCoordinator, publish_verifier_artifact
    from src.rvl_systems.verifier_rpc import TCPVerifierClient, VerifierWorkerServer
    from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    entered, release = asyncio.Event(), asyncio.Event()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a = publish_verifier_artifact(root/'a', version=1, content=b'{"0":0.1,"1":0.9}', suffix='.json')
        b = publish_verifier_artifact(root/'b', version=1, content=b'{"0":0.9,"1":0.1}', suffix='.json')
        async def loader(manifest):
            if manifest.sha256 == a.sha256:
                entered.set()
                await release.wait()
            table = json.loads(Path(manifest.path).read_text())
            return FunctionalVerifier(lambda g: table[g.response])
        server = VerifierWorkerServer(FunctionalVerifier(lambda _:0), worker_id='v', deployment_loader=loader, deployment_state_path=root/'worker.json')
        client = TCPVerifierClient(*(await server.start()))
        old = VerifierDeploymentCoordinator([client], state_path=root/'coordinator.json')
        task = asyncio.create_task(client.prepare_verifier(asdict(a),coordinator_epoch=old.epoch))
        try:
            await asyncio.wait_for(entered.wait(),5)
            new = VerifierDeploymentCoordinator([client],state_path=root/'coordinator.json')
            await new.prepare(b)
            release.set()
            try:
                await task
                rejected = False
            except RuntimeError:
                rejected = True
            await new.activate()
            health = await client.ping()
            samples=[]
            for token in (0,1):
                g=Generation('p','prompt',str(token),-math.log(2),1,0,{'prompt_token_ids':[0],'response_token_ids':[token],'response_token_logprobs':[-math.log(2)],'sampling_temperature':1.0})
                samples.append(await client.verify(g,expected_verifier_version=1))
            class BinaryPolicy(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.theta=torch.nn.Parameter(torch.tensor(0.0))
                def forward(self,input_ids):
                    return SimpleNamespace(logits=torch.stack([self.theta,-self.theta]).expand(input_ids.shape[0],input_ids.shape[1],2))
            model=BinaryPolicy()
            trainer=HFCausalLMGRPOTrainer(model,config=HFTTrainerConfig(learning_rate=.1,max_grad_norm=100))
            trainer.optimizer=torch.optim.SGD(model.parameters(),lr=.1)
            metrics=trainer.train_step(samples)
            grad=float(model.theta.grad)
            theta=float(model.theta.detach())
            r0,r1=[s.reward for s in samples]
            actual_oracle=-(r0-r1)/(2*math.sqrt(((r0-r1)/2)**2+1e-6))
            intended_oracle=-.8/(2*math.sqrt(.4**2+1e-6))
            return {'rewards':[r0,r1],'versions':[s.verifier_version for s in samples],'active_sha256':health['active_artifact_sha256'],'intended_sha256':b.sha256,'obsolete_prepare_rejected':rejected,'loss_gradient':grad,'actual_reward_gradient_oracle':actual_oracle,'intended_B_gradient_oracle':intended_oracle,'gradient_error_to_intended':abs(grad-intended_oracle),'agrees_with_actual_rewards':abs(grad-actual_oracle)<=1e-6,'theta_after_sgd':theta,'sgd_error':abs(theta+.1*grad),'prob_response_0_after':float(torch.sigmoid(2*model.theta).detach()),'metrics':metrics,'torch':torch.__version__,'note':'Two-outcome tensor integration, not a pretrained LM or task-capability experiment.'}
        finally:
            release.set()
            await asyncio.gather(task,return_exceptions=True)
            await server.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    repo=Path(a.repo).resolve();sys.path.insert(0,str(repo))
    result=asyncio.run(asyncio.wait_for(run(),60))
    result['worker_source_sha256']=hashlib.sha256((repo/'src/rvl_systems/verifier_rpc.py').read_bytes()).hexdigest()
    result['trainer_source_sha256']=hashlib.sha256((repo/'src/rvl_systems/hf_trainer.py').read_bytes()).hexdigest()
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('rewards','loss_gradient','gradient_error_to_intended','theta_after_sgd','agrees_with_actual_rewards')}))
if __name__=='__main__':main()
