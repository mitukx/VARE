import asyncio
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, sys.argv[1])
from src.rvl_systems.verifier import FunctionalVerifier
from src.rvl_systems.verifier_deployment import VerifierDeploymentCoordinator, publish_verifier_artifact
from src.rvl_systems.verifier_rpc import TCPVerifierClient, VerifierWorkerServer
from src.rvl_systems.types import Generation

async def run(identical=False):
    entered, release = asyncio.Event(), asyncio.Event()
    events=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        a=publish_verifier_artifact(root/'a',version=1,content=b'{"reward":0.1}')
        b=a if identical else publish_verifier_artifact(root/'b',version=1,content=b'{"reward":0.9}')
        calls=0
        async def loader(m):
            nonlocal calls
            calls+=1
            if calls==1:
                events.append('A loader blocked')
                entered.set()
                await release.wait()
            r=json.loads(Path(m.path).read_text())['reward']
            return FunctionalVerifier(lambda _:r)
        state=root/'worker.json'
        server=VerifierWorkerServer(FunctionalVerifier(lambda _:0),worker_id='v',deployment_loader=loader,deployment_state_path=state)
        client=TCPVerifierClient(*(await server.start()),timeout_s=5)
        coordinator=VerifierDeploymentCoordinator([client],state_path=root/'coordinator.json')
        task=asyncio.create_task(client.prepare_verifier(asdict(a),coordinator_epoch=coordinator.epoch))
        try:
            await asyncio.wait_for(entered.wait(),5)
            await coordinator.prepare(b)
            events.append({'B ACK':b.sha256})
            release.set()
            events.append({'A ACK':await task})
            try:
                await coordinator.activate()
                success=True
                events.append('B coordinator declared activation success')
            except RuntimeError as e:
                success=False
                events.append({'activation_rejected':str(e)})
            health=await client.ping()
            # On rejected activation the bootstrap verifier is still active.
            row=await client.verify(Generation('p','prompt','response',0.,1,0.,{}),expected_verifier_version=health['verifier_version'])
            durable=json.loads(state.read_text())
            wrong_success=success and health['active_artifact_sha256']!=b.sha256
            try:
                await coordinator.deploy(b)
                recovery_error=None
            except RuntimeError as e:
                recovery_error=str(e)
            recovered=await client.ping()
            return {'identical':identical,'events':events,'expected_sha':b.sha256,'activation_success':success,'health':health,'reward':row.reward,'durable':durable,'wrong_artifact_success':wrong_success,'recovery_error':recovery_error,'recovery_sha':recovered['active_artifact_sha256'],'fixed_contract_pass':not wrong_success and recovered['active_artifact_sha256']==b.sha256 and (not identical or success)}
        finally:
            release.set()
            await asyncio.gather(task,return_exceptions=True)
            await server.close()

async def main():
    return [await asyncio.wait_for(run(x),15) for x in [False,True]]
print(json.dumps(asyncio.run(main()),indent=2,sort_keys=True))
