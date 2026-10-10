"""Deterministic public-API deployment races; use --repo with a pinned RVL checkout."""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

CASES = ('epoch_same_version', 'epoch_distinct_version', 'active_advance', 'async_control', 'restart')

async def run_case(case):
    from src.rvl_systems.types import Generation
    from src.rvl_systems.verifier import FunctionalVerifier
    from src.rvl_systems.verifier_deployment import VerifierDeploymentCoordinator, publish_verifier_artifact
    from src.rvl_systems.verifier_rpc import TCPVerifierClient, VerifierWorkerServer
    entered, release = asyncio.Event(), asyncio.Event()
    events = []
    def record(event, **data):
        events.append({'index': len(events), 'event': event, **data})
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a = publish_verifier_artifact(root / 'a', version=1, content=b'{"reward":0.1}', suffix='.json')
        b = publish_verifier_artifact(root / 'b', version=1 if case in ('epoch_same_version','restart') else 2, content=b'{"reward":0.9}', suffix='.json')
        def plain_loader(manifest):
            reward = json.loads(Path(manifest.path).read_text())['reward']
            return FunctionalVerifier(lambda _: reward)
        async def loader(manifest):
            if manifest.sha256 == a.sha256:
                record('old_loader_entered', version=manifest.version, sha256=manifest.sha256)
                entered.set()
                await release.wait()
                record('old_loader_released')
            return plain_loader(manifest)
        state = root / 'worker.json'
        server = VerifierWorkerServer(FunctionalVerifier(lambda _: 0.0), worker_id='v', deployment_loader=loader, deployment_state_path=state)
        address = await server.start()
        client = TCPVerifierClient(*address, timeout_s=5)
        old = VerifierDeploymentCoordinator([client], state_path=root / 'coordinator.json')
        task = None
        restored = None
        stale_rejected = False
        activation_error = None
        try:
            if case == 'async_control':
                release.set()
                await old.deploy(a)
                expected = a
                record('control_activated')
            else:
                task = asyncio.create_task(client.prepare_verifier(asdict(a), coordinator_epoch=old.epoch))
                await asyncio.wait_for(entered.wait(), timeout=5)
                current = old if case == 'active_advance' else VerifierDeploymentCoordinator([client], state_path=root / 'coordinator.json')
                await current.prepare(b)
                record('new_prepare_ack', epoch=current.epoch, version=b.version, sha256=b.sha256)
                if case == 'active_advance':
                    await current.activate()
                    record('new_activated_before_old_resume', version=b.version)
                release.set()
                try:
                    ack = await task
                    record('old_prepare_ack', **ack)
                except RuntimeError as exc:
                    stale_rejected = True
                    record('old_prepare_rejected', error=str(exc))
                expected = b
                try:
                    if case == 'active_advance':
                        await client.activate_verifier(a.version, coordinator_epoch=current.epoch)
                    else:
                        await current.activate()
                    record('activation_returned_success')
                except RuntimeError as exc:
                    activation_error = str(exc)
                    record('activation_rejected', error=str(exc))
            health = await client.ping()
            record('health', **health)
            durable = json.loads(state.read_text())
            observed = await client.verify(Generation('p', 'prompt', 'response', 0.0, 1, 0.0, {}), expected_verifier_version=health['verifier_version'])
            record('reward', version=observed.verifier_version, reward=observed.reward)
            expected_reward = 0.1 if case == 'async_control' else 0.9
            checks = {
                'active_version_matches_intended': health['verifier_version'] == expected.version,
                'active_sha_matches_intended': health['active_artifact_sha256'] == expected.sha256,
                'reward_matches_intended': observed.reward == expected_reward,
                'durable_sha_matches_intended': (durable.get('active_manifest') or {}).get('sha256') == expected.sha256,
            }
            if case != 'async_control':
                checks['obsolete_prepare_rejected'] = stale_rejected
                checks['new_activation_succeeds_or_downgrade_rejected'] = (activation_error is not None) if case == 'active_advance' else (activation_error is None)
            if case == 'restart':
                await server.close()
                restored = VerifierWorkerServer(FunctionalVerifier(lambda _: 0.0), worker_id='v', deployment_loader=plain_loader, deployment_state_path=state)
                client2 = TCPVerifierClient(*(await restored.start()))
                health2 = await client2.ping()
                row2 = await client2.verify(Generation('p', 'prompt', 'response', 0.0, 1, 0.0, {}), expected_verifier_version=health2['verifier_version'])
                record('restart_health', **health2)
                record('restart_reward', version=row2.verifier_version, reward=row2.reward)
                checks['restart_preserves_intended_sha_and_reward'] = health2['active_artifact_sha256'] == b.sha256 and row2.reward == 0.9
            return {'case': case, 'expected': {'version': expected.version, 'sha256': expected.sha256, 'reward': expected_reward}, 'checks': checks, 'violations': sum(not value for value in checks.values()), 'events': events}
        finally:
            release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            if restored is not None:
                await restored.close()
            await server.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    source = repo / 'src/rvl_systems/verifier_rpc.py'
    started = time.perf_counter()
    rows = [asyncio.run(asyncio.wait_for(run_case(case), timeout=15)) for case in CASES]
    result = {'repo_head': subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip(), 'worker_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'python': sys.version, 'platform': platform.platform(), 'elapsed_s': time.perf_counter()-started, 'cases': rows, 'violations': sum(row['violations'] for row in rows), 'failed_cases': sum(row['violations'] > 0 for row in rows), 'note': 'Event-controlled source-level TCP reproduction, no model training or GPU.'}
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('worker_source_sha256','elapsed_s','violations','failed_cases')}))

if __name__ == '__main__':
    main()
