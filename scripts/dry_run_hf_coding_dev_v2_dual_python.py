"""Validate dual-Python readiness and print a non-executing HF study launch plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MLX_PYTHON = Path('/Users/user/.cache/uv/archive-v0/xUAS2gRzA5ZEPhE5/bin/python')
GRADER_PYTHON = Path('/opt/homebrew/Caskroom/miniforge/base/bin/python')
MODEL = Path('/Users/user/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b')
IMAGE = 'vare/hf-coding-dev-arm64@sha256:0363cbea4db3e3739f09eaa6d402ed2328d1bf768bfead190af9ee706e663d97'
SOURCE_MIRRORS = {
    'huggingface/accelerate': Path('/Users/user/.cache/vare/source-mirrors/accelerate'),
    'huggingface/trl': Path('/Users/user/.cache/vare/source-mirrors/trl'),
}
V1_PROTOCOL_PATH = ROOT / 'experiments/hf_coding_dev_inference_v1/protocol-v1.json'
V1_LOCK_PATH = ROOT / 'experiments/hf_coding_dev_inference_v1/protocol-v1.lock.json'
PROTOCOL_PATH = ROOT / 'experiments/hf_coding_dev_inference_v1/protocol-v2.json'
LOCK_PATH = ROOT / 'experiments/hf_coding_dev_inference_v1/protocol-v2.lock.json'
HOST_PREFLIGHT = ROOT / 'scripts/preflight_hf_coding_dev_v1_host_grader.py'
RESOURCE_PREFLIGHT = ROOT / 'scripts/audit_coding_agent_dev_resource_gate_v1.py'
V1_PROTOCOL_SHA256 = '03147165d73d75568506b52ec59c5701db084bf60a01d2ea5d52af969d2fe1bd'
V1_LOCK_SHA256 = '898a13bf57c48acdc53cdfe2534f81079dde7f80eebb30852c0baf3affc766ce'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _run(argv: list[str], env: dict[str, str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


def verify_locked_inputs(protocol: dict[str, Any], lock: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    protocol_hash = sha256(PROTOCOL_PATH)
    checks.append({'name': 'inference_protocol_lock', 'passed': protocol_hash == lock.get('protocol_sha256'),
                   'actual_sha256': protocol_hash, 'expected_sha256': lock.get('protocol_sha256')})
    checks.extend([
        {'name': 'preserved_v1_protocol', 'passed': sha256(V1_PROTOCOL_PATH) == V1_PROTOCOL_SHA256,
         'actual_sha256': sha256(V1_PROTOCOL_PATH), 'expected_sha256': V1_PROTOCOL_SHA256},
        {'name': 'preserved_v1_lock', 'passed': sha256(V1_LOCK_PATH) == V1_LOCK_SHA256,
         'actual_sha256': sha256(V1_LOCK_PATH), 'expected_sha256': V1_LOCK_SHA256},
        {'name': 'v2_protocol_identity', 'passed': protocol.get('protocol_version') == 2
         and protocol.get('protocol_id') == lock.get('protocol_id')},
        {'name': 'v1_provenance_in_v2_lock', 'passed': lock.get('preserved_v1_protocol_sha256') == V1_PROTOCOL_SHA256
         and lock.get('preserved_v1_lock_sha256') == V1_LOCK_SHA256},
    ])
    for relative, expected in lock['implementation_sha256'].items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        checks.append({'name': f'locked_implementation:{relative}', 'passed': actual == expected,
                       'actual_sha256': actual, 'expected_sha256': expected})
    for task_id in protocol['task_cohort']['task_ids']:
        spec = protocol['task_cohort']['tasks'][task_id]
        task_path = ROOT / spec['descriptor_path']
        manifest_path = ROOT / f"benchmarks/hf_coding_dev_v1/{task_id}/baseline-source-manifest-v1.json"
        public_path = ROOT / f'benchmarks/hf_coding_dev_v1/{task_id}/test_public.py'
        api_path = ROOT / f'benchmarks/hf_coding_dev_v1/{task_id}/evaluator/task_api.py'
        values = [
            (f'{task_id}:descriptor', task_path, spec['descriptor_sha256']),
            (f'{task_id}:manifest', manifest_path, spec['baseline_manifest_sha256']),
            (f'{task_id}:public_tests', public_path, lock['public_test_sha256'][task_id]),
            (f'{task_id}:task_api', api_path, lock['task_api_sha256'][task_id]),
        ]
        for name, path, expected in values:
            actual = sha256(path) if path.is_file() else None
            checks.append({'name': name, 'passed': actual == expected, 'actual_sha256': actual,
                           'expected_sha256': expected})
    return checks


def build_dual_python_plan(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    launch = protocol['launch']
    plan = []
    for task in launch['study_order']:
        task_id = task['task_id']
        for rollout in task['rollouts']:
            seed = rollout.rsplit('-', 1)[1]
            agent = [value.replace('{TASK_ID}', task_id).replace('{ROLLOUT_DIR}', '{NEW_EVIDENCE_ROOT}/' + task_id + '/' + rollout)
                     .replace('{WORKSPACE}', '{NEW_WORKSPACE_ROOT}/' + task_id + '/' + rollout)
                     .replace('{SEED}', seed).replace('{SHARED_DEADLINE_EPOCH}', '{SHARED_DEADLINE_EPOCH}')
                     for value in launch['agent_argv_template']]
            agent = [value for value in agent if value != '{SAMPLING_FLAG}']
            agent.append('--do-sample')
            grader = [value.replace('{TASK_ID}', task_id).replace('{ROLLOUT_DIR}', '{NEW_EVIDENCE_ROOT}/' + task_id + '/' + rollout)
                      .replace('{WORKSPACE}', '{NEW_WORKSPACE_ROOT}/' + task_id + '/' + rollout)
                      for value in launch['grader_argv_template']]
            prepare = [value.replace('{TASK_ID}', task_id).replace('{WORKSPACE}', '{NEW_WORKSPACE_ROOT}/' + task_id + '/' + rollout)
                       for value in launch['prepare_argv_template']]
            plan.append({
                'task_id': task_id, 'rollout': rollout, 'seed': int(seed),
                'prepare': {'env': {'DOCKER_CONTEXT': 'colima', 'PYTHONPATH': 'src'}, 'argv': prepare},
                'agent': {'interpreter': str(MLX_PYTHON), 'env': {'DOCKER_CONTEXT': 'colima', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONPATH': 'src'}, 'argv': agent},
                'grader': {'interpreter': str(GRADER_PYTHON), 'env': {'DOCKER_CONTEXT': 'colima', 'PYTHONPATH': 'src'}, 'argv': grader},
            })
    return plan


def plan_if_ready(protocol: dict[str, Any], checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not all(check.get('passed') is True for check in checks):
        return []
    return build_dual_python_plan(protocol)


def validate(*, dry_run: bool) -> dict[str, Any]:
    if not dry_run:
        raise RuntimeError('this validation launcher only supports --dry-run; it never executes a model')
    protocol = json.loads(PROTOCOL_PATH.read_text())
    lock = json.loads(LOCK_PATH.read_text())
    env = os.environ.copy()
    env.update({'DOCKER_CONTEXT': 'colima', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONPATH': 'src'})
    checks = verify_locked_inputs(protocol, lock)

    # Host-side grader dependency check runs before any MLX or model-related check.
    grader_preflight = _run([str(GRADER_PYTHON), str(HOST_PREFLIGHT)], env)
    try:
        grader_result = json.loads(grader_preflight.stdout.strip())
    except json.JSONDecodeError:
        grader_result = {'decision': 'NO-GO', 'stdout': grader_preflight.stdout, 'stderr': grader_preflight.stderr}
    checks.append({'name': 'host_grader_preflight', 'passed': grader_preflight.returncode == 0 and grader_result.get('decision') == 'GO',
                   'interpreter': str(GRADER_PYTHON), 'returncode': grader_preflight.returncode, 'result': grader_result})

    mlx_code = (
        'import importlib.metadata as m,json,platform,sys,mlx.core as mx; '
        'print(json.dumps({"python":platform.python_version(),"executable":sys.executable,'
        '"mlx":m.version("mlx"),"mlx_lm":m.version("mlx-lm"),'
        '"metal":mx.metal.is_available(),"device":str(mx.default_device())}))'
    )
    mlx_run = _run([str(MLX_PYTHON), '-c', mlx_code], env)
    try:
        mlx_result = json.loads(mlx_run.stdout.strip())
    except json.JSONDecodeError:
        mlx_result = {'stdout': mlx_run.stdout, 'stderr': mlx_run.stderr}
    mlx_ok = (mlx_run.returncode == 0 and mlx_result.get('python') == '3.12.12'
              and mlx_result.get('mlx') == '0.32.2' and mlx_result.get('mlx_lm') == '0.32.2'
              and mlx_result.get('metal') is True and mlx_result.get('device') == 'Device(gpu, 0)')
    checks.append({'name': 'mlx_inference_runtime', 'passed': mlx_ok, 'interpreter': str(MLX_PYTHON), 'result': mlx_result})
    checks.append({'name': 'grader_python_runtime', 'passed': grader_result.get('python') == '3.12.12'
                   and grader_result.get('torch') == '2.9.1'
                   and grader_result.get('python_executable') == str(GRADER_PYTHON),
                   'interpreter': str(GRADER_PYTHON), 'result': grader_result})

    # Run the existing resource/hash gate with the explicitly pinned MLX Python.
    # It hashes the pinned checkpoint but does not deserialize or load model weights.
    resource_result: dict[str, Any]
    try:
        with tempfile.TemporaryDirectory(prefix='vare-hf-coding-v2-preflight-') as temporary:
            resource_output = Path(temporary) / 'resource-preflight.json'
            resource_run = _run([
                str(MLX_PYTHON), str(RESOURCE_PREFLIGHT), '--model-snapshot', str(MODEL),
                '--output', str(resource_output),
            ], env, timeout=180)
            resource_result = json.loads(resource_run.stdout.strip()) if resource_run.stdout.strip() else {}
            resource_ok = resource_run.returncode == 0 and resource_result.get('decision') == 'GO'
            checks.append({'name': 'mlx_resource_and_model_hash_preflight', 'passed': resource_ok,
                           'interpreter': str(MLX_PYTHON), 'returncode': resource_run.returncode,
                           'result': resource_result, 'stderr': resource_run.stderr.strip(),
                           'weights_loaded': False})
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, RuntimeError) as exc:
        resource_result = {'decision': 'NO-GO', 'error': f'{type(exc).__name__}: {exc}'}
        checks.append({'name': 'mlx_resource_and_model_hash_preflight', 'passed': False,
                       'interpreter': str(MLX_PYTHON), 'result': resource_result, 'weights_loaded': False})

    colima = _run(['colima', 'status'], env)
    checks.append({'name': 'colima_running', 'passed': colima.returncode == 0,
                   'returncode': colima.returncode, 'stdout': colima.stdout.strip(), 'stderr': colima.stderr.strip()})
    context = _run(['docker', 'context', 'show'], env)
    checks.append({'name': 'docker_context_colima', 'passed': context.returncode == 0 and context.stdout.strip() == 'colima',
                   'returncode': context.returncode, 'stdout': context.stdout.strip()})
    image = _run(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Os}}|{{.Architecture}}|{{json .RepoDigests}}'], env)
    checks.append({'name': 'pinned_grader_image', 'passed': image.returncode == 0 and IMAGE in image.stdout and 'linux|arm64' in image.stdout,
                   'returncode': image.returncode, 'stdout': image.stdout.strip(), 'stderr': image.stderr.strip(), 'expected': IMAGE})

    for task in protocol['launch']['study_order']:
        task_id = task['task_id']
        descriptor = json.loads((ROOT / protocol['task_cohort']['tasks'][task_id]['descriptor_path']).read_text())
        repo = SOURCE_MIRRORS[descriptor['source']['repository']]
        rev = descriptor['source']['base_revision']
        result = _run(['git', '-C', str(repo), 'cat-file', '-e', f'{rev}^{{commit}}'], env)
        checks.append({'name': f'source_mirror:{task_id}', 'passed': result.returncode == 0,
                       'mirror': str(repo), 'commit': rev, 'returncode': result.returncode})

    model_ok = MODEL.is_dir() and MODEL.name == protocol['model']['revision'] and (MODEL / 'model.safetensors').is_file()
    checks.append({'name': 'pinned_model_snapshot_present_without_loading', 'passed': model_ok,
                   'revision': MODEL.name if MODEL.is_dir() else None,
                   'weight_bytes': (MODEL / 'model.safetensors').stat().st_size if model_ok else None,
                   'weights_loaded': False})
    all_passed = all(check.get('passed') is True for check in checks)
    explicit_interpreters = (
        protocol['launch']['agent_argv_template'][0] == str(MLX_PYTHON)
        and protocol['launch']['grader_argv_template'][0] == str(GRADER_PYTHON)
        and protocol['launch']['prepare_argv_template'][0] == str(MLX_PYTHON)
    )
    checks.append({'name': 'protocol_explicit_dual_python', 'passed': explicit_interpreters,
                   'agent_python': protocol['launch']['agent_argv_template'][0],
                   'grader_python': protocol['launch']['grader_argv_template'][0]})
    all_passed = all(check.get('passed') is True for check in checks)
    exposure = protocol['task_cohort']['prior_model_exposure']
    return {
        'schema_version': 1,
        'decision': 'READY' if all_passed else 'BLOCKED',
        'environment_prerequisites_passed': all_passed,
        'blocked_reason': None if all_passed else 'one or more interpreter, resource, source, protocol, or container gates failed',
        'inference_authorized': False,
        'training_started': False,
        'model_weights_loaded': False,
        'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(PROTOCOL_PATH),
        'protocol_lock_sha256': sha256(LOCK_PATH),
        'checks': checks,
        'all_prerequisites_passed': all_passed,
        'future_dual_python_plan': plan_if_ready(protocol, checks),
        'task_exposure': exposure,
        'protocol_note': 'This is a dry-run only. The four tasks were exposed to the model under protocol v1 and any v2 rerun is not fresh-task evidence. Do not resume or repair the protocol-invalid v1 study.',
        'resource_preflight': resource_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', required=True, help='required; no model inference is available in this launcher')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/hf-coding-dev-inference-v2/dual-python-validation-v2/dry-run-v2.json')
    args = parser.parse_args()
    result = validate(dry_run=args.dry_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.is_symlink():
        parser.error(f'refusing to overwrite dry-run evidence: {args.output}')
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'decision': result['decision'], 'all_prerequisites_passed': result['all_prerequisites_passed'],
                      'planned_rollouts': len(result['future_dual_python_plan']), 'model_weights_loaded': False,
                      'output': str(args.output)}, sort_keys=True))
    return 0 if result['decision'] == 'READY' else 2


if __name__ == '__main__':
    raise SystemExit(main())
