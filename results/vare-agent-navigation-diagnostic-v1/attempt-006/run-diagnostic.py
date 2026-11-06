from __future__ import annotations
import hashlib, importlib.metadata, json, os, platform, runpy, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO.parent.parent
OUT = REPO / 'results/vare-agent-navigation-diagnostic-v1/attempt-006'
PROTOCOL_PATH = OUT / 'protocol.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main() -> int:
    sys.path.insert(0, str(REPO / 'src'))
    runner_path = REPO / PROTOCOL['agent']['runner_path']
    assert sha(runner_path) == PROTOCOL['agent']['runner_sha256']
    assert sha(REPO / PROTOCOL['agent']['navigation_module_path']) == PROTOCOL['agent']['navigation_module_sha256']
    ns = runpy.run_path(str(runner_path))
    assert ns['ROOT'] == ROOT
    assert ns['TOOL_ADAPTER_VERSION'] == PROTOCOL['agent']['tool_adapter_version']
    assert sha_bytes(ns['SYSTEM_PROMPT'].encode()) == PROTOCOL['agent']['system_prompt_sha256']
    assert sha_bytes(ns['TOOL_HELP'].encode()) == PROTOCOL['agent']['tool_help_sha256']
    assert sha(REPO / PROTOCOL['gate_boundary']['frozen_gate_path']) == PROTOCOL['gate_boundary']['frozen_gate_sha256']
    gate = json.loads((REPO / PROTOCOL['gate_boundary']['frozen_gate_path']).read_text())
    assert gate['decision']['status'] == 'NO_GO' and gate['decision']['eligible_unique_heldout_tasks'] == 0
    assert PROTOCOL['authorization']['not_authorized'] == [
        'GRPO/SFT training', 'optimizer update', 'checkpoint save or reload',
        'Development evaluation', 'held-out task access or evaluation',
        'using this rollout as independent evaluation evidence'
    ]

    manifest_path = ROOT / PROTOCOL['model']['model_manifest_path']
    model_path = manifest_path.parent / 'model-qwen2.5-coder-7b-4bit'
    assert sha(manifest_path) == PROTOCOL['model']['model_manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    actual_files, tree_sha = ns['hash_tree'](model_path)
    assert actual_files == manifest['files']
    assert tree_sha == PROTOCOL['model']['model_tree_sha256'] == manifest['tree_sha256']
    assert manifest['model_id'] == PROTOCOL['model']['id']
    assert manifest['revision'] == PROTOCOL['model']['revision']
    assert manifest['chat_template_sha256'] == PROTOCOL['model']['chat_template_sha256']
    assert importlib.metadata.version('mlx') == PROTOCOL['model']['mlx_version']
    assert importlib.metadata.version('mlx-lm') == PROTOCOL['model']['mlx_lm_version']
    assert platform.machine() == 'arm64'

    task_def_path = ROOT / 'outputs/vare-rsi-heldout-qualification-v2-20261105/task-definition-pydantic-14115-v3.json'
    prompt_variant_path = REPO / PROTOCOL['task']['task_prompt_variant_path']
    assert sha(task_def_path) == PROTOCOL['task']['task_definition_sha256']
    assert sha(prompt_variant_path) == PROTOCOL['task']['task_prompt_variant_sha256']
    definition = json.loads(task_def_path.read_text())
    assert definition['authorization']['candidate_or_model_inference'] is False
    assert definition['authorization']['heldout_access'] is False
    env_path = ROOT / PROTOCOL['trusted_grader']['environment_lock_path']
    assert sha(env_path) == PROTOCOL['trusted_grader']['environment_lock_sha256']
    env = json.loads(env_path.read_text())
    assert env['image_id'] == PROTOCOL['trusted_grader']['image_id']
    assert env['network'].startswith('none') and PROTOCOL['trusted_grader']['candidate_feedback'] is False

    prompt_variant = json.loads(prompt_variant_path.read_text())
    record = {
        'task_id': PROTOCOL['task']['task_id'],
        'task_family': definition['task_family'],
        'split': PROTOCOL['task']['split'],
        'task_definition_path': str(task_def_path.relative_to(ROOT)),
        'qualification_summary_path': '',
        'candidate_id': 'diagnostic-only-attempt-006',
        'prompt_variant_path': str(prompt_variant_path),
        'prompt_variant_sha256': PROTOCOL['task']['task_prompt_variant_sha256'],
        'task_definition_sha256': PROTOCOL['task']['task_definition_sha256'],
    }
    task = ns['load_task'](record)
    record['prompt_sha256'] = ns['sha256_json'](ns['make_agent_messages'](task))
    task['record'] = record
    assert record['prompt_sha256'] == PROTOCOL['agent']['agent_prompt_sha256']
    assert task['prompt'] == prompt_variant['task_prompt']
    assert task['editable_paths'] == PROTOCOL['task']['editable_paths']
    assert task['test_path'] == PROTOCOL['task']['grader_test_path']
    assert task['test_sha256'] == PROTOCOL['task']['grader_test_sha256']
    assert task['selector'] == PROTOCOL['task']['grader_selector']
    ns['verify_task_workspace'](task)
    source = (task['baseline_root'] / task['editable_paths'][0]).read_text()
    assert 'class MockValSer' in source
    preflight_record = {
        'preflight': 'passed',
        'protocol_sha256': sha(PROTOCOL_PATH),
        'runner_sha256': sha(runner_path),
        'navigation_module_sha256': sha(REPO / PROTOCOL['agent']['navigation_module_path']),
        'regression_test_sha256': sha(REPO / PROTOCOL['agent']['regression_test_path']),
        'regression_test_result': PROTOCOL['agent']['regression_test_result'],
        'model_tree_sha256': tree_sha,
        'model_manifest_sha256': sha(manifest_path),
        'model_path': str(model_path),
        'task_id': record['task_id'],
        'split': record['split'],
        'task_prompt_sha256': record['prompt_sha256'],
        'task_source_sha256': sha(task['baseline_root'] / task['editable_paths'][0]),
        'task_test_sha256': task['test_sha256'],
        'grader_image': env['image_id'],
        'no_go': gate['decision'],
        'heldout_access': False,
        'training_authorized': False,
    }
    preflight_path = OUT / 'preflight.json'
    encoded_preflight = json.dumps(preflight_record, indent=2) + '\n'
    if preflight_path.exists():
        assert preflight_path.read_text() == encoded_preflight
    else:
        preflight_path.write_text(encoded_preflight)
    print(encoded_preflight, end='', flush=True)
    if '--preflight-only' in sys.argv:
        return 0

    from mlx_lm import load
    start = time.monotonic()
    model, tokenizer = load(str(model_path))
    load_seconds = round(time.monotonic() - start, 3)
    start_state = ns['current_memory_state']()
    guard = ns['ResourceGuard'](start_state, free_floor_pct=15.0, swap_growth_allowed_bytes=0)
    run_dir = OUT / 'rollout-001'
    workspace = ROOT / 'work/pydantic-14115-navdiag-v6-candidate'
    assert not (run_dir / 'trajectory.json').exists() and not (run_dir / 'summary.json').exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    result = ns['run_agent'](
        model, tokenizer, task,
        workspace=workspace,
        artifact_dir=run_dir,
        policy_id=f"{PROTOCOL['model']['id']}@{PROTOCOL['model']['revision']}:diagnostic-only",
        phase='diagnostic_only_previous_train_task',
        sample_index=PROTOCOL['rollout']['sample_index'],
        seed=PROTOCOL['rollout']['seed'],
        temperature=PROTOCOL['rollout']['temperature'],
        guard=guard,
    )
    record_out = result['record']
    trajectory_path = run_dir / 'trajectory.json'
    trajectory_sha = sha(trajectory_path)
    grade_path = run_dir / 'trusted-grade'
    grade_path.mkdir(parents=True, exist_ok=False)
    grade = ns['grade_candidate'](task, result['workspace'], grade_path, grader_name='vare-navdiag-14115-v6-qwen25-coder7b')
    trace = record_out['tool_trace']
    token_count = sum(len(row.get('completion_token_ids', [])) for row in trace)
    behavior = [x for row in trace for x in row.get('sampling_behavior_logprobs', [])]
    tool_counts = {}
    for row in trace:
        action = row.get('action') or {}
        tool_counts[action.get('tool', 'invalid')] = tool_counts.get(action.get('tool', 'invalid'), 0) + 1
    edit_tool_calls = sum(
        1 for row in trace
        if (row.get('action') or {}).get('tool') in {'replace_text', 'apply_patch'}
    )
    valid_edits = sum(
        1 for row in trace
        if (row.get('action') or {}).get('tool') in {'replace_text', 'apply_patch'}
        and isinstance(row.get('tool_result'), dict)
        and row['tool_result'].get('applied') is True
    )
    finite_logprobs = sum(isinstance(x, (int, float)) and __import__('math').isfinite(x) for x in behavior)
    summary = {
        'schema': 'vare_agent_navigation_diagnostic_result_v6',
        'protocol_id': PROTOCOL['protocol_id'],
        'protocol_sha256': sha(PROTOCOL_PATH),
        'model_id': PROTOCOL['model']['id'],
        'model_revision': PROTOCOL['model']['revision'],
        'model_tree_sha256': tree_sha,
        'model_manifest_sha256': sha(manifest_path),
        'mlx_version': importlib.metadata.version('mlx'),
        'mlx_lm_version': importlib.metadata.version('mlx-lm'),
        'model_load_seconds': load_seconds,
        'task_id': record_out['task_id'],
        'split': record_out['split'],
        'historical_prompt_exposure': True,
        'evaluation_evidence': False,
        'development_evaluation': False,
        'heldout_access': False,
        'no_go_gate_modified': False,
        'turns': record_out['tool_turns'],
        'generated_tokens': token_count,
        'behavior_logprob_entries': len(behavior),
        'behavior_logprobs_finite': finite_logprobs == len(behavior),
        'tool_counts': tool_counts,
        'edit_tool_calls': edit_tool_calls,
        'valid_edit_calls': valid_edits,
        'changed_files': record_out['changed_files'],
        'submitted': record_out['submitted'],
        'stop_reason': record_out['stop_reason'],
        'trajectory_path': str(trajectory_path.relative_to(REPO)),
        'trajectory_file_sha256': trajectory_sha,
        'candidate_source_hashes': record_out['candidate_source_hashes'],
        'trusted_grade': grade,
        'optimizer_updates': 0,
        'checkpoint_saved': False,
        'checkpoint_reloaded': False,
        'development_evaluation_performed': False,
        'resource_guard_last_state': guard.last_state,
        'wall_seconds': record_out['wall_seconds'],
    }
    ns['atomic_json'](run_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0

try:
    raise SystemExit(main())
except Exception as exc:
    (OUT / 'rollout-001').mkdir(parents=True, exist_ok=True)
    error_path = OUT / 'rollout-001' / 'execution-error.json'
    if error_path.exists():
        error_path = OUT / 'rollout-001' / f'execution-error-{time.time_ns()}.json'
    error_path.write_text(json.dumps({
        'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()
    }, indent=2) + '\n')
    raise
