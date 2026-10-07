#!/usr/bin/env python3
"""Offline, CPU-only, bounded tool-use pilot for pinned VARE tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPOSITORY = 'Qwen/Qwen2.5-0.5B-Instruct'
TOOLS = [
    {'type': 'function', 'function': {'name': 'list_files', 'description': 'List the exact source files available in this task workspace.', 'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'read_file', 'description': 'Read one UTF-8 task source file. Provide its repository-relative path; a workspace-prefixed path is mapped only if it ends in one exact allowlisted source path.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'search_text', 'description': 'Search a literal string in the allowed task source files.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'apply_patch', 'description': 'Apply a small unified diff to existing task source files. Keep the patch minimal; paths must be allowlisted.', 'parameters': {'type': 'object', 'properties': {'patch': {'type': 'string'}}, 'required': ['patch'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'check_syntax', 'description': 'Parse one allowed Python file for syntax errors without executing it.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'finish', 'description': 'Finish the attempt when you believe the task is complete. The evaluator will judge the workspace.', 'parameters': {'type': 'object', 'properties': {'summary': {'type': 'string'}}, 'required': ['summary'], 'additionalProperties': False}}},
]
TOOL_NAMES = {item['function']['name'] for item in TOOLS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_tree(root: Path, files: list[str]) -> dict[str, str]:
    return {name: sha256(root / name) for name in files if (root / name).is_file()}


def run_git(root: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(root), *args], check=True, text=True,
                          capture_output=True).stdout.strip()


def resolve_allowed_path(raw_path: str, allowed: set[str]) -> str | None:
    """Map a model-supplied path to one exact allowlisted relative file."""
    normalized = raw_path.replace('\\', '/').lstrip('./')
    matches = [path for path in allowed if normalized == path or normalized.endswith('/' + path)]
    return matches[0] if len(matches) == 1 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True, help='Local model snapshot directory; no network download is attempted')
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instruction', required=True,
                        help='Task-specific user message; preserve the exact text in the run record')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-turns', type=int, default=16)
    parser.add_argument('--max-no-tool-retries', type=int, default=2)
    parser.add_argument('--max-new-tokens-per-turn', type=int, default=256)
    parser.add_argument('--max-total-new-tokens', type=int, default=3072)
    parser.add_argument('--wall-seconds', type=int, default=900)
    args = parser.parse_args()

    model_dir = args.model_path.resolve()
    if not model_dir.is_dir():
        parser.error(f'local model snapshot directory does not exist: {model_dir}')
    workspace = args.workspace.resolve()
    task_root = args.task_root.resolve()
    result_path = args.output.resolve()
    task = json.loads((task_root / 'task.json').read_text())
    allowed = set(task['source']['files'])
    missing = sorted(name for name in allowed if not (workspace / name).is_file())
    if missing:
        parser.error(f'workspace missing allowed source files: {missing}')
    if not os.environ.get('HF_HUB_OFFLINE') == '1' or not os.environ.get('TRANSFORMERS_OFFLINE') == '1':
        parser.error('HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required')

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32,
    ).to('cpu').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if not all(parameter.device.type == 'cpu' for parameter in model.parameters()):
        raise RuntimeError('model parameter escaped CPU')

    task_text = (task_root / 'TASK.md').read_text(encoding='utf-8')
    system = (
        'You are an offline code repair agent. Solve only the task below. The workspace contains only the task source files. '
        'Read the source path named in the user instruction, then use tools to inspect and edit. Use apply_patch for small source diffs; never rewrite a whole file. '
        'Do not invent paths. The workspace is already prepared; never ask for a workspace path, more information, or permission. '
        'The external grader is hidden. Do not describe a fix without applying it. Call finish only after changing a source file. '
        'You have a limited tool budget; each assistant turn before finish must call a tool. '
        'TASK:\n' + task_text.split('## Reproduce', 1)[0].rstrip()
    )
    messages: list[dict[str, str]] = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': args.instruction},
    ]
    initial_messages = [dict(message) for message in messages]
    transcript: list[dict[str, Any]] = []
    before = digest_tree(workspace, sorted(allowed))
    token_total = 0
    stop_reason = 'turn_budget_exhausted'
    final_summary = ''
    no_tool_retries = 0
    pattern = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)

    for turn in range(args.max_turns):
        elapsed = time.perf_counter() - start
        if elapsed >= args.wall_seconds:
            stop_reason = 'wall_time_exhausted'
            break
        prompt = tokenizer.apply_chat_template(
            messages, tools=TOOLS, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(prompt, return_tensors='pt')
        remaining = args.max_total_new_tokens - token_total
        if remaining <= 0:
            stop_reason = 'token_budget_exhausted'
            break
        limit = min(args.max_new_tokens_per_turn, remaining)
        with torch.inference_mode():
            output = model.generate(
                input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'],
                do_sample=True, temperature=0.2, top_p=0.95, top_k=20,
                max_new_tokens=limit, pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = output[0][encoded['input_ids'].shape[1]:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=False)
        token_total += int(new_ids.shape[0])
        entry: dict[str, Any] = {'turn': turn + 1, 'model_output': raw, 'tool_calls': []}
        calls = list(pattern.finditer(raw))
        if not calls:
            messages.append({'role': 'assistant', 'content': raw})
            transcript.append(entry)
            current_hashes = digest_tree(workspace, sorted(allowed))
            if current_hashes == before and no_tool_retries < args.max_no_tool_retries:
                no_tool_retries += 1
                messages.append({'role': 'user', 'content':
                    'Your previous response did not call a tool and no source file changed. Continue the task now. '
                    'Call read_file on the exact path named in my first instruction. Do not ask a question.'})
                entry['automatic_reminder'] = no_tool_retries
                continue
            stop_reason = 'no_tool_call_after_edit' if current_hashes != before else 'no_tool_call_retries_exhausted'
            break
        parsed_calls = []
        for match in calls:
            try:
                call = json.loads(match.group(1))
                name = call.get('name')
                arguments = call.get('arguments', {})
                if name not in TOOL_NAMES or not isinstance(arguments, dict):
                    raise ValueError('unsupported tool call')
                parsed_calls.append((match, name, arguments))
            except Exception as exc:
                entry['tool_calls'].append({'error': f'{type(exc).__name__}: {exc}', 'raw_call': match.group(1)})
        if parsed_calls:
            messages.append({'role': 'assistant', 'tool_calls': [
                {'type': 'function', 'function': {'name': name, 'arguments': arguments}}
                for _, name, arguments in parsed_calls
            ]})
        else:
            messages.append({'role': 'assistant', 'content': raw})
        for match, name, arguments in parsed_calls:
            try:
                if name == 'list_files':
                    result: Any = sorted(allowed)
                elif name == 'read_file':
                    raw_path = arguments.get('path', '')
                    path = resolve_allowed_path(raw_path, allowed)
                    if path is None:
                        result = {'error': 'path is not allowed', 'allowed_paths': sorted(allowed)}
                    else:
                        result = {'resolved_path': path,
                                  'content': (workspace / path).read_text(encoding='utf-8')}
                elif name == 'search_text':
                    query = arguments.get('query', '')
                    if not query:
                        result = {'error': 'query must not be empty'}
                    else:
                        matches = []
                        for path in sorted(allowed):
                            for line_no, line in enumerate((workspace / path).read_text(encoding='utf-8').splitlines(), 1):
                                if query in line:
                                    matches.append({'path': path, 'line': line_no, 'text': line[:500]})
                        result = matches[:80]
                elif name == 'apply_patch':
                    patch_text = arguments.get('patch', '')
                    paths = []
                    for line in patch_text.splitlines():
                        if line.startswith(('--- a/', '+++ b/')):
                            candidate_path = line[6:]
                            if candidate_path != '/dev/null':
                                paths.append(candidate_path)
                    forbidden_features = ('GIT binary patch', 'new file mode', 'deleted file mode',
                                          'rename from ', 'rename to ', 'Subproject commit')
                    if not paths or any(path not in allowed for path in paths) or len(set(paths)) != len(paths) // 2:
                        result = {'error': 'patch paths must be existing allowlisted source files'}
                    elif any(feature in patch_text for feature in forbidden_features):
                        result = {'error': 'binary, create, delete, rename and submodule patches are disabled'}
                    else:
                        checked = subprocess.run(['git', 'apply', '--check', '--'], cwd=workspace,
                                                 input=patch_text, text=True, capture_output=True)
                        if checked.returncode != 0:
                            result = {'error': 'patch failed validation', 'detail': checked.stderr[:1000]}
                        else:
                            applied = subprocess.run(['git', 'apply', '--'], cwd=workspace,
                                                     input=patch_text, text=True, capture_output=True)
                            result = {'applied': applied.returncode == 0,
                                      'paths': sorted(set(paths)),
                                      'error': applied.stderr[:1000] if applied.returncode else None}
                elif name == 'check_syntax':
                    raw_path = arguments.get('path', '')
                    path = resolve_allowed_path(raw_path, allowed)
                    if path is None:
                        result = {'error': 'path is not allowed', 'allowed_paths': sorted(allowed)}
                    else:
                        import ast
                        try:
                            ast.parse((workspace / path).read_text(encoding='utf-8'), filename=path)
                            result = {'syntax_valid': True, 'path': path}
                        except SyntaxError as exc:
                            result = {'syntax_valid': False, 'path': path, 'error': str(exc)}
                else:
                    if digest_tree(workspace, sorted(allowed)) == before:
                        result = {'finished': False, 'error': 'source is unchanged; inspect and edit before finishing'}
                    else:
                        final_summary = str(arguments.get('summary', ''))
                        result = {'finished': True}
                        stop_reason = 'agent_finished'
                entry['tool_calls'].append({'name': name, 'arguments': arguments, 'result': result})
                messages.append({'role': 'tool', 'name': name, 'content': json.dumps(result, ensure_ascii=False)})
                if name == 'finish':
                    break
            except Exception as exc:
                entry['tool_calls'].append({'error': f'{type(exc).__name__}: {exc}', 'raw_call': match.group(1)})
                messages.append({'role': 'tool', 'name': 'invalid_tool_call', 'content': json.dumps({'error': str(exc)})})
        transcript.append(entry)
        if stop_reason == 'agent_finished':
            break

    elapsed = time.perf_counter() - start
    after = digest_tree(workspace, sorted(allowed))
    diff = run_git(workspace, 'diff', '--binary', task['source']['base_revision'], '--', *sorted(allowed))
    result = {
        'schema_version': 1, 'task_id': task['task_id'], 'seed': args.seed,
        'task_revisions': {'base': task['source']['base_revision'],
                           'calibration': task['source']['calibration_revision']},
        'task_input_hashes': {
            'TASK.md': sha256(task_root / 'TASK.md'),
            'task.json': sha256(task_root / 'task.json'),
            'protocol.lock.json': sha256(task_root / 'protocol.lock.json'),
            'grader.py': sha256(task_root / 'evaluator/grade.py'),
        },
        'initial_messages': initial_messages,
        'model_repository': MODEL_REPOSITORY, 'model_snapshot': model_dir.name,
        'model_files': {path.name: sha256(path) for path in sorted(model_dir.iterdir()) if path.is_file()},
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'transformers': __import__('transformers').__version__,
                    'platform': platform.platform(), 'device': 'cpu', 'threads': torch.get_num_threads(),
                    'cuda_available': torch.cuda.is_available(), 'mps_available_unused': torch.backends.mps.is_available()},
        'budgets': {'max_turns': args.max_turns, 'per_turn_tokens': args.max_new_tokens_per_turn,
                    'total_new_tokens': args.max_total_new_tokens, 'wall_seconds': args.wall_seconds,
                    'max_no_tool_retries': args.max_no_tool_retries},
        'stop_reason': stop_reason, 'agent_summary': final_summary,
        'elapsed_seconds': round(elapsed, 3), 'new_tokens': token_total,
        'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'workspace_revision': run_git(workspace, 'rev-parse', 'HEAD'),
        'candidate_diff': diff,
        'candidate_diff_sha256': hashlib.sha256(diff.encode()).hexdigest(),
        'source_hashes_before': before, 'source_hashes_after': after,
        'tool_turns': len(transcript), 'no_tool_retries': no_tool_retries, 'transcript': transcript,
        'offline_environment_required': True,
        'wall_budget_checked_between_generations': True,
        'claim_limit': 'One small cached model on one historical task; not frontier-agent performance or novel fix discovery.',
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: result[key] for key in ['task_id', 'seed', 'stop_reason', 'elapsed_seconds', 'new_tokens', 'tool_turns', 'candidate_diff_sha256']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
