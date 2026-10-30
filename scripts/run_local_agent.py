#!/usr/bin/env python3
"""Offline, bounded tool-use runner for pinned VARE coding tasks."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from pinned_workspace_git import pinned_git, read_regular_source  # noqa: E402
from vare.agent_edit_format import ParseFailure, apply_parsed_edit, parse_submission, require_edit_format  # noqa: E402
from vare.agent_task_contract import load_agent_task  # noqa: E402

MODEL_REPOSITORY = 'Qwen/Qwen2.5-0.5B-Instruct'
DEFAULT_READ_LINES = 180
MAX_READ_LINES = 200
TOOLS = [
    {'type': 'function', 'function': {'name': 'list_files', 'description': 'List the exact source files available in this task workspace.', 'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'read_file', 'description': 'Read at most 200 lines from one UTF-8 task source file. By default, returns lines 1-180. For large files, pass inclusive start_line and end_line; the result reports the returned range, total_lines, and has_more.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'start_line': {'type': 'integer', 'minimum': 1}, 'end_line': {'type': 'integer', 'minimum': 1}}, 'required': ['path'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'search_text', 'description': 'Search a literal string in the allowed task source files.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'apply_patch', 'description': 'Apply a small unified diff to existing task source files. Keep the patch minimal; paths must be allowlisted.', 'parameters': {'type': 'object', 'properties': {'patch': {'type': 'string'}}, 'required': ['patch'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'check_syntax', 'description': 'Parse one allowed Python file for syntax errors without executing it.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'finish', 'description': 'Finish the attempt when you believe the task is complete. The evaluator will judge the workspace.', 'parameters': {'type': 'object', 'properties': {'summary': {'type': 'string'}}, 'required': ['summary'], 'additionalProperties': False}}},
]
PUBLIC_TEST_TOOL = {
    'type': 'function',
    'function': {
        'name': 'run_public_tests',
        'description': 'Run the unchanged task-provided test_public.py inside the isolated, network-disabled candidate container. You cannot edit that test file.',
        'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
    },
}
WRITE_FILE_TOOL = {
    'type': 'function',
    'function': {
        'name': 'write_file',
        'description': 'Replace one allowlisted source file with its complete new UTF-8 text. Do not send a unified diff.',
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {'type': 'string'},
                'content': {'type': 'string'},
            },
            'required': ['path', 'content'],
            'additionalProperties': False,
        },
    },
}


def tools_for(edit_format: str, *, public_tests: bool = False) -> list[dict[str, Any]]:
    require_edit_format(edit_format)
    if edit_format == 'unified_diff':
        tools = list(TOOLS)
    else:
        tools = [WRITE_FILE_TOOL if item['function']['name'] == 'apply_patch' else item for item in TOOLS]
    if public_tests:
        tools.insert(-1, PUBLIC_TEST_TOOL)
    return tools


def generate_turn(backend: str, model: Any, tokenizer: Any, messages: list[dict[str, Any]],
                  tools: list[dict[str, Any]], *, limit: int, args: argparse.Namespace) -> tuple[str, int, int]:
    """Generate one bounded assistant response using a locally loaded backend."""
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if prompt_tokens >= args.max_context_tokens:
        raise ValueError('context_budget_exhausted')
    limit = min(limit, args.max_context_tokens - prompt_tokens)
    if backend == 'mlx':
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(
            temp=0.0 if not args.do_sample else args.temperature,
            top_p=0.0 if not args.do_sample else args.top_p,
            top_k=0 if not args.do_sample else args.top_k,
        )
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=limit,
                       sampler=sampler, verbose=False)
        return raw, len(tokenizer.encode(raw, add_special_tokens=False)), prompt_tokens

    import torch
    encoded = tokenizer(prompt, return_tensors='pt')
    with torch.inference_mode():
        output = model.generate(
            input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'],
            do_sample=args.do_sample,
            **({} if not args.do_sample else {
                'temperature': args.temperature, 'top_p': args.top_p, 'top_k': args.top_k,
            }),
            max_new_tokens=limit, pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output[0][encoded['input_ids'].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=False), int(new_ids.shape[0]), prompt_tokens


def run_public_tests(workspace: Path, image: str, expected_test_hash: str) -> dict[str, Any]:
    """Run only the immutable public smoke file in VARE's candidate sandbox."""
    test_file = workspace / 'test_public.py'
    if not test_file.is_file() or sha256(test_file) != expected_test_hash:
        return {'passed': False, 'error': 'public test file is missing or changed'}
    from vare.container_isolation import DockerIsolationRunner, IsolationLimits

    limits = IsolationLimits(image=image, timeout_s=45, memory_mb=512, cpus=1.0,
                             pids_limit=32, max_output_bytes=20_000)
    started = time.perf_counter()
    try:
        result = asyncio.run(DockerIsolationRunner().run_candidate(
            limits, allowed_workspace_root=workspace.parent, workspace=workspace,
            command=['python', '/workspace/test_public.py'],
        ))
    except Exception as exc:
        return {'passed': False, 'error': f'{type(exc).__name__}: {exc}',
                'elapsed_seconds': round(time.perf_counter() - started, 3)}
    return {'passed': result.returncode == 0, 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr,
            'elapsed_seconds': round(time.perf_counter() - started, 3),
            'isolation': 'docker-network-none-read-only-root-only-workspace-mount'}


def refresh_public_tests_after_edit(
    edit_result: dict[str, Any], workspace: Path, image: str | None,
    expected_test_hash: str | None,
) -> dict[str, Any] | None:
    """Return a fresh smoke-test result after an applied source edit."""
    if edit_result.get('applied') is not True or not image or not expected_test_hash:
        return None
    return run_public_tests(workspace, image, expected_test_hash)


def attempt_configured_edit(workspace: Path, text: str, edit_format: str, allowed: set[str]) -> dict[str, Any]:
    """Parse and apply one edit. A parse failure is not a task score."""
    parsed = parse_submission(text, edit_format=edit_format)
    if isinstance(parsed, ParseFailure):
        reason = parsed.reason if parsed.reason != 'no_edit' else (
            'malformed_diff' if edit_format == 'unified_diff' else 'malformed_file'
        )
        return {'applied': False, 'format_parse_failure': reason}
    applied = apply_parsed_edit(workspace, parsed, allowed)
    if not applied.applied:
        return {
            'applied': False,
            'format_parse_failure': None,
            'error': applied.error or 'edit was not applied',
            'paths': list(applied.paths),
        }
    return {'applied': True, 'paths': list(applied.paths), 'format_parse_failure': None}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_tree(root: Path, files: list[str]) -> dict[str, str]:
    return {
        name: hashlib.sha256(read_regular_source(root, name)).hexdigest()
        for name in sorted(files)
    }


def snapshot_allowed_sources(root: Path, files: list[str]) -> dict[str, bytes]:
    """Retain the exact candidate-visible bytes before model edits begin."""
    return {name: read_regular_source(root, name) for name in sorted(files)}


def run_git(root: Path, *args: str) -> str:
    return pinned_git(root, *args).decode("utf-8").strip()


def candidate_source_diff(
    root: Path, files: list[str], before: dict[str, bytes], base_revision: str,
) -> str:
    """Export a canonical Git patch from the pinned candidate workspace."""
    ordered = sorted(files)
    if set(before) != set(ordered):
        raise ValueError('invocation source snapshot does not match the allowlist')
    if re.fullmatch(r"[0-9a-f]{40}", base_revision) is None:
        raise ValueError("candidate base revision is not a full Git commit SHA")
    if run_git(root, 'rev-parse', 'HEAD') != base_revision:
        raise ValueError('candidate workspace HEAD changed during model invocation')
    for name in ordered:
        read_regular_source(root, name)
        original = pinned_git(root, 'cat-file', 'blob', f'{base_revision}:{name}')
        if before[name] != original:
            raise ValueError(f'invocation source snapshot differs from pinned Git blob: {name}')
    with tempfile.TemporaryDirectory(prefix="vare-pinned-index-") as temp_dir:
        index_file = Path(temp_dir) / "index"
        pinned_git(root, "read-tree", base_revision, index_file=index_file)
        status = pinned_git(
            root, "status", "--porcelain=v1", "--untracked-files=all",
        ).decode("utf-8")
        allowed_statuses = {f" M {name}" for name in ordered}
        if any(line not in allowed_statuses for line in status.splitlines() if line):
            raise ValueError("candidate workspace contains staged or non-allowlisted changes")
        diff_bytes = pinned_git(
            root, "diff", "--no-ext-diff", "--no-textconv", "--no-renames",
            "--binary", "--", *ordered, index_file=index_file,
            max_output_bytes=2 * 1024 * 1024,
        )
    diff = diff_bytes.decode('utf-8').strip()
    if len(diff.encode('utf-8')) > 2 * 1024 * 1024:
        raise ValueError('candidate diff exceeds the evidence parser limit')
    return diff


def resolve_allowed_path(raw_path: str, allowed: set[str]) -> str | None:
    """Map a model-supplied path to one exact allowlisted relative file."""
    normalized = raw_path.replace('\\', '/').lstrip('./')
    matches = [path for path in allowed if normalized == path or normalized.endswith('/' + path)]
    return matches[0] if len(matches) == 1 else None


def bounded_source_page(
    text: str, *, start_line: int = 1, end_line: int | None = None,
) -> dict[str, Any]:
    """Return a bounded inclusive line range and enough metadata to page safely."""
    if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
        raise ValueError('start_line must be a positive integer')
    if end_line is not None and (
        isinstance(end_line, bool) or not isinstance(end_line, int) or end_line < 1
    ):
        raise ValueError('end_line must be a positive integer')
    if end_line is not None and end_line < start_line:
        raise ValueError('end_line must be greater than or equal to start_line')

    lines = text.splitlines(keepends=True)
    total_lines = len(lines)
    if start_line > max(total_lines, 1):
        raise ValueError(f'start_line exceeds the source length ({total_lines} lines)')
    requested_end = end_line if end_line is not None else start_line + DEFAULT_READ_LINES - 1
    returned_end = min(requested_end, start_line + MAX_READ_LINES - 1, total_lines)
    content = ''.join(lines[start_line - 1:returned_end])
    return {
        'content': content,
        'start_line': start_line,
        'end_line': returned_end,
        'total_lines': total_lines,
        'has_more': returned_end < total_lines,
    }


def no_tool_retry_reminder(raw: str, *, edit_format: str, allowed_paths: list[str],
                           token_limit: int) -> str:
    """Give the agent actionable feedback when a bounded response has no tool call."""
    if '<tool_call>' in raw and '</tool_call>' not in raw:
        path_list = ', '.join(allowed_paths)
        if edit_format == 'whole_file':
            return (
                f'Your previous tool call was cut off at the {token_limit}-token response limit. '
                'Do not reread the source. Emit one complete write_file tool call now, using the exact '
                f'allowed path ({path_list}) and the complete updated source file. Keep it concise enough '
                'to end with </tool_call>.'
            )
        return (
            f'Your previous tool call was cut off at the {token_limit}-token response limit. '
            'Do not reread the source or rewrite the whole file. Emit one complete apply_patch tool call '
            f'with a minimal change and exact allowed paths ({path_list}); end it with </tool_call>.'
        )
    return (
        'Your previous response did not call a tool and no source file changed. Continue the task now. '
        'Call read_file on the exact path named in my first instruction. Do not ask a question.'
    )


def normalize_tool_call_format(raw: str, tool_names: set[str]) -> tuple[str, bool]:
    """Normalize strict JSON tool calls emitted with chat-only model wrappers.

    Qwen2.5-Coder can serialize a requested function call as bare JSON, a
    Markdown JSON block, or a JSON payload wrapped in ChatML tokens even when
    the tokenizer template was given the tool schema. Keep the original model
    output in the trajectory; this normalized copy is used only by the
    existing strict tool-call parser.
    """
    if "<tool_call>" in raw:
        return raw, False

    def encode_if_tool_call(text: str) -> str | None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if (not isinstance(payload, dict) or set(payload) != {"name", "arguments"}
                or payload.get("name") not in tool_names
                or not isinstance(payload.get("arguments"), dict)):
            return None
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    candidate = raw.strip()
    candidate = re.sub(r"^<\|im_start\|>\s*(?:assistant)?\s*", "", candidate)
    if candidate.startswith("```json") or candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    try:
        payload, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return raw, False
    suffix = candidate[end:]
    allowed_suffix = re.fullmatch(
        r"(?:\s|```|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>)*", suffix,
    )
    if allowed_suffix is None or not isinstance(payload, dict):
        return raw, False
    encoded = encode_if_tool_call(json.dumps(payload, ensure_ascii=False))
    if encoded is None:
        return raw, False
    return f"<tool_call>{encoded}</tool_call>", True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True, help='Local model snapshot directory; no network download is attempted')
    parser.add_argument('--model-repository', default=MODEL_REPOSITORY)
    parser.add_argument('--backend', choices=('transformers_cpu', 'mlx'), default='transformers_cpu')
    parser.add_argument('--public-test-image', help='Pinned Docker image for isolated execution of test_public.py')
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instruction', required=True,
                        help='Task-specific user message; preserve the exact text in the run record')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-turns', type=int, default=16)
    parser.add_argument('--max-no-tool-retries', type=int, default=2)
    parser.add_argument('--max-new-tokens-per-turn', type=int, default=256)
    parser.add_argument('--max-total-new-tokens', type=int, default=3072)
    parser.add_argument('--max-context-tokens', type=int, default=8192)
    parser.add_argument('--wall-seconds', type=int, default=900)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--do-sample', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--edit-format', choices=('unified_diff', 'whole_file'), default='unified_diff')
    parser.add_argument('--require-split', choices=('train', 'heldout'))
    args = parser.parse_args()

    model_dir = args.model_path.resolve()
    if not model_dir.is_dir():
        parser.error(f'local model snapshot directory does not exist: {model_dir}')
    workspace = args.workspace.resolve()
    task_root = args.task_root.resolve()
    result_path = args.output.resolve()
    if result_path.exists():
        parser.error(f'refusing to overwrite existing agent record: {result_path}')
    contract = load_agent_task(task_root)
    if args.require_split is not None and contract.split != args.require_split:
        parser.error(f'task split {contract.split!r} does not match required split {args.require_split}')
    allowed = set(contract.allowed_files)
    missing = sorted(name for name in allowed if not (workspace / name).is_file())
    if missing:
        parser.error(f'workspace missing allowed source files: {missing}')
    if not os.environ.get('HF_HUB_OFFLINE') == '1' or not os.environ.get('TRANSFORMERS_OFFLINE') == '1':
        parser.error('HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required')

    start = time.perf_counter()
    if args.backend == 'mlx':
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            parser.error('the MLX backend requires Apple Silicon macOS')
        import mlx.core as mx
        from mlx_lm import load
        mx.random.seed(args.seed)
        model, tokenizer = load(str(model_dir))
        mlx_lm = __import__('mlx_lm')
        runtime = {'python': platform.python_version(), 'mlx': getattr(__import__('mlx'), '__version__', 'unknown'),
                   'mlx_lm': getattr(mlx_lm, '__version__', 'unknown'),
                   'platform': platform.platform(), 'device': 'apple_metal', 'seed': args.seed}
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.float32,
        ).to('cpu').eval()
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        if not all(parameter.device.type == 'cpu' for parameter in model.parameters()):
            raise RuntimeError('model parameter escaped CPU')
        runtime = {'python': platform.python_version(), 'torch': torch.__version__,
                   'transformers': __import__('transformers').__version__,
                   'platform': platform.platform(), 'device': 'cpu', 'threads': torch.get_num_threads(),
                   'cuda_available': torch.cuda.is_available(),
                   'mps_available_unused': torch.backends.mps.is_available()}

    task_text = contract.prompt_text
    test_files = {name for name in allowed if Path(name).name == 'test_public.py'}
    editable = allowed - test_files
    if args.public_test_image and len(test_files) != 1:
        parser.error('isolated public-test execution requires exactly one test_public.py in the task')
    tools = tools_for(args.edit_format, public_tests=bool(args.public_test_image))
    tool_names = {item['function']['name'] for item in tools}
    if args.edit_format == 'whole_file':
        edit_sentence = (
            'Read the source path named in the user instruction, then use tools to inspect and edit. '
            'Use write_file to replace an allowlisted source file with its complete new text. Do not send a unified diff. '
            'Keep the replacement concise enough to fit within the per-turn output budget. '
        )
    else:
        edit_sentence = (
            'Read the source path named in the user instruction, then use tools to inspect and edit. Use apply_patch for small source diffs; never rewrite a whole file. '
            'The patch must use exact headers --- a/<allowed source path> and +++ b/<allowed source path>; do not wrap it in a code fence. '
        )
    system = (
        'You are an offline code repair agent. Solve only the task below. The workspace contains only the task source files. '
        + edit_sentence
        + 'Allowed source paths: ' + ', '.join(sorted(editable)) + '. First call list_files, then read the exact source path named in the task. '
        + 'read_file returns at most 180 lines by default and 200 lines per call; use total_lines and has_more to request the next needed inclusive line range. '
        + 'Use search_text to locate relevant code before paging through large files. '
        + 'Do not invent paths. The workspace is already prepared; never ask for a workspace path, more information, or permission. '
        + 'The external grader is hidden. Do not describe a fix without applying it. Call finish only after changing a source file. '
        + ('After changing source, call run_public_tests; read its output and repair failures. The test file is immutable. '
           'Call finish only after the public smoke test passes. ' if args.public_test_image else '')
        + 'You have a limited tool budget; each assistant turn before finish must call a tool. '
        'TASK:\n' + task_text.split('## Reproduce', 1)[0].rstrip()
    )
    messages: list[dict[str, str]] = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': args.instruction},
    ]
    initial_messages = [dict(message) for message in messages]
    transcript: list[dict[str, Any]] = []
    workspace_revision = run_git(workspace, 'rev-parse', 'HEAD')
    if workspace_revision != contract.diff_revision:
        parser.error('candidate workspace HEAD differs from the task diff revision')
    source_before = snapshot_allowed_sources(workspace, sorted(allowed))
    before = digest_tree(workspace, sorted(allowed))
    token_total = 0
    stop_reason = 'turn_budget_exhausted'
    final_summary = ''
    no_tool_retries = 0
    edit_attempted = False
    edit_applied = False
    public_test_hash = sha256(workspace / next(iter(test_files))) if test_files else None
    public_test_result = None
    format_parse_failure = None
    format_parse_failure_count = 0
    pattern = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)

    def note_edit(edit_result: dict[str, Any]) -> None:
        nonlocal edit_attempted, edit_applied, format_parse_failure, format_parse_failure_count
        edit_attempted = True
        failure = edit_result.get('format_parse_failure')
        if failure:
            format_parse_failure = failure
            format_parse_failure_count += 1
        if edit_result.get('applied') is True:
            edit_applied = True

    for turn in range(args.max_turns):
        elapsed = time.perf_counter() - start
        if elapsed >= args.wall_seconds:
            stop_reason = 'wall_time_exhausted'
            break
        remaining = args.max_total_new_tokens - token_total
        if remaining <= 0:
            stop_reason = 'token_budget_exhausted'
            break
        limit = min(args.max_new_tokens_per_turn, remaining)
        generation_start = time.perf_counter()
        try:
            raw, generated_tokens, prompt_tokens = generate_turn(
                args.backend, model, tokenizer, messages, tools, limit=limit, args=args,
            )
        except ValueError as exc:
            if str(exc) != 'context_budget_exhausted':
                raise
            stop_reason = 'context_budget_exhausted'
            break
        generation_elapsed = time.perf_counter() - generation_start
        token_total += generated_tokens
        entry: dict[str, Any] = {'turn': turn + 1, 'model_output': raw, 'tool_calls': [],
                                 'generated_tokens': generated_tokens,
                                 'prompt_tokens': prompt_tokens,
                                 'generation_seconds': round(generation_elapsed, 6)}
        parse_text, recovered_tool_format = normalize_tool_call_format(raw, tool_names)
        if recovered_tool_format:
            entry['tool_call_format_recovery'] = 'strict_qwen_json_normalization'
        calls = list(pattern.finditer(parse_text))
        if not calls:
            recovered = parse_submission(raw, edit_format=args.edit_format)
            if not isinstance(recovered, ParseFailure) or recovered.reason != 'no_edit':
                edit_result = attempt_configured_edit(workspace, raw, args.edit_format, editable)
                note_edit(edit_result)
                if edit_result.get('applied') is True:
                    public_test_result = refresh_public_tests_after_edit(
                        edit_result, workspace, args.public_test_image, public_test_hash,
                    )
                    if public_test_result is not None:
                        entry['automatic_public_test'] = public_test_result
                        edit_result = {**edit_result, 'automatic_public_test': public_test_result}
                entry['tool_calls'].append({'name': 'recovered_edit', 'result': edit_result})
                messages.append({'role': 'assistant', 'content': raw})
                messages.append({
                    'role': 'tool', 'name': 'recovered_edit',
                    'content': json.dumps(edit_result, ensure_ascii=False),
                })
                transcript.append(entry)
                if edit_result.get('applied') is True:
                    continue
                stop_reason = 'malformed_edit'
                break
            messages.append({'role': 'assistant', 'content': raw})
            transcript.append(entry)
            current_hashes = digest_tree(workspace, sorted(allowed))
            if current_hashes == before and no_tool_retries < args.max_no_tool_retries:
                no_tool_retries += 1
                messages.append({'role': 'user', 'content': no_tool_retry_reminder(
                    raw, edit_format=args.edit_format, allowed_paths=sorted(editable),
                    token_limit=limit,
                )})
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
                if name not in tool_names or not isinstance(arguments, dict):
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
                        text = read_regular_source(workspace, path).decode('utf-8')
                        result = {
                            'resolved_path': path,
                            **bounded_source_page(
                                text,
                                start_line=arguments.get('start_line', 1),
                                end_line=arguments.get('end_line'),
                            ),
                        }
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
                    result = attempt_configured_edit(
                        workspace, str(arguments.get('patch', '')), args.edit_format, editable,
                    )
                    note_edit(result)
                    if result.get('applied') is True:
                        public_test_result = refresh_public_tests_after_edit(
                            result, workspace, args.public_test_image, public_test_hash,
                        )
                        if public_test_result is not None:
                            entry['automatic_public_test'] = public_test_result
                            result = {**result, 'automatic_public_test': public_test_result}
                elif name == 'write_file':
                    result = attempt_configured_edit(
                        workspace,
                        json.dumps({
                            'path': arguments.get('path', ''),
                            'content': arguments.get('content', ''),
                        }),
                        args.edit_format,
                        editable,
                    )
                    note_edit(result)
                    if result.get('applied') is True:
                        public_test_result = refresh_public_tests_after_edit(
                            result, workspace, args.public_test_image, public_test_hash,
                        )
                        if public_test_result is not None:
                            entry['automatic_public_test'] = public_test_result
                            result = {**result, 'automatic_public_test': public_test_result}
                elif name == 'run_public_tests':
                    if args.public_test_image and public_test_hash:
                        public_test_result = run_public_tests(
                            workspace, args.public_test_image, public_test_hash,
                        )
                    else:
                        public_test_result = {
                            'passed': False,
                            'error': 'public test image is not configured',
                        }
                    result = public_test_result
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
                    elif args.public_test_image and not (public_test_result or {}).get('passed'):
                        result = {'finished': False, 'error': 'public tests have not passed; run them and repair failures'}
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
    diff = candidate_source_diff(workspace, sorted(allowed), source_before, workspace_revision)
    if args.backend == 'mlx':
        runtime['mlx_active_memory_bytes'] = int(mx.metal.get_active_memory())
        runtime['mlx_peak_memory_bytes'] = int(mx.metal.get_peak_memory())
    result = {
        'schema_version': 1, 'task_id': contract.task_id, 'task_schema': contract.schema,
        'seed': args.seed,
        'task_revisions': {'base': contract.base_revision,
                           'calibration': contract.calibration_revision},
        'task_input_hashes': contract.input_hashes,
        'initial_messages': initial_messages,
        'model_repository': args.model_repository, 'model_snapshot': model_dir.name,
        'model_files': {path.name: sha256(path) for path in sorted(model_dir.iterdir()) if path.is_file()},
        'runtime': runtime,
        'budgets': {'max_turns': args.max_turns, 'max_new_tokens_per_turn': args.max_new_tokens_per_turn,
                    'max_total_new_tokens': args.max_total_new_tokens,
                    'max_context_tokens': args.max_context_tokens, 'wall_seconds': args.wall_seconds,
                    'max_no_tool_retries': args.max_no_tool_retries,
                    'temperature': args.temperature, 'top_p': args.top_p, 'top_k': args.top_k,
                    'do_sample': args.do_sample, 'edit_format': args.edit_format},
        'edit_format': args.edit_format,
        'edit_attempted': edit_attempted,
        'edit_applied': edit_applied,
        'public_test_result': public_test_result,
        'format_parse_failure': format_parse_failure,
        'format_parse_failure_count': format_parse_failure_count,
        'stop_reason': stop_reason, 'agent_summary': final_summary,
        'elapsed_seconds': round(elapsed, 3), 'new_tokens': token_total,
        'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'workspace_revision': workspace_revision,
        'candidate_diff': diff,
        'candidate_diff_sha256': hashlib.sha256(diff.encode()).hexdigest(),
        'source_hashes_before': before, 'source_hashes_after': after,
        'tool_turns': len(transcript), 'no_tool_retries': no_tool_retries, 'transcript': transcript,
        'offline_environment_required': True,
        'wall_budget_checked_between_generations': True,
        'claim_limit': 'One small cached model on one historical task; not frontier-agent performance or novel fix discovery.',
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with result_path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    except FileExistsError:
        parser.error(f'refusing to overwrite existing agent record: {result_path}')
    print(json.dumps({key: result[key] for key in ['task_id', 'seed', 'stop_reason', 'elapsed_seconds', 'new_tokens', 'tool_turns', 'candidate_diff_sha256']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
