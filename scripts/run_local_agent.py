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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.agent_edit_format import ParseFailure, apply_parsed_edit, parse_submission, require_edit_format  # noqa: E402
from vare.agent_task_contract import load_agent_task  # noqa: E402

MODEL_REPOSITORY = 'Qwen/Qwen2.5-0.5B-Instruct'
_LOCAL_IMAGE_ID = re.compile(r'sha256:[0-9a-f]{64}\Z')
_REPOSITORY_IMAGE_DIGEST = re.compile(r'[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}\Z')
TOOLS = [
    {'type': 'function', 'function': {'name': 'list_files', 'description': 'List the exact source files available in this task workspace.', 'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'read_file', 'description': 'Read up to 80 inclusive lines from one UTF-8 task source file. Provide its repository-relative path and, for a later section, optional start_line and end_line. If omitted, reads the first 80 lines. A workspace-prefixed path is mapped only if it ends in one exact allowlisted source path.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'start_line': {'type': 'integer'}, 'end_line': {'type': 'integer'}}, 'required': ['path'], 'additionalProperties': False}}},
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

    started = time.perf_counter()
    try:
        pinned_image = resolve_pinned_public_test_image(image)
        limits = IsolationLimits(image=pinned_image, timeout_s=45, memory_mb=512, cpus=1.0,
                                 pids_limit=32, max_output_bytes=20_000)
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


def resolve_pinned_public_test_image(image: str) -> str:
    """Map an immutable local image ID to a repository digest accepted by Docker isolation."""
    if _LOCAL_IMAGE_ID.fullmatch(image) is None:
        return image
    inspected = subprocess.run(
        ['docker', 'image', 'inspect', image, '--format', '{{json .RepoDigests}}'],
        capture_output=True, text=True, check=False, timeout=10,
    )
    if inspected.returncode != 0:
        raise ValueError('local public-test image ID is unavailable')
    try:
        repository_digests = json.loads(inspected.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError('Docker returned invalid repository-digest metadata') from exc
    if not isinstance(repository_digests, list):
        raise ValueError('local public-test image has no repository digest')
    for digest in sorted(repository_digests):
        if not isinstance(digest, str) or _REPOSITORY_IMAGE_DIGEST.fullmatch(digest) is None:
            continue
        verified = subprocess.run(
            ['docker', 'image', 'inspect', digest, '--format', '{{.Id}}'],
            capture_output=True, text=True, check=False, timeout=10,
        )
        if verified.returncode == 0 and verified.stdout.strip() == image:
            return digest
    raise ValueError('local public-test image ID does not resolve to an exact repository digest')


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
    return {name: sha256(root / name) for name in files if (root / name).is_file()}


def run_git(root: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(root), *args], check=True, text=True,
                          capture_output=True).stdout.strip()


def resolve_allowed_path(raw_path: str, allowed: set[str]) -> str | None:
    """Map a model-supplied path to one exact allowlisted relative file."""
    normalized = raw_path.replace('\\', '/').lstrip('./')
    matches = [path for path in allowed if normalized == path or normalized.endswith('/' + path)]
    return matches[0] if len(matches) == 1 else None


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
        # Some Qwen2.5 generations wrap an otherwise valid tool payload in an
        # extra opening brace and two extra closing braces. Recover only that
        # exact framing error; malformed arguments or mixed content in a block
        # remain untouched for the strict parser to reject.
        block_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
        recovered_any = False

        def recover_duplicate_wrapper(match: re.Match[str]) -> str:
            nonlocal recovered_any
            content = match.group(1).strip()
            if not content.startswith('{{'):
                return match.group(0)
            try:
                payload, end = json.JSONDecoder().raw_decode(content[1:])
            except json.JSONDecodeError:
                return match.group(0)
            # The malformed generations observed in the retained trace add
            # exactly one brace at each edge: ``{{<json-object>}}``. Accept
            # only that paired framing error. Requiring two trailing braces
            # here misses the actual model output and sends it to json.loads
            # with the extra leading brace still present.
            if re.fullmatch(r'(?:\}\}?\s*|\s+\}\})', content[1:][end:]) is None:
                return match.group(0)
            if (not isinstance(payload, dict) or set(payload) != {'name', 'arguments'}
                    or payload.get('name') not in tool_names
                    or not isinstance(payload.get('arguments'), dict)):
                return match.group(0)
            recovered_any = True
            encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
            return f'<tool_call>{encoded}</tool_call>'

        normalized = block_pattern.sub(recover_duplicate_wrapper, raw)
        return normalized, recovered_any

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


def parse_tool_call_payload(raw_call: str, tools: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Parse one complete JSON tool payload and enforce its declared schema."""
    call = json.loads(raw_call)
    if not isinstance(call, dict) or set(call) != {'name', 'arguments'}:
        raise ValueError('tool payload must contain exactly name and arguments')
    name = call['name']
    arguments = call['arguments']
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError('tool name and arguments have invalid types')
    specification = next(
        (tool['function'] for tool in tools if tool['function']['name'] == name), None,
    )
    if specification is None:
        raise ValueError('unsupported tool call')
    parameters = specification['parameters']
    properties = parameters['properties']
    required = set(parameters.get('required', []))
    if not required <= set(arguments) or set(arguments) - set(properties):
        raise ValueError('tool arguments do not match the declared schema')
    for key, value in arguments.items():
        expected_type = properties[key].get('type')
        if expected_type == 'string' and not isinstance(value, str):
            raise ValueError(f'tool argument {key!r} must be a string')
        if expected_type == 'object' and not isinstance(value, dict):
            raise ValueError(f'tool argument {key!r} must be an object')
        if expected_type == 'integer' and (type(value) is not int or value < 1):
            raise ValueError(f'tool argument {key!r} must be a positive integer')
    return name, arguments


def read_file_range(path: Path, display_path: str, start_line: int = 1,
                    end_line: int | None = None) -> dict[str, Any]:
    """Read a bounded inclusive line range without flooding the model context."""
    lines = path.read_text(encoding='utf-8').splitlines()
    requested_end = start_line + 79 if end_line is None else end_line
    if start_line < 1 or requested_end < start_line or requested_end - start_line >= 80:
        return {'error': 'read_file range must contain at most 80 positive-numbered lines'}
    if start_line > len(lines):
        return {'error': f'start_line exceeds file length ({len(lines)} lines)'}
    actual_end = min(requested_end, len(lines))
    return {
        'resolved_path': display_path,
        'start_line': start_line,
        'end_line': actual_end,
        'total_lines': len(lines),
        'truncated': actual_end < len(lines),
        'content': '\n'.join(lines[start_line - 1:actual_end]),
    }


def build_initial_messages(task_text: str, allowed_editable_files: list[str],
                           edit_format: str, public_tests: bool,
                           instruction: str) -> list[dict[str, str]]:
    """Build the exact frozen system/user messages used by a coding attempt."""
    if edit_format == 'whole_file':
        edit_sentence = (
            'Read the source path named in the user instruction, then use tools to inspect and edit. '
            'Use write_file to replace an allowlisted source file with its complete new text. Do not send a unified diff. '
            'Keep the replacement concise enough to fit within the per-turn output budget. '
        )
    elif edit_format == 'unified_diff':
        edit_sentence = (
            'Read the source path named in the user instruction, then use tools to inspect and edit. Use apply_patch for small source diffs; never rewrite a whole file. '
            'The patch must use exact headers --- a/<allowed source path> and +++ b/<allowed source path>; do not wrap it in a code fence. '
        )
    else:
        raise ValueError(f'unsupported edit format: {edit_format}')
    system = (
        'You are an offline code repair agent. Solve only the task below. The workspace contains only the task source files. '
        + edit_sentence
        + 'Allowed source paths: ' + ', '.join(sorted(allowed_editable_files)) + '. First call list_files. Then search for the relevant symbol and read its source with an exact path and a bounded line range. '
        + 'Do not invent paths. The workspace is already prepared; never ask for a workspace path, more information, or permission. '
        + 'The external grader is hidden. Do not describe a fix without applying it. Call finish only after changing a source file. '
        + ('After changing source, call run_public_tests; read its output and repair failures. The test file is immutable. '
           'Call finish only after the public smoke test passes. ' if public_tests else '')
        + 'For large files, use search_text to find the relevant symbol and read_file with start_line and end_line; each read is limited to 80 lines. '
        + 'You have a limited tool budget; each assistant turn before finish must call a tool. '
        'TASK:\n' + task_text.split('## Reproduce', 1)[0].rstrip()
    )
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': instruction}]


def parse_tool_call_blocks(
    text: str, tools: list[dict[str, Any]],
) -> tuple[list[re.Match[str]], list[tuple[re.Match[str], str, dict[str, Any]]],
           list[tuple[re.Match[str], str]]]:
    """Parse complete tool-call blocks without truncating nested JSON objects."""
    blocks = list(re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL))
    parsed: list[tuple[re.Match[str], str, dict[str, Any]]] = []
    failures: list[tuple[re.Match[str], str]] = []
    for block in blocks:
        try:
            name, arguments = parse_tool_call_payload(block.group(1), tools)
            parsed.append((block, name, arguments))
        except Exception as exc:
            failures.append((block, f'{type(exc).__name__}: {exc}'))
    return blocks, parsed, failures


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
    messages = build_initial_messages(
        task_text, sorted(editable), args.edit_format, bool(args.public_test_image), args.instruction,
    )
    initial_messages = [dict(message) for message in messages]
    transcript: list[dict[str, Any]] = []
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
            entry['tool_call_format_recovery'] = 'strict_allowlisted_json_recovery'
        calls, parsed_calls, call_errors = parse_tool_call_blocks(parse_text, tools)
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
        for match, error in call_errors:
            entry['tool_calls'].append({'error': error, 'raw_call': match.group(1)})
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
                        result = read_file_range(
                            workspace / path, path,
                            arguments.get('start_line', 1),
                            arguments.get('end_line'),
                        )
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
    diff = run_git(workspace, 'diff', '--binary', contract.diff_revision, '--', *sorted(allowed))
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
    try:
        with result_path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    except FileExistsError:
        parser.error(f'refusing to overwrite existing agent record: {result_path}')
    print(json.dumps({key: result[key] for key in ['task_id', 'seed', 'stop_reason', 'elapsed_seconds', 'new_tokens', 'tool_turns', 'candidate_diff_sha256']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
