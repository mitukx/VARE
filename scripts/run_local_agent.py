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
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.agent_edit_format import ParseFailure, apply_parsed_edit, parse_submission, require_edit_format  # noqa: E402
from vare.agent_task_contract import load_agent_task, verify_agent_workspace  # noqa: E402
from vare.agent_structured_edit import (  # noqa: E402
    StructuredEditError,
    StructuredEditController,
    apply_structured_edit,
    sha256_bytes,
    unified_diff,
    workspace_revision,
)

MODEL_REPOSITORY = 'Qwen/Qwen2.5-0.5B-Instruct'
TOOLS: list[dict[str, Any]] = [
    {'type': 'function', 'function': {'name': 'list_files', 'description': 'List the exact source files available in this task workspace.', 'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'read_file', 'description': 'Read one UTF-8 task source file. Provide its repository-relative path; a workspace-prefixed path is mapped only if it ends in one exact allowlisted source path.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path'], 'additionalProperties': False}}},
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


def tools_for(
    edit_format: str, *, public_tests: bool = False, opaque_handles: bool = False,
) -> list[dict[str, Any]]:
    if edit_format == 'structured':
        tools = [item for item in TOOLS if item['function']['name'] not in {'apply_patch'}]
        tools = [
            {'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a bounded 1-based line range from one allowed UTF-8 source file. The result includes the snapshot handles for the next edit.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'start_line': {'type': 'integer'}, 'end_line': {'type': 'integer'}}, 'required': ['path'], 'additionalProperties': False}}}
            if item['function']['name'] == 'read_file' else item
            for item in tools
        ]
        tools = [
            {'type': 'function', 'function': {'name': 'search_text', 'description': 'Search a literal query in allowed source text; optionally restrict to one allowed path.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}, 'path': {'type': 'string'}}, 'required': ['query'], 'additionalProperties': False}}}
            if item['function']['name'] == 'search_text' else item
            for item in tools
        ]
        handle_properties = (
            {'workspace_revision_handle': {'type': 'string'}, 'source_preimage_handle': {'type': 'string'}}
            if opaque_handles else {'expected_hash': {'type': 'string'}, 'expected_revision': {'type': 'string'}}
        )
        handle_required = (
            ['workspace_revision_handle', 'source_preimage_handle']
            if opaque_handles else ['expected_hash', 'expected_revision']
        )
        edit_handle_description = (
            'Supply workspace_revision_handle and source_preimage_handle from read_file; VARE resolves them to exact host-side hashes and rejects stale snapshots.'
            if opaque_handles else
            'Supply expected_hash and expected_revision exactly as returned by read_file; stale snapshots are rejected.'
        )
        tools.extend([
            {'type': 'function', 'function': {'name': 'replace_text', 'description': f'Replace one exact source excerpt. {edit_handle_description}', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}, **handle_properties}, 'required': ['path', 'old_text', 'new_text', *handle_required], 'additionalProperties': False}}},
            {'type': 'function', 'function': {'name': 'insert_text', 'description': f'Insert text immediately after one exact source anchor. {edit_handle_description}', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'anchor': {'type': 'string'}, 'content': {'type': 'string'}, **handle_properties}, 'required': ['path', 'anchor', 'content', *handle_required], 'additionalProperties': False}}},
            {'type': 'function', 'function': {'name': 'inspect_diff', 'description': 'Inspect the deterministic diff generated by VARE for all current source edits.', 'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
        ])
    elif edit_format == 'unified_diff':
        require_edit_format(edit_format)
        tools = list(TOOLS)
    else:
        require_edit_format(edit_format)
        tools = [WRITE_FILE_TOOL if item['function']['name'] == 'apply_patch' else item for item in TOOLS]
    if public_tests:
        tools.insert(-1, PUBLIC_TEST_TOOL)
    return tools


def candidate_visible_tool_result(
    name: str, result: dict[str, Any], *, opaque_handles: bool,
    max_read_chars: int = 6_000,
) -> Any:
    """Keep exact provenance in the transcript while sending bounded feedback to the model."""
    if not opaque_handles:
        return result
    def bounded_diff(value: Any) -> str:
        diff = value if isinstance(value, str) else ""
        if len(diff) <= 6_000:
            return diff
        digest = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        return (
            diff[:2_900]
            + f"\n... diff truncated for model context; chars={len(diff)} sha256={digest} ...\n"
            + diff[-2_900:]
        )

    if name == 'read_file':
        visible = {
            key: result[key]
            for key in (
                'path', 'start_line', 'end_line', 'workspace_revision_handle',
                'next_line', 'truncated', 'source_preimage_handle', 'read_only', 'error',
            )
            if key in result
        }
        content = result.get('content', '')
        if isinstance(content, str):
            payload = {**visible, 'content': content}
            if len(json.dumps(payload, ensure_ascii=False)) > max_read_chars:
                start_line = visible.get('start_line')
                if not isinstance(start_line, int) or isinstance(start_line, bool):
                    start_line = 1
                lines = content.splitlines(keepends=True)
                prefix: list[str] = []
                for line in lines:
                    candidate = ''.join([*prefix, line])
                    count = len(prefix) + 1
                    candidate_visible = {
                        **visible,
                        'content': candidate,
                        'end_line': start_line + count - 1,
                        'next_line': start_line + count,
                        'truncated': True,
                    }
                    if len(json.dumps(candidate_visible, ensure_ascii=False)) > max_read_chars:
                        break
                    prefix.append(line)
                count = len(prefix)
                visible['content'] = ''.join(prefix)
                visible['end_line'] = start_line + count - 1
                visible['next_line'] = start_line + count
                visible['truncated'] = True
            else:
                visible['content'] = content
        return visible

    if name == 'search_text':
        # The controller may return up to 30 long lines. Keep the model-facing
        # payload within the same per-turn budget used by source reads.
        matches: list[Any] = result if isinstance(result, list) else []
        search_visible: list[dict[str, Any]] = []
        for match in matches:
            if not isinstance(match, dict):
                continue
            item = {key: match[key] for key in ('path', 'line', 'text') if key in match}
            search_visible.append(item)
            if len(json.dumps(search_visible, ensure_ascii=False)) > max_read_chars:
                search_visible.pop()
                break
        if not search_visible and max_read_chars > 2:
            return []
        # If one match is too large, retain only a bounded excerpt. A subsequent
        # search/read can retrieve more context on a later turn.
        while search_visible and len(json.dumps(search_visible, ensure_ascii=False)) > max_read_chars:
            text = search_visible[-1].get('text')
            if isinstance(text, str) and text:
                search_visible[-1]['text'] = text[: max(0, len(text) // 2)]
            else:
                search_visible.pop()
        return search_visible

    if name in {'replace_text', 'insert_text'}:
        visible = {key: result[key] for key in ('applied', 'path', 'error') if key in result}
        if 'diff' in result:
            visible['diff'] = bounded_diff(result['diff'])
        feedback = result.get('automatic_public_test')
        if isinstance(feedback, dict):
            visible['automatic_public_test'] = {
                key: feedback[key]
                for key in ('passed', 'repeated_unchanged_failure', 'feedback', 'next_action')
                if key in feedback
            }
        return visible
    if name == 'inspect_diff':
        return {
            'diff': bounded_diff(result.get('diff', '')),
            'changed': result.get('changed', False),
        }
    if name == 'run_public_tests':
        return {
            key: result[key]
            for key in ('passed', 'repeated_unchanged_failure', 'feedback', 'next_action')
            if key in result
        }
    return result


def should_stop_after_tool(name: str, result: dict[str, Any]) -> bool:
    """A rejected finalization is feedback, not a terminal agent action."""
    return name == 'finish' and result.get('finished') is True


def record_tool_call(
    entry: dict[str, Any], name: str, arguments: dict[str, Any], raw_result: Any,
) -> None:
    """Retain the unabridged host result even when model feedback is summarized."""
    entry.setdefault('tool_calls', []).append(
        {'name': name, 'arguments': arguments, 'result': raw_result},
    )


def record_automatic_public_test(entry: dict[str, Any], result: dict[str, Any]) -> None:
    """Retain each raw post-edit test outcome, including earlier failures."""
    entry.setdefault('automatic_public_tests', []).append(result)
    entry['automatic_public_test'] = result


def bounded_turn_payload(model_result: Any, remaining_chars: int) -> str:
    """Serialize one model-facing result within the shared per-turn character budget."""
    if remaining_chars <= 0:
        return ''
    payload = json.dumps(model_result, ensure_ascii=False)
    if len(payload) <= remaining_chars:
        return payload
    if isinstance(model_result, dict) and isinstance(model_result.get('diff'), str):
        diff = model_result['diff']
        digest = hashlib.sha256(diff.encode('utf-8')).hexdigest()
        summary_fields = {key: value for key, value in model_result.items() if key != 'diff'}
        summary_fields.update({
            'diff_truncated': True,
            'diff_chars': len(diff),
            'diff_sha256': digest,
        })
        empty_excerpt = json.dumps({**summary_fields, 'diff': ''}, ensure_ascii=False)
        marker = f'\n... diff excerpt truncated; chars={len(diff)} sha256={digest} ...\n'
        available = max(0, remaining_chars - len(empty_excerpt))
        while available > 0:
            keep = max(0, available - len(marker))
            head = (keep + 1) // 2
            tail = keep // 2
            excerpt = diff[:head] + marker[:available - keep] + (diff[-tail:] if tail else '')
            candidate = json.dumps({**summary_fields, 'diff': excerpt}, ensure_ascii=False)
            if len(candidate) <= remaining_chars:
                return candidate
            available -= max(1, len(candidate) - remaining_chars)
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    truncation_summary = json.dumps({
        'feedback_truncated': True,
        'original_chars': len(payload),
        'sha256': digest,
    }, ensure_ascii=False)
    if len(truncation_summary) <= remaining_chars:
        return truncation_summary
    return ''


def generate_turn(backend: str, model: Any, tokenizer: Any, messages: list[dict[str, Any]],
                  tools: list[dict[str, Any]], *, limit: int, args: argparse.Namespace) -> tuple[str, int, int, dict[str, Any]]:
    """Generate one bounded response and retain backend token-rate telemetry."""
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if prompt_tokens >= args.max_context_tokens:
        raise ValueError('context_budget_exhausted')
    limit = min(limit, args.max_context_tokens - prompt_tokens)
    if backend == 'mlx':
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(
            temp=0.0 if not args.do_sample else args.temperature,
            top_p=0.0 if not args.do_sample else args.top_p,
            top_k=0 if not args.do_sample else args.top_k,
        )
        pieces: list[str] = []
        last = None
        for response in stream_generate(model, tokenizer, prompt, max_tokens=limit, sampler=sampler):
            pieces.append(response.text)
            last = response
        raw = ''.join(pieces)
        generated_tokens = getattr(last, 'generation_tokens', None)
        if not isinstance(generated_tokens, int) or generated_tokens < 0:
            generated_tokens = len(tokenizer.encode(raw, add_special_tokens=False))
        return raw, generated_tokens, prompt_tokens, {
            'prompt_tokens_per_second': getattr(last, 'prompt_tps', None),
            'decode_tokens_per_second': getattr(last, 'generation_tps', None),
            'mlx_peak_memory_gib': getattr(last, 'peak_memory', None),
        }

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
    generated_tokens = int(new_ids.shape[0])
    return tokenizer.decode(new_ids, skip_special_tokens=False), generated_tokens, prompt_tokens, {
        'prompt_tokens_per_second': None,
        'decode_tokens_per_second': None,
        'mlx_peak_memory_gib': None,
    }


def run_public_tests(
    workspace: Path, image: str, expected_test_hash: str, memory_mb: int = 512,
) -> dict[str, Any]:
    """Run only the immutable public smoke file in VARE's candidate sandbox."""
    test_file = workspace / 'test_public.py'
    if not test_file.is_file() or sha256(test_file) != expected_test_hash:
        return {'passed': False, 'error': 'public test file is missing or changed'}
    from vare.container_isolation import DockerIsolationRunner, IsolationLimits

    limits = IsolationLimits(image=image, timeout_s=45, memory_mb=memory_mb, cpus=1.0,
                             pids_limit=32, max_output_bytes=20_000)
    started = time.perf_counter()
    try:
        result = asyncio.run(DockerIsolationRunner().run_candidate(
            limits, allowed_workspace_root=workspace.parent, workspace=workspace,
            command=public_test_command(),
            readonly_workspace=True,
        ))
    except Exception as exc:
        return {'passed': False, 'error': f'{type(exc).__name__}: {exc}',
                'elapsed_seconds': round(time.perf_counter() - started, 3)}
    return {'passed': result.returncode == 0, 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr,
            'elapsed_seconds': round(time.perf_counter() - started, 3),
            'isolation': 'docker-network-none-read-only-root-only-workspace-mount'}


def public_test_command() -> list[str]:
    """Run task tests with the archived Transformers source tree importable."""
    return [
        'python', '-c',
        "import runpy,sys;sys.path.insert(0,'/workspace/src');tests=runpy.run_path('/workspace/test_public.py');[fn() for name,fn in tests.items() if name.startswith('test_') and callable(fn)]",
    ]


def refresh_public_tests_after_edit(
    edit_result: dict[str, Any], workspace: Path, image: str | None,
    expected_test_hash: str | None, memory_mb: int = 512,
) -> dict[str, Any] | None:
    """Return a fresh smoke-test result after an applied source edit."""
    if edit_result.get('applied') is not True or not image or not expected_test_hash:
        return None
    return run_public_tests(workspace, image, expected_test_hash, memory_mb)


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
                           token_limit: int, opaque_handles: bool = False) -> str:
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
        if edit_format == 'structured':
            handles = (
                'workspace_revision_handle and source_preimage_handle'
                if opaque_handles else 'expected_hash and expected_revision'
            )
            return (
                f'Your previous response was cut off at the {token_limit}-token limit. Do not reread the whole file. '
                'Emit one complete short JSON tool call now. Use read_file for a small line range, then one replace_text '
                f'or insert_text action with the exact {handles} returned by read_file.'
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
    parser.add_argument('--public-test-memory-mb', type=int, default=512,
                        help='Memory limit for the isolated public regression test container')
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instruction', required=True,
                        help='Task-specific user message; preserve the exact text in the run record')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-turns', type=int, default=16)
    parser.add_argument('--max-no-tool-retries', type=int, default=2)
    parser.add_argument('--max-new-tokens-per-turn', type=int, default=512)
    parser.add_argument('--max-total-new-tokens', type=int, default=3072)
    parser.add_argument('--max-context-tokens', type=int, default=8192)
    parser.add_argument('--wall-seconds', type=int, default=900)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--do-sample', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--edit-format', choices=('unified_diff', 'whole_file', 'structured'), default='unified_diff')
    parser.add_argument('--require-split', choices=('train', 'development', 'heldout'))
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
    try:
        verify_agent_workspace(workspace, contract)
    except ValueError as exc:
        parser.error(str(exc))
    if not os.environ.get('HF_HUB_OFFLINE') == '1' or not os.environ.get('TRANSFORMERS_OFFLINE') == '1':
        parser.error('HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required')

    start = time.perf_counter()
    if args.backend == 'mlx':
        from importlib.metadata import version
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            parser.error('the MLX backend requires Apple Silicon macOS')
        import mlx.core as mx
        from mlx_lm import load
        if not mx.metal.is_available() or str(mx.default_device()) != 'Device(gpu, 0)':
            parser.error('the MLX backend requires Metal GPU as the active default device')
        mx.random.seed(args.seed)
        model, tokenizer = load(str(model_dir))
        runtime = {'python': platform.python_version(), 'mlx': version('mlx'),
                   'mlx_lm': version('mlx-lm'),
                   'platform': platform.platform(), 'device': 'apple_metal', 'seed': args.seed}
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        model = cast(Any, AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.float32,
        )).to('cpu').eval()
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
    use_opaque_handles = args.edit_format == 'structured' and bool(args.public_test_image)
    edit_controller = (
        StructuredEditController(
            workspace, editable, readable_files=allowed,
            max_read_lines=40, max_read_chars=8_000,
        )
        if use_opaque_handles else None
    )
    tools = tools_for(
        args.edit_format,
        public_tests=bool(args.public_test_image),
        opaque_handles=use_opaque_handles,
    )
    tool_names = {item['function']['name'] for item in tools}
    if args.edit_format == 'structured':
        if use_opaque_handles:
            edit_sentence = (
                'Use short structured actions, never emit a unified diff or rewrite a whole file. Read bounded source regions. '
                'For replace_text or insert_text, copy the short workspace_revision_handle and source_preimage_handle returned '
                'by the latest read; do not invent or reproduce hashes. VARE creates the diff and resolves handles to exact '
                'host-side source hashes. After a failed public test, act on its failing assertion and make a different targeted '
                'edit. Inspect the final diff after the last edit and test, then finish. '
            )
        else:
            edit_sentence = (
                'Use short structured actions, never emit a unified diff or rewrite a whole file. Read bounded source ranges, '
                'then call replace_text or insert_text with the exact expected_hash and expected_revision returned by read_file. '
                'VARE creates the diff. Inspect it, run public tests, revise using fresh hashes, then finish. '
            )
    elif args.edit_format == 'whole_file':
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
        + 'Allowed source paths: ' + ', '.join(sorted(editable)) + '. First call list_files. Then search the allowed source for two or three distinctive terms from the issue before reading a narrow region around the relevant implementation. '
        'Do not inspect a large file by reading sequential chunks. Use the search results to locate one likely function, inspect at most a short source region, and form a minimal edit plan before changing code. '
        + 'Do not invent paths. The workspace is already prepared; never ask for a workspace path, more information, or permission. '
        + 'The external grader is hidden. Do not describe a fix without applying it. Call finish only after changing a source file. '
        + ('After each source edit VARE automatically runs the immutable public smoke test and includes bounded failure feedback in the edit response. '
           'Use this loop: inspect, diagnose, plan a minimal change, edit, test, interpret the failure, and revise if needed. '
           'When a public test fails, carry its failing assertion into your next action and search/read the implicated code or make a different targeted edit; never resubmit the same patch. '
           'Inspect the final diff after the latest edit and call finish only after the latest public test passes. ' if args.public_test_image else '')
        + 'You have a limited tool budget; each assistant turn before finish must call a tool. '
        'TASK:\n' + task_text.split('## Reproduce', 1)[0].rstrip()
    )
    messages: list[dict[str, Any]] = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': args.instruction},
    ]
    initial_messages = [dict(message) for message in messages]
    transcript: list[dict[str, Any]] = []
    before = digest_tree(workspace, sorted(allowed))
    initial_editable_bytes = {name: (workspace / name).read_bytes() for name in editable}
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
            raw, generated_tokens, prompt_tokens, generation_metrics = generate_turn(
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
                                 **generation_metrics,
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
                        args.public_test_memory_mb,
                    )
                    if public_test_result is not None:
                        record_automatic_public_test(entry, public_test_result)
                        edit_result = {**edit_result, 'automatic_public_test': public_test_result}
                entry['tool_calls'].append({'name': 'recovered_edit', 'result': edit_result})
                messages.append({'role': 'assistant', 'content': raw})
                recovered_feedback = (
                    bounded_turn_payload(edit_result, 6_000)
                    if edit_controller is not None
                    else json.dumps(edit_result, ensure_ascii=False)
                )
                messages.append({
                    'role': 'tool', 'name': 'recovered_edit',
                    'content': recovered_feedback,
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
                    token_limit=limit, opaque_handles=edit_controller is not None,
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
        turn_output_chars = 0
        turn_output_budget = 6_000
        for match, name, arguments in parsed_calls:
            try:
                model_feedback_override: dict[str, Any] | None = None
                if name == 'list_files':
                    result: Any = sorted(allowed)
                elif name == 'read_file':
                    raw_path = arguments.get('path', '')
                    path = resolve_allowed_path(raw_path, allowed)
                    if path is None:
                        result = {'error': 'path is not allowed', 'allowed_paths': sorted(allowed)}
                    elif edit_controller is not None:
                        start_line = arguments.get('start_line', 1)
                        end_line = arguments.get('end_line')
                        if (not isinstance(start_line, int) or isinstance(start_line, bool)
                                or (end_line is not None and (
                                    not isinstance(end_line, int) or isinstance(end_line, bool)
                                ))):
                            result = {'error': 'line range must use integers'}
                        else:
                            try:
                                result = edit_controller.read_region(
                                    path, start_line=start_line, end_line=end_line,
                                )
                            except StructuredEditError as exc:
                                result = {'error': str(exc)}
                    elif args.edit_format == 'structured':
                        source = (workspace / path).read_text(encoding='utf-8')
                        lines = source.splitlines()
                        start_line = arguments.get('start_line', 1)
                        end_line = arguments.get('end_line', min(len(lines), 160))
                        if (not isinstance(start_line, int) or isinstance(start_line, bool)
                                or not isinstance(end_line, int) or isinstance(end_line, bool)
                                or start_line < 1 or end_line < start_line or end_line - start_line >= 200):
                            result = {'error': 'line range must be 1-based and contain at most 200 lines'}
                        else:
                            selected = lines[start_line - 1:end_line]
                            result = {
                                'resolved_path': path,
                                'start_line': start_line,
                                'end_line': min(end_line, len(lines)),
                                'line_count': len(lines),
                                'content': '\n'.join(selected) + ('\n' if selected else ''),
                                'expected_hash': sha256_bytes((workspace / path).read_bytes()),
                                'expected_revision': workspace_revision(workspace, editable),
                            }
                    else:
                        result = {'resolved_path': path,
                                  'content': (workspace / path).read_text(encoding='utf-8')}
                elif name == 'search_text':
                    query = arguments.get('query', '')
                    if not query:
                        result = {'error': 'query must not be empty'}
                    else:
                        matches = []
                        raw_path = arguments.get('path')
                        scoped = resolve_allowed_path(raw_path, allowed) if raw_path else None
                        if raw_path and scoped is None:
                            result = {'error': 'path is not allowed', 'allowed_paths': sorted(allowed)}
                        else:
                            paths = [scoped] if scoped else sorted(allowed)
                            for path in paths:
                                for line_no, line in enumerate((workspace / path).read_text(encoding='utf-8').splitlines(), 1):
                                    if query in line:
                                        line_limit = 300 if edit_controller is not None else 500
                                        matches.append({'path': path, 'line': line_no, 'text': line[:line_limit]})
                            result = matches[:30] if edit_controller is not None else matches[:80]
                elif name in {'replace_text', 'insert_text'}:
                    try:
                        if edit_controller is not None:
                            result = edit_controller.apply_edit(
                                action=name,
                                path=str(arguments.get('path', '')),
                                workspace_revision_handle=str(arguments.get('workspace_revision_handle', '')),
                                source_preimage_handle=str(arguments.get('source_preimage_handle', '')),
                                old_text=str(arguments.get('old_text', '')),
                                new_text=str(arguments.get('new_text', arguments.get('content', ''))),
                                anchor=str(arguments.get('anchor', '')),
                            )
                        else:
                            result = apply_structured_edit(
                                workspace, editable,
                                action=name,
                                path=str(arguments.get('path', '')),
                                expected_hash=str(arguments.get('expected_hash', '')),
                                expected_revision=str(arguments.get('expected_revision', '')),
                                old_text=str(arguments.get('old_text', '')),
                                new_text=str(arguments.get('new_text', arguments.get('content', ''))),
                                anchor=str(arguments.get('anchor', '')),
                            )
                        note_edit(result)
                        if result.get('applied') is True:
                            # A pass belongs to the exact source revision that was tested.
                            public_test_result = None
                            if args.public_test_image and public_test_hash:
                                public_test_result = refresh_public_tests_after_edit(
                                    result,
                                    workspace,
                                    args.public_test_image,
                                    public_test_hash,
                                    args.public_test_memory_mb,
                                )
                                if public_test_result is not None:
                                    record_automatic_public_test(entry, public_test_result)
                                    if edit_controller is not None:
                                        feedback = edit_controller.record_public_test_result(public_test_result)
                                        result = {**result, 'automatic_public_test': feedback}
                                    else:
                                        result = {**result, 'automatic_public_test': public_test_result}
                    except StructuredEditError as exc:
                        result = {'applied': False, 'error': str(exc)}
                        note_edit(result)
                elif name == 'inspect_diff':
                    if edit_controller is not None:
                        result = edit_controller.inspect_diff()
                    else:
                        diff = unified_diff(workspace, initial_editable_bytes, editable)
                        result = {
                            'diff': diff,
                            'diff_sha256': hashlib.sha256(diff.encode()).hexdigest(),
                            'source_hashes': digest_tree(workspace, sorted(editable)),
                            'workspace_revision': workspace_revision(workspace, editable),
                        }
                elif name == 'apply_patch':
                    result = attempt_configured_edit(
                        workspace, str(arguments.get('patch', '')), args.edit_format, editable,
                    )
                    note_edit(result)
                    if result.get('applied') is True:
                        public_test_result = refresh_public_tests_after_edit(
                            result, workspace, args.public_test_image, public_test_hash,
                            args.public_test_memory_mb,
                        )
                        if public_test_result is not None:
                            record_automatic_public_test(entry, public_test_result)
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
                            args.public_test_memory_mb,
                        )
                        if public_test_result is not None:
                            record_automatic_public_test(entry, public_test_result)
                            result = {**result, 'automatic_public_test': public_test_result}
                elif name == 'run_public_tests':
                    if args.public_test_image and public_test_hash:
                        public_test_result = run_public_tests(
                            workspace, args.public_test_image, public_test_hash,
                            args.public_test_memory_mb,
                        )
                    else:
                        public_test_result = {
                            'passed': False,
                            'error': 'public test image is not configured',
                        }
                    if edit_controller is not None:
                        model_feedback_override = edit_controller.record_public_test_result(
                            public_test_result,
                        )
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
                        if edit_controller is not None:
                            try:
                                finalized = edit_controller.finalize_for_independent_grading()
                                final_summary = str(arguments.get('summary', ''))
                                result = {
                                    'finished': True,
                                    'candidate_ready_for_independent_grading': True,
                                    'verified_success': False,
                                    'diff_sha256': finalized['diff_sha256'],
                                }
                                stop_reason = 'agent_finished'
                            except StructuredEditError as exc:
                                result = {'finished': False, 'error': str(exc)}
                        else:
                            final_summary = str(arguments.get('summary', ''))
                            result = {'finished': True}
                            stop_reason = 'agent_finished'
                feedback_result = model_feedback_override or result
                model_result = candidate_visible_tool_result(
                    name, feedback_result,
                    opaque_handles=edit_controller is not None,
                    max_read_chars=max(0, turn_output_budget - turn_output_chars),
                )
                model_payload = json.dumps(model_result, ensure_ascii=False)
                if edit_controller is not None:
                    remaining = max(0, turn_output_budget - turn_output_chars)
                    model_payload = bounded_turn_payload(model_result, remaining)
                    turn_output_chars += len(model_payload)
                record_tool_call(entry, name, arguments, result)
                messages.append({'role': 'tool', 'name': name, 'content': model_payload})
                if should_stop_after_tool(name, result):
                    break
            except Exception as exc:
                entry['tool_calls'].append({'error': f'{type(exc).__name__}: {exc}', 'raw_call': match.group(1)})
                error_payload = json.dumps({'error': str(exc)}, ensure_ascii=False)
                if edit_controller is not None:
                    remaining = max(0, turn_output_budget - turn_output_chars)
                    error_payload = bounded_turn_payload({'error': str(exc)}, remaining)
                    turn_output_chars += len(error_payload)
                messages.append({'role': 'tool', 'name': 'invalid_tool_call', 'content': error_payload})
        transcript.append(entry)
        if stop_reason == 'agent_finished':
            break

    elapsed = time.perf_counter() - start
    after = digest_tree(workspace, sorted(allowed))
    if args.edit_format == 'structured':
        diff = unified_diff(workspace, initial_editable_bytes, editable)
        recorded_workspace_revision = contract.base_revision or workspace_revision(workspace, editable)
    else:
        diff = run_git(workspace, 'diff', '--binary', contract.diff_revision, '--', *sorted(allowed))
        recorded_workspace_revision = run_git(workspace, 'rev-parse', 'HEAD')
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
        'workspace_revision': recorded_workspace_revision,
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
