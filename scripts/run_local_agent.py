#!/usr/bin/env python3
"""Offline, bounded tool-use runner for pinned VARE coding tasks."""
from __future__ import annotations

import argparse
import asyncio
import difflib
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
MAX_TOOL_RESULT_BYTES = 12_000
MAX_TOOL_RESULT_TOKENS = 2_048
MAX_SEARCH_MATCHES = 80
MAX_SEARCH_SNIPPET_CHARS = 500
CONTEXT_COMPACTION_NOTE = (
    'Earlier tool interactions were removed to fit the configured context window. '
    'The source workspace retains its current state; reread files if needed and do not repeat edits.'
)
TOOLS = [
    {'type': 'function', 'function': {'name': 'list_files', 'description': 'List exact source paths available in this task workspace. Results are paginated; use next_offset as offset to continue.', 'parameters': {'type': 'object', 'properties': {'offset': {'type': 'integer', 'minimum': 0}}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'read_file', 'description': 'Read one UTF-8 task source file with a hard 12,000-byte and 2,048-token response limit. By default, returns at most 180 lines. Pass inclusive start_line/end_line to inspect an exact range; a completed explicit range has no next_cursor even when later file lines exist. Follow next_cursor only when the returned page was truncated by a line or response limit. The result includes the resolved path and full-file SHA-256.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'start_line': {'type': 'integer', 'minimum': 1}, 'end_line': {'type': 'integer', 'minimum': 1}, 'char_offset': {'type': 'integer', 'minimum': 0}}, 'required': ['path'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'search_text', 'description': 'Search a literal string in allowed task source files. Returns at most 80 matches in a hard 12,000-byte and 2,048-token response; use offset from next_offset to continue.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0}}, 'required': ['query'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'apply_patch', 'description': 'Apply a complete unified diff, not replacement source code. The patch must start with --- a/<exact allowed path> and +++ b/<exact allowed path>, then include @@ hunk headers and -old/+new lines. Keep it minimal; paths must be allowlisted.', 'parameters': {'type': 'object', 'properties': {'patch': {'type': 'string'}}, 'required': ['patch'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'replace_text', 'description': 'Safely replace one exact source snippet in one allowlisted file. Copy old_text exactly from read_file and provide the full new_text. It applies only when old_text occurs exactly once; ambiguous or missing context is rejected. Use this for one focused change; use apply_patch for multi-hunk edits.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}}, 'required': ['path', 'old_text', 'new_text'], 'additionalProperties': False}}},
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
        tools = [
            WRITE_FILE_TOOL if item['function']['name'] == 'apply_patch' else item
            for item in TOOLS if item['function']['name'] != 'replace_text'
        ]
    if public_tests:
        tools.insert(-1, PUBLIC_TEST_TOOL)
    return tools


def generate_turn(backend: str, model: Any, tokenizer: Any, messages: list[dict[str, Any]],
                  tools: list[dict[str, Any]], *, limit: int, args: argparse.Namespace) -> tuple[str, int, int]:
    """Generate one bounded assistant response using a locally loaded backend."""
    prompt = render_chat_prompt(tokenizer, messages, tools)
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


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, Any]],
                       tools: list[dict[str, Any]]) -> str:
    """Render the complete tool history with the model tokenizer's native template."""
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def compact_tool_context(
    messages: list[dict[str, Any]], tokenizer: Any, tools: list[dict[str, Any]], *,
    max_context_tokens: int,
) -> dict[str, Any] | None:
    """Drop oldest complete tool rounds only when the rendered history exceeds its budget.

    The initial system/task messages and the newest complete tool round are retained. The
    full unabridged history remains in the run transcript; this only bounds model context.
    """
    initial_message_count = 2
    if len(messages) <= initial_message_count:
        return None

    def prompt_token_count(current: list[dict[str, Any]]) -> int:
        prompt = render_chat_prompt(tokenizer, current, tools)
        return len(tokenizer.encode(prompt, add_special_tokens=False))

    before_tokens = prompt_token_count(messages)
    if before_tokens < max_context_tokens:
        return None

    compacted = [dict(message) for message in messages]
    removed_rounds: list[dict[str, Any]] = []
    note_present = any(
        message.get('role') == 'user' and message.get('content') == CONTEXT_COMPACTION_NOTE
        for message in compacted[initial_message_count:]
    )

    def complete_tool_rounds() -> list[tuple[int, int, list[str]]]:
        rounds: list[tuple[int, int, list[str]]] = []
        index = initial_message_count
        while index < len(compacted):
            message = compacted[index]
            calls = message.get('tool_calls') if message.get('role') == 'assistant' else None
            if not isinstance(calls, list) or not calls:
                index += 1
                continue
            if not all(
                isinstance(call, dict)
                and isinstance(call.get('id'), str)
                and call['id']
                and isinstance(call.get('function'), dict)
                and isinstance(call['function'].get('name'), str)
                and call['function']['name']
                for call in calls
            ):
                index += 1
                continue
            call_ids = [call['id'] for call in calls]
            expected_ids = set(call_ids)
            if len(expected_ids) != len(call_ids):
                index += 1
                continue
            names = [call['function']['name'] for call in calls]
            end = index + 1
            observed_ids: list[str] = []
            while end < len(compacted) and compacted[end].get('role') == 'tool':
                call_id = compacted[end].get('tool_call_id')
                if not isinstance(call_id, str):
                    break
                observed_ids.append(call_id)
                end += 1
            if (
                expected_ids
                and len(observed_ids) == len(expected_ids)
                and set(observed_ids) == expected_ids
            ):
                rounds.append((index, end, names))
                index = end
            else:
                index += 1
        return rounds

    after_tokens = before_tokens
    while after_tokens >= max_context_tokens:
        rounds = complete_tool_rounds()
        # Keep the newest tool round intact so the model always has its latest observation.
        if len(rounds) <= 1:
            break
        start, end, names = rounds[0]
        del compacted[start:end]
        removed_rounds.append({'tool_names': names, 'tool_result_count': end - start - 1})
        if not note_present:
            compacted.insert(start, {'role': 'user', 'content': CONTEXT_COMPACTION_NOTE})
            note_present = True
        after_tokens = prompt_token_count(compacted)

    if removed_rounds:
        messages[:] = compacted
    return {
        'prompt_tokens_before': before_tokens,
        'prompt_tokens_after': after_tokens,
        'max_context_tokens': max_context_tokens,
        'removed_rounds': removed_rounds,
        'context_fits': after_tokens < max_context_tokens,
    }


def append_tool_interaction(
    messages: list[dict[str, Any]], calls_and_results: list[tuple[str, dict[str, Any], Any]],
) -> None:
    """Append paired assistant tool calls and matching tool-result messages."""
    first_call = len(messages)
    messages.append({'role': 'assistant', 'tool_calls': [
        {
            'id': f'call_{first_call}_{index}', 'type': 'function',
            'function': {'name': name, 'arguments': arguments},
        }
        for index, (name, arguments, _result) in enumerate(calls_and_results)
    ]})
    for index, (name, _arguments, result) in enumerate(calls_and_results):
        messages.append({
            'role': 'tool', 'name': name, 'tool_call_id': f'call_{first_call}_{index}',
            'content': json.dumps(result, ensure_ascii=False),
        })


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
        result = {'applied': False, 'format_parse_failure': reason}
        if edit_format == 'unified_diff' and reason == 'malformed_diff':
            path = sorted(allowed)[0] if allowed else '<allowed path>'
            result['error'] = (
                'The patch was not a unified diff. Send a complete diff, not replacement code, '
                'with these exact headers and hunk form (replace the example lines with exact source context): '
                f'--- a/{path}\n+++ b/{path}\n@@ -10,1 +10,1 @@\n-old line\n+new line'
            )
        return result
    applied = apply_parsed_edit(workspace, parsed, allowed)
    if not applied.applied:
        return {
            'applied': False,
            'format_parse_failure': None,
            'error': applied.error or 'edit was not applied',
            'paths': list(applied.paths),
        }
    return {'applied': True, 'paths': list(applied.paths), 'format_parse_failure': None}


def attempt_exact_text_replacement(
    workspace: Path, path: str, old_text: str, new_text: str, allowed: set[str],
    *, max_snippet_chars: int = 6_000,
) -> dict[str, Any]:
    """Apply an exact, unique snippet replacement through the normal diff guard."""
    resolved = resolve_allowed_path(path, allowed)
    if resolved is None:
        return {'applied': False, 'error': 'path is not an exact allowlisted source file'}
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        return {'applied': False, 'error': 'old_text and new_text must be strings'}
    if not old_text:
        return {'applied': False, 'error': 'old_text must not be empty'}
    if len(old_text) > max_snippet_chars or len(new_text) > max_snippet_chars:
        return {'applied': False, 'error': f'snippets exceed the {max_snippet_chars}-character limit'}
    try:
        original = read_regular_source(workspace, resolved).decode('utf-8')
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {'applied': False, 'error': f'could not read allowlisted source: {type(exc).__name__}'}
    count = original.count(old_text)
    if count != 1:
        return {
            'applied': False,
            'error': 'old_text must match exactly once; narrow the source context and retry',
            'match_count': count,
        }
    updated = original.replace(old_text, new_text, 1)
    patch = ''.join(difflib.unified_diff(
        original.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile=f'a/{resolved}', tofile=f'b/{resolved}', n=3,
    ))
    if not patch:
        return {'applied': False, 'error': 'replacement does not change the source'}
    result = attempt_configured_edit(workspace, patch, 'unified_diff', allowed)
    result['edit_mode'] = 'exact_text_replacement'
    return result


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
    diff = diff_bytes.decode('utf-8')
    if diff and not diff.endswith('\n'):
        diff += '\n'
    if len(diff.encode('utf-8')) > 2 * 1024 * 1024:
        raise ValueError('candidate diff exceeds the evidence parser limit')
    return diff


def resolve_allowed_path(raw_path: str, allowed: set[str]) -> str | None:
    """Map a model-supplied path to one exact allowlisted relative file."""
    normalized = raw_path.replace('\\', '/').lstrip('./')
    matches = [path for path in allowed if normalized == path or normalized.endswith('/' + path)]
    return matches[0] if len(matches) == 1 else None


def _tool_result_fits(result: Any, tokenizer: Any | None, *,
                      max_bytes: int, max_tokens: int) -> bool:
    serialized = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
    if len(serialized.encode('utf-8')) > max_bytes:
        return False
    return tokenizer is None or len(tokenizer.encode(
        serialized, add_special_tokens=False,
    )) <= max_tokens


def bounded_source_page(
    text: str, *, start_line: int = 1, end_line: int | None = None,
    char_offset: int = 0, source_identity: dict[str, Any] | None = None,
    tokenizer: Any | None = None, max_output_bytes: int = MAX_TOOL_RESULT_BYTES,
    max_output_tokens: int = MAX_TOOL_RESULT_TOKENS,
) -> dict[str, Any]:
    """Return an inclusive source page under hard serialized-byte/token limits.

    A very long individual line may be returned as a character slice. Its
    ``next_cursor`` points to the exact source line and character offset so
    the agent can continue without losing line identity.
    """
    if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
        raise ValueError('start_line must be a positive integer')
    if end_line is not None and (
        isinstance(end_line, bool) or not isinstance(end_line, int) or end_line < 1
    ):
        raise ValueError('end_line must be a positive integer')
    if end_line is not None and end_line < start_line:
        raise ValueError('end_line must be greater than or equal to start_line')
    if isinstance(char_offset, bool) or not isinstance(char_offset, int) or char_offset < 0:
        raise ValueError('char_offset must be a non-negative integer')
    if (type(max_output_bytes) is not int or max_output_bytes <= 0
            or type(max_output_tokens) is not int or max_output_tokens <= 0):
        raise ValueError('tool response budgets must be positive integers')

    lines = text.splitlines(keepends=True)
    total_lines = len(lines)
    if start_line > max(total_lines, 1):
        raise ValueError(f'start_line exceeds the source length ({total_lines} lines)')
    if total_lines == 0 and char_offset != 0:
        raise ValueError('char_offset exceeds the source length (0 characters)')
    if total_lines and char_offset > len(lines[start_line - 1]):
        raise ValueError('char_offset exceeds the selected source line')
    source_range_end = min(end_line, total_lines) if end_line is not None else total_lines
    page_limit = MAX_READ_LINES if end_line is not None else DEFAULT_READ_LINES
    returned_end = min(source_range_end, start_line + page_limit - 1)
    identity = dict(source_identity or {})
    identity.setdefault('source_sha256', hashlib.sha256(text.encode('utf-8')).hexdigest())

    def next_after(line_number: int) -> dict[str, int] | None:
        if line_number < source_range_end:
            return {'start_line': line_number + 1, 'char_offset': 0}
        return None

    def make_result(content: str, last_line: int,
                    cursor: dict[str, int] | None) -> dict[str, Any]:
        return {
            **identity,
            'content': content,
            'start_line': start_line,
            'end_line': last_line,
            'total_lines': total_lines,
            'has_more': cursor is not None,
            'next_cursor': cursor,
        }

    empty_cursor = ({'start_line': start_line, 'char_offset': char_offset}
                    if start_line <= returned_end else None)
    result = make_result('', start_line - 1, empty_cursor)
    if not _tool_result_fits(
        result, tokenizer, max_bytes=max_output_bytes, max_tokens=max_output_tokens,
    ):
        raise ValueError('source identity metadata exceeds the tool response budget')

    content = ''
    last_line = start_line - 1
    for line_number in range(start_line, returned_end + 1):
        offset = char_offset if line_number == start_line else 0
        piece = lines[line_number - 1][offset:]
        cursor = next_after(line_number)
        candidate = make_result(content + piece, line_number, cursor)
        if _tool_result_fits(
            candidate, tokenizer, max_bytes=max_output_bytes, max_tokens=max_output_tokens,
        ):
            content += piece
            last_line = line_number
            continue

        low, high = 0, len(piece)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            partial_cursor = {'start_line': line_number, 'char_offset': offset + middle}
            partial = make_result(content + piece[:middle], line_number, partial_cursor)
            if _tool_result_fits(
                partial, tokenizer, max_bytes=max_output_bytes,
                max_tokens=max_output_tokens,
            ):
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best:
            content += piece[:best]
            last_line = line_number
            return make_result(
                content, last_line,
                {'start_line': line_number, 'char_offset': offset + best},
            )
        if content:
            return make_result(content, last_line, {'start_line': line_number, 'char_offset': offset})
        raise ValueError('tool response budget is too small to return source content')

    cursor = next_after(last_line) if last_line >= start_line else None
    result = make_result(content, last_line, cursor)
    if not _tool_result_fits(
        result, tokenizer, max_bytes=max_output_bytes, max_tokens=max_output_tokens,
    ):
        raise AssertionError('bounded source page exceeded its response budget')
    return result


def bounded_search_results(
    matches: list[dict[str, Any]], *, offset: int = 0, tokenizer: Any | None = None,
    max_matches: int = MAX_SEARCH_MATCHES, max_output_bytes: int = MAX_TOOL_RESULT_BYTES,
    max_output_tokens: int = MAX_TOOL_RESULT_TOKENS,
) -> dict[str, Any]:
    """Return stable search matches with pagination and hard response budgets."""
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError('offset must be a non-negative integer')
    if type(max_matches) is not int or max_matches <= 0:
        raise ValueError('max_matches must be a positive integer')
    selected = matches[offset:offset + max_matches]
    result: dict[str, Any] = {
        'matches': [], 'offset': offset, 'has_more': offset + len(selected) < len(matches),
        'next_offset': offset if not selected else offset + len(selected),
    }
    for index, original in enumerate(selected):
        item = dict(original)
        full_text = str(item.get('text', ''))
        item['text'] = full_text[:MAX_SEARCH_SNIPPET_CHARS]
        item['text_truncated'] = len(full_text) > len(item['text'])
        candidate_matches = [*result['matches'], item]
        next_offset = offset + len(candidate_matches)
        candidate = {
            **result,
            'matches': candidate_matches,
            'has_more': next_offset < len(matches),
            'next_offset': next_offset if next_offset < len(matches) else None,
        }
        if _tool_result_fits(
            candidate, tokenizer, max_bytes=max_output_bytes, max_tokens=max_output_tokens,
        ):
            result = candidate
            continue

        low, high = 0, len(item['text'])
        best = -1
        while low <= high:
            middle = (low + high) // 2
            shortened = {**item, 'text': item['text'][:middle], 'text_truncated': True}
            shortened_result = {
                **result,
                'matches': [*result['matches'], shortened],
                'has_more': True,
                'next_offset': offset + len(result['matches']) + 1,
            }
            if _tool_result_fits(
                shortened_result, tokenizer, max_bytes=max_output_bytes,
                max_tokens=max_output_tokens,
            ):
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best >= 0:
            shortened = {**item, 'text': item['text'][:best], 'text_truncated': True}
            result = {
                **result,
                'matches': [*result['matches'], shortened],
                'has_more': True,
                'next_offset': offset + len(result['matches']) + 1,
            }
        else:
            result = {
                **result, 'has_more': True,
                'next_offset': offset + len(result['matches']),
            }
        return result
    if result['next_offset'] == offset + len(matches[offset:]):
        result['next_offset'] = None
    if not _tool_result_fits(
        result, tokenizer, max_bytes=max_output_bytes, max_tokens=max_output_tokens,
    ):
        raise AssertionError('bounded search result exceeded its response budget')
    return result


def bounded_list_page(
    items: list[str], *, offset: int = 0, tokenizer: Any | None = None,
    max_items: int = 100, max_output_bytes: int = MAX_TOOL_RESULT_BYTES,
    max_output_tokens: int = MAX_TOOL_RESULT_TOKENS,
) -> dict[str, Any]:
    """Return a paginated path list under the same hard tool-response limits."""
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError('offset must be a non-negative integer')
    if type(max_items) is not int or max_items <= 0:
        raise ValueError('max_items must be a positive integer')
    selected = items[offset:offset + max_items]
    result: dict[str, Any] = {
        'files': [], 'offset': offset, 'has_more': offset < len(items),
        'next_offset': offset if not selected else offset + len(selected),
    }
    for item in selected:
        next_items = [*result['files'], item]
        next_offset = offset + len(next_items)
        candidate = {
            **result, 'files': next_items,
            'has_more': next_offset < len(items),
            'next_offset': next_offset if next_offset < len(items) else None,
        }
        if not _tool_result_fits(
            candidate, tokenizer, max_bytes=max_output_bytes, max_tokens=max_output_tokens,
        ):
            if result['files']:
                return {
                    **result, 'has_more': True,
                    'next_offset': offset + len(result['files']),
                }
            raise ValueError('tool response budget is too small to return a source path')
        result = candidate
    if not selected or result['next_offset'] == len(items):
        result['next_offset'] = None
    if not _tool_result_fits(
        result, tokenizer, max_bytes=max_output_bytes, max_tokens=max_output_tokens,
    ):
        raise AssertionError('bounded path list exceeded its response budget')
    return result


def bounded_tool_result(result: Any, tokenizer: Any) -> Any:
    """Apply a final hard cap to non-source tool outputs before adding history."""
    if _tool_result_fits(
        result, tokenizer, max_bytes=MAX_TOOL_RESULT_BYTES,
        max_tokens=MAX_TOOL_RESULT_TOKENS,
    ):
        return result
    serialized = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
    digest = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
    clipped: Any = result
    if isinstance(result, dict):
        clipped = dict(result)
        for field in ('stdout', 'stderr', 'error', 'summary', 'text', 'content'):
            value = clipped.get(field)
            if isinstance(value, str):
                clipped[field] = value[:2_000]
                clipped[f'{field}_truncated'] = len(value) > 2_000
        for field, value in tuple(clipped.items()):
            if isinstance(value, list) and len(value) > 20:
                clipped[field] = value[:20]
                clipped[f'{field}_truncated'] = True
    clipped_with_metadata = {
        **clipped, 'response_truncated': True, 'original_result_sha256': digest,
    } if isinstance(clipped, dict) else clipped
    if _tool_result_fits(
        clipped_with_metadata, tokenizer, max_bytes=MAX_TOOL_RESULT_BYTES,
        max_tokens=MAX_TOOL_RESULT_TOKENS,
    ):
        return clipped_with_metadata
    fallback = {
        'error': 'tool result exceeded the response budget; use a narrower lookup',
        'response_truncated': True,
        'original_result_sha256': digest,
    }
    if not _tool_result_fits(
        fallback, tokenizer, max_bytes=MAX_TOOL_RESULT_BYTES,
        max_tokens=MAX_TOOL_RESULT_TOKENS,
    ):
        raise ValueError('tool tokenizer cannot encode the bounded error response')
    return fallback


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
            'Read the source path named in the user instruction, then use tools to inspect and edit. For one focused change, prefer replace_text: copy an exact old_text snippet from the source and provide its full new_text. It succeeds only when old_text matches exactly once. Use apply_patch for multi-hunk changes; never rewrite a whole file. '
            'The apply_patch patch argument must be a complete unified diff, not replacement code. Use exact headers and hunk syntax, with no Markdown fence: '
            '--- a/<exact allowed path> newline +++ b/<same exact path> newline @@ -10,1 +10,1 @@ newline -old line newline +new line. '
            'Replace example lines with exact source context; do not send only the replacement function body. '
        )
    system = (
        'You are an offline code repair agent. Solve only the task below. The workspace contains only the task source files. '
        + edit_sentence
        + 'Allowed source paths: ' + ', '.join(sorted(editable)) + '. First call list_files, then read the exact source path named in the task. '
        + f'read_file and search_text results have hard {MAX_TOOL_RESULT_BYTES}-byte and {MAX_TOOL_RESULT_TOKENS}-token limits. '
        + 'read_file returns at most 180 lines by default and 200 lines per call; follow next_cursor.start_line and next_cursor.char_offset exactly, including when a source line is unusually long. '
        + 'The full-file SHA-256 and resolved path identify the source used for each page. Search first, then read only the matching line range and nearby context. '
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
    context_compaction_events: list[dict[str, Any]] = []
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
        context_compaction = compact_tool_context(
            messages, tokenizer, tools, max_context_tokens=args.max_context_tokens,
        )
        if context_compaction is not None:
            context_compaction['turn'] = turn + 1
            context_compaction_events.append(context_compaction)
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
        if context_compaction is not None:
            entry['context_compaction'] = context_compaction
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
        if not parsed_calls:
            messages.append({'role': 'assistant', 'content': raw})
        calls_and_results: list[tuple[str, dict[str, Any], Any]] = []
        for match, name, arguments in parsed_calls:
            try:
                if name == 'list_files':
                    result: Any = bounded_list_page(
                        sorted(allowed), offset=arguments.get('offset', 0), tokenizer=tokenizer,
                    )
                elif name == 'read_file':
                    raw_path = arguments.get('path', '')
                    path = resolve_allowed_path(raw_path, allowed)
                    if path is None:
                        result = {'error': 'path is not allowed; use list_files to retrieve exact allowed paths'}
                    else:
                        source = read_regular_source(workspace, path)
                        text = source.decode('utf-8')
                        result = bounded_source_page(
                            text,
                            start_line=arguments.get('start_line', 1),
                            end_line=arguments.get('end_line'),
                            char_offset=arguments.get('char_offset', 0),
                            source_identity={
                                'resolved_path': path,
                                'source_sha256': hashlib.sha256(source).hexdigest(),
                            },
                            tokenizer=tokenizer,
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
                                    matches.append({'path': path, 'line': line_no, 'text': line})
                        result = bounded_search_results(
                            matches, offset=arguments.get('offset', 0), tokenizer=tokenizer,
                        )
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
                elif name == 'replace_text':
                    result = attempt_exact_text_replacement(
                        workspace,
                        arguments.get('path', ''),
                        arguments.get('old_text', ''),
                        arguments.get('new_text', ''),
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
                result = bounded_tool_result(result, tokenizer)
                entry['tool_calls'].append({'name': name, 'arguments': arguments, 'result': result})
                calls_and_results.append((name, arguments, result))
                if name == 'finish':
                    break
            except Exception as exc:
                result = bounded_tool_result(
                    {'error': f'{type(exc).__name__}: {exc}'}, tokenizer,
                )
                entry['tool_calls'].append({
                    'name': name, 'arguments': arguments, 'result': result,
                })
                calls_and_results.append((name, arguments, result))
        if calls_and_results:
            append_tool_interaction(messages, calls_and_results)
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
        'context_compaction_events': context_compaction_events,
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
