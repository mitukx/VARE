#!/usr/bin/env python3
"""Replay saved agent transcripts through context compaction without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from scripts.run_local_agent import (
    append_tool_interaction,
    compact_tool_context,
    no_tool_retry_reminder,
    render_chat_prompt,
    tools_for,
)


def reconstruct_attempt_messages(
    attempt: dict[str, Any], tokenizer: Any, tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """Rebuild the unabridged prompt history and verify every recorded prompt length."""
    if attempt.get('edit_attempted') or attempt.get('edit_applied'):
        raise ValueError('retrospective replay only supports attempts without edit actions')
    messages = [dict(message) for message in attempt['initial_messages']]
    if len(messages) != 2 or messages[0].get('role') != 'system' or messages[1].get('role') != 'user':
        raise ValueError('attempt must contain the original system and task messages')
    match = re.search(r'Allowed source paths: (.*?)\. First call', messages[0].get('content', ''))
    if match is None:
        raise ValueError('could not recover allowed source paths from the frozen system prompt')
    allowed_paths = match.group(1).split(', ')
    budgets = attempt['budgets']
    no_tool_retries = attempt['no_tool_retries']
    no_tool_turns = 0
    prompt_checks: list[dict[str, int]] = []

    for turn in attempt['transcript']:
        rendered = render_chat_prompt(tokenizer, messages, tools)
        reconstructed_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
        recorded_tokens = turn['prompt_tokens']
        if reconstructed_tokens != recorded_tokens:
            raise ValueError(
                f"turn {turn['turn']} prompt mismatch: reconstructed {reconstructed_tokens}, "
                f"recorded {recorded_tokens}"
            )
        prompt_checks.append({
            'turn': turn['turn'],
            'reconstructed_tokens': reconstructed_tokens,
            'recorded_tokens': recorded_tokens,
        })

        calls = turn.get('tool_calls', [])
        if calls:
            calls_and_results = []
            for call in calls:
                if not all(key in call for key in ('name', 'arguments', 'result')):
                    raise ValueError(f"turn {turn['turn']} contains an unsupported tool record")
                calls_and_results.append((call['name'], call['arguments'], call['result']))
            append_tool_interaction(messages, calls_and_results)
            continue

        messages.append({'role': 'assistant', 'content': turn['model_output']})
        no_tool_turns += 1
        if no_tool_turns <= no_tool_retries:
            messages.append({
                'role': 'user',
                'content': no_tool_retry_reminder(
                    turn['model_output'], edit_format=attempt['edit_format'],
                    allowed_paths=allowed_paths,
                    token_limit=budgets['max_new_tokens_per_turn'],
                ),
            })

    if no_tool_turns not in (no_tool_retries, no_tool_retries + 1):
        raise ValueError('saved no-tool retry count is inconsistent with the transcript')
    return messages, prompt_checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True, help='saved run_local_agent JSON record')
    parser.add_argument('--model-path', type=Path, required=True, help='local tokenizer snapshot path')
    parser.add_argument('--output', type=Path, required=True, help='new retrospective replay JSON path')
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'refusing to overwrite retrospective replay: {args.output}')

    attempt_bytes = args.attempt.read_bytes()
    attempt = json.loads(attempt_bytes)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tools = tools_for(attempt['edit_format'])
    messages, prompt_checks = reconstruct_attempt_messages(attempt, tokenizer, tools)
    before_tokens = len(tokenizer.encode(
        render_chat_prompt(tokenizer, messages, tools), add_special_tokens=False,
    ))
    event = compact_tool_context(
        messages, tokenizer, tools,
        max_context_tokens=attempt['budgets']['max_context_tokens'],
    )
    after_tokens = len(tokenizer.encode(
        render_chat_prompt(tokenizer, messages, tools), add_special_tokens=False,
    ))
    record = {
        'schema_version': 1,
        'analysis_type': 'saved_transcript_context_compaction_replay',
        'task_id': attempt['task_id'],
        'attempt_sha256': hashlib.sha256(attempt_bytes).hexdigest(),
        'model_repository': attempt['model_repository'],
        'model_snapshot': attempt['model_snapshot'],
        'tokenizer_snapshot': attempt['model_snapshot'],
        'max_context_tokens': attempt['budgets']['max_context_tokens'],
        'prompt_checks': prompt_checks,
        'prompt_tokens_before_compaction': before_tokens,
        'compaction_event': event,
        'prompt_tokens_after_compaction': after_tokens,
        'model_generations': 0,
        'grader_calls': 0,
        'claim_limit': 'Offline replay of saved messages only; no model capability or repair outcome is measured.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
