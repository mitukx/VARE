#!/usr/bin/env python3
"""Tokenizer-only preflight for the v9/v10 multi-turn benchmark (no weights, no generation).

Builds scripted 8- and 16-turn histories with the real deterministic tool
executor, then checks with the cached chat template and tokenizer that every
turn's shared prefix is an exact token prefix of a tool-choice stage and an
argument-slot stage prompt, and that prompts stay inside the context budget.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from transformers import AutoTokenizer

from scripts.run_local_agent import tools_for
from scripts.run_v10_multiturn_worker_v1 import (
    TOOL_INTERFACE, SyntheticWorkspace, initial_messages, tools_for_turn,
)

SCRIPTED_EDITS = {
    'inventory': [('levels[item] + amount', 'levels.get(item, 0) + amount'), ('total = count', 'total += count')],
    'textstats': [('text.split(" ")', 'text.split()'), ('(len(words) + 1)', 'len(words)')],
    'intervals': [('first[0] < second[1] and second[0] < first[1]',
                   'first[0] <= second[1] and second[0] <= first[1]'),
                  ('in intervals:', 'in sorted(intervals):')],
}


def scripted_actions(variant: dict, horizon: int) -> list[tuple[str, dict]]:
    lines = len(variant['source'].splitlines())
    path = variant['path']
    edits = SCRIPTED_EDITS[variant['variant_id']]
    cycle = [('list_files', {}), ('search_source', {'query': 'return'}), ('read_lines', {'path': path, 'start': 1, 'end': lines}),
             ('edit', 0), ('check_syntax', {'path': path}), ('run_public_tests', {}),
             ('read_lines', {'path': path, 'start': 1, 'end': lines}), ('edit', 1), ('run_public_tests', {})]
    actions = []
    while len(actions) < horizon - 1:
        actions.extend(cycle)
    actions = actions[:horizon - 1]
    resolved = []
    for name, payload in actions:
        if name == 'edit':
            expected, replacement = edits[payload]
            resolved.append(('replace_exact_text_transactional', {'expected_text': expected, 'replacement_text': replacement}))
        else:
            resolved.append((name, payload))
    resolved.append(('finish', {'summary': 'scripted preflight'}))
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-context-tokens', type=int, default=8192)
    parser.add_argument('--turn-token-budget', type=int, default=256)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    fixtures = json.loads(args.fixtures.read_text(encoding='utf-8'))
    all_tools = tools_for('structured_replace', public_tests=True, tool_interface=TOOL_INTERFACE)

    def ids_for(text: str) -> list[int]:
        bos = getattr(tokenizer, 'bos_token', None)
        add_special = bos is None or not text.startswith(bos)
        return list(tokenizer.encode(text, add_special_tokens=add_special))

    def render(messages, tools, generation):
        return tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                             add_generation_prompt=generation, enable_thinking=False)

    records = []
    for variant in fixtures['variants']:
        for horizon in (8, 16):
            with tempfile.TemporaryDirectory() as temp:
                workspace = SyntheticWorkspace(Path(temp), variant)
                messages = initial_messages(variant, horizon)
                lease = None
                for turn, (name, arguments) in enumerate(scripted_actions(variant, horizon), start=1):
                    tools = tools_for_turn(all_tools, turn, horizon)
                    prefix_ids = ids_for(render(messages, tools, False))
                    choice = messages + [{'role': 'user', 'content': 'Choose exactly one tool name.'}]
                    slot = choice + [{'role': 'assistant', 'content': name},
                                     {'role': 'user', 'content': 'Generate the value for the next argument.'}]
                    stage_ids = [ids_for(render(choice, tools, True)), ids_for(render(slot, tools, True))]
                    exact = [len(ids) > len(prefix_ids) and ids[:len(prefix_ids)] == prefix_ids for ids in stage_ids]
                    if name == 'replace_exact_text_transactional':
                        arguments = {'lease': lease, **arguments}
                    result = workspace.execute(name, arguments)
                    if name == 'read_lines' and isinstance(result, dict):
                        lease = result.get('lease')
                    records.append({
                        'variant': variant['variant_id'], 'horizon': horizon, 'turn': turn, 'action': name,
                        'prefix_tokens': len(prefix_ids), 'max_stage_tokens': max(len(ids) for ids in stage_ids),
                        'all_stage_prefixes_exact': all(exact),
                        'tool_status': (result.get('status') if isinstance(result, dict) else 'list'),
                        'edit_applied': result.get('applied') if isinstance(result, dict) else None,
                    })
                    messages = messages + [
                        {'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': {'name': name, 'arguments': arguments}}]},
                        {'role': 'tool', 'name': name, 'content': json.dumps(result, ensure_ascii=False)},
                    ]
                records.append({'variant': variant['variant_id'], 'horizon': horizon, 'turn': 'final',
                                'scripted_final_source_changed': workspace.source_text() != variant['source']})
    turn_records = [item for item in records if item['turn'] != 'final']
    max_tokens = max(item['max_stage_tokens'] for item in turn_records)
    summary = {
        'note': 'Tokenizer and deterministic tool executor only; no model weights loaded and no generation.',
        'all_prefixes_exact': all(item['all_stage_prefixes_exact'] for item in turn_records),
        'max_scripted_stage_prompt_tokens': max_tokens,
        'context_headroom_tokens': args.max_context_tokens - max_tokens - args.turn_token_budget,
        'records': records,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=1, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({key: summary[key] for key in ('all_prefixes_exact', 'max_scripted_stage_prompt_tokens',
                                                    'context_headroom_tokens')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
