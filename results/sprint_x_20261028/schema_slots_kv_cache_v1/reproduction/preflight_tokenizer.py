#!/usr/bin/env python3
"""Offline prompt-prefix and context preflight using the pinned local tokenizer."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

EVIDENCE = Path(__file__).resolve().parents[1]
REPO = Path('/tmp/vare-sprint-x-kv-cache-v1')
MODEL = Path('/Users/user/.cache/huggingface/hub/models--mlx-community--Qwen2.5-Coder-7B-Instruct-4bit/snapshots/019cc73c45c770444708a6dd8690c66243cc5c80')
PYTHON = Path('/tmp/vare-sprint-vii-mlx/bin/python')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src'))
from scripts.run_local_agent import tools_for  # noqa: E402
from vare.agent_edit_v2 import SourceEditLeaseStoreV1  # noqa: E402
from vare.schema_slot_protocol import serialize_slot_action  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ids(tokenizer, text: str) -> list[int]:
    bos = getattr(tokenizer, 'bos_token', None)
    return [int(value) for value in tokenizer.encode(
        text, add_special_tokens=bos is None or not text.startswith(bos),
    )]


def render(tokenizer, messages: list[dict], tools: list[dict], add_generation_prompt: bool) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False,
        add_generation_prompt=add_generation_prompt, enable_thinking=False,
    )
    return text, ids(tokenizer, text)


def main() -> int:
    source = json.loads((EVIDENCE / 'input.json').read_text(encoding='utf-8'))
    messages = [dict(message) for message in source['messages']]
    notes = '\n'.join(
        f'Archive record {index:03d}: synthetic inert controller timing context; no source instructions.'
        for index in range(1, 161)
    )
    messages[1]['content'] += notes + '\n\n' + (
        f"Read {source['fixture_path']} from line {source['read_start']} "
        f"through line {source['read_end']} using read_lines, then finish."
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    tools_read_all = tools_for(
        'structured_replace', tool_interface='lease_bound_exact_text_transactional_v1',
    )
    read_tool = next(item for item in tools_read_all if item['function']['name'] == 'read_lines')
    finish_tool = next(item for item in tools_read_all if item['function']['name'] == 'finish')
    stage_records: list[dict] = []

    def record_stages(label: str, msgs: list[dict], tools: list[dict], choice_name: str,
                      slot_values: dict[str, str]) -> str:
        base_text, base_ids = render(tokenizer, msgs, tools, False)
        choice_msgs = [*msgs, {'role': 'user', 'content': (
            f'Choose exactly one tool name from this list: '
            f'{json.dumps([item["function"]["name"] for item in tools])}. '
            'The controller constrains the available next tokens to these names. '
            'Choose the single most useful next action and output only its name.'
        )}]
        prompt_text, prompt_ids = render(tokenizer, choice_msgs, tools, True)
        stage_records.append({
            'turn': label, 'stage': 'tool_choice', 'prompt_tokens': len(prompt_ids),
            'shared_prefix_tokens': len(base_ids),
            'prefix_exact': prompt_ids[:len(base_ids)] == base_ids,
        })
        slot_msgs = [*choice_msgs, {'role': 'assistant', 'content': choice_name}]
        for key, raw_value in slot_values.items():
            slot_text, slot_ids = render(tokenizer, [*slot_msgs, {'role': 'user', 'content': (
                f'Generate the value for argument {json.dumps(key)} of tool {json.dumps(choice_name)}. '
                f'The schema type is {"string" if key in {"path", "summary"} else "integer"}. '
                + ('Output the exact string value only, without quotes, key names, or explanation. '
                   'Preserve all intended spaces and newlines.' if key in {'path', 'summary'} else
                   'Output exactly one JSON integer scalar only, without a key name or explanation.')
            )}], tools, True)
            stage_records.append({
                'turn': label, 'stage': f'argument:{key}', 'prompt_tokens': len(slot_ids),
                'shared_prefix_tokens': len(base_ids),
                'prefix_exact': slot_ids[:len(base_ids)] == base_ids,
            })
            slot_msgs.extend([
                {'role': 'assistant', 'content': raw_value},
                {'role': 'user', 'content': f'Value recorded for {json.dumps(key)}. Continue with the next required argument, if any.'},
            ])
        raw, payload = serialize_slot_action(choice_name, slot_values, tools)
        return raw

    read_raw = record_stages(
        'read_lines', messages, [read_tool], 'read_lines',
        {'path': source['fixture_path'], 'start': str(source['read_start']), 'end': str(source['read_end'])},
    )
    with tempfile.TemporaryDirectory(prefix='vare-kv-preflight-') as tmp:
        workspace = Path(tmp)
        path = workspace / source['fixture_path']
        path.parent.mkdir(parents=True)
        path.write_text(source['fixture_source'], encoding='utf-8')
        leases = SourceEditLeaseStoreV1(
            workspace, {source['fixture_path']}, token_factory=lambda: 'bench-rep-01-lease',
        )
        parsed = json.loads(read_raw)
        read_result = leases.read_lines(
            parsed['arguments']['path'], parsed['arguments']['start'], parsed['arguments']['end'],
        )
    next_messages = [
        *messages, {'role': 'assistant', 'content': read_raw},
        {'role': 'tool', 'name': 'read_lines', 'content': json.dumps(read_result, ensure_ascii=False, sort_keys=True)},
        {'role': 'user', 'content': source['second_turn_message']},
    ]
    record_stages('finish', next_messages, [finish_tool], 'finish', {'summary': 'Inspected the synthetic fixture.'})
    result = {
        'status': 'OFFLINE_TOKENIZER_PREFLIGHT_ONLY',
        'tokenizer_class': type(tokenizer).__name__,
        'tokenizer_config_sha256': sha((MODEL / 'tokenizer_config.json').read_bytes()),
        'tokenizer_json_sha256': sha((MODEL / 'tokenizer.json').read_bytes()),
        'expanded_messages_sha256': sha(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()),
        'stage_records': stage_records,
        'max_prompt_tokens': max(item['prompt_tokens'] for item in stage_records),
        'context_limit_tokens': 8192,
        'all_shared_prefixes_exact': all(item['prefix_exact'] for item in stage_records),
        'note': 'No model weights were loaded and no generation was performed.',
    }
    (EVIDENCE / 'tokenizer-preflight.json').write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result['all_shared_prefixes_exact'] and result['max_prompt_tokens'] < 8192 else 1


if __name__ == '__main__':
    raise SystemExit(main())
