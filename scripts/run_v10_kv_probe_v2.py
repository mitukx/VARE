#!/usr/bin/env python3
"""Real-model KV restore probe, v2 (protocol amendment 001).

The v1 probe in run_v10_multiturn_worker_v1.py converted every cache state
element with mx.array, including KVCache's integer offset in mlx-lm 0.32.x,
and crashed before producing any comparison. This v2 probe clones state the
same way the schema-slot controller does (arrays copied, scalars kept) and is
otherwise identical: chunked full prefill vs restored prefix + suffix on real
prompts, plus a stale-prefix negative control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.run_v10_multiturn_worker_v1 import (
    TOOL_INTERFACE, _version, initial_messages, tools_for_turn, write_json_exclusive,
)


def clone_state(mx: Any, value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(clone_state(mx, item) for item in value)
    if isinstance(value, list):
        return [clone_state(mx, item) for item in value]
    if isinstance(value, dict):
        return {key: clone_state(mx, item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return mx.array(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from scripts.run_local_agent import tools_for

    fixtures = json.loads(args.fixtures.read_text(encoding='utf-8'))
    all_tools = tools_for('structured_replace', public_tests=True, tool_interface=TOOL_INTERFACE)
    tools = tools_for_turn(all_tools, 1, 8)
    model, tokenizer = load(str(args.model))

    def ids_for(text: str) -> list[int]:
        bos = getattr(tokenizer, 'bos_token', None)
        add_special = bos is None or not text.startswith(bos)
        return [int(token) for token in tokenizer.encode(text, add_special_tokens=add_special)]

    def last_logits_full(ids: list[int]) -> Any:
        cache = make_prompt_cache(model)
        logits = None
        for start in range(0, len(ids), 2048):
            logits = model(mx.array([ids[start:start + 2048]]), cache=cache)
            mx.eval(logits)
        return logits[0, -1].astype(mx.float32)

    def last_logits_restored(prefix_ids: list[int], suffix_ids: list[int]) -> Any:
        prefix_cache = make_prompt_cache(model)
        mx.eval(model(mx.array([prefix_ids]), cache=prefix_cache))
        states = [clone_state(mx, cache.state) for cache in prefix_cache]
        restored = make_prompt_cache(model)
        for cache, state in zip(restored, states):
            cache.state = clone_state(mx, state)
        logits = model(mx.array([suffix_ids]), cache=restored)
        mx.eval(logits)
        return logits[0, -1].astype(mx.float32)

    def compare(a: Any, b: Any) -> dict[str, Any]:
        diff = mx.abs(a - b)
        top_a = [int(x) for x in mx.argsort(-a)[:5].tolist()]
        top_b = [int(x) for x in mx.argsort(-b)[:5].tolist()]
        return {'max_abs_logit_diff': float(mx.max(diff).item()),
                'mean_abs_logit_diff': float(mx.mean(diff).item()),
                'argmax_equal': top_a[0] == top_b[0], 'top5_overlap': len(set(top_a) & set(top_b))}

    choice_suffix = {'role': 'user', 'content': 'Choose exactly one tool name from the list. Output only its name.'}
    probes = []
    for variant in fixtures['variants']:
        history = initial_messages(variant, 8)
        read = {'path': variant['path'], 'start': 1, 'end': len(variant['source'].splitlines())}
        content = variant['source']
        tool_result = {'status': 'read_complete', 'path': variant['path'], 'start_line': 1,
                       'end_line': read['end'], 'content': content, 'lease': 'lease-001'}
        history_with_read = history + [
            {'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': {'name': 'read_lines', 'arguments': read}}]},
            {'role': 'tool', 'name': 'read_lines', 'content': json.dumps(tool_result, ensure_ascii=False)},
        ]
        altered = dict(tool_result, content=content.replace('return', 'yield', 1))
        altered_history = history_with_read[:-1] + [
            {'role': 'tool', 'name': 'read_lines', 'content': json.dumps(altered, ensure_ascii=False)},
        ]
        for label, msgs, wrong_msgs in (('turn1', history, None), ('after_read', history_with_read, altered_history)):
            prefix_text = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False,
                                                        add_generation_prompt=False, enable_thinking=False)
            stage_text = tokenizer.apply_chat_template(msgs + [choice_suffix], tools=tools, tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
            prefix_ids, stage_ids = ids_for(prefix_text), ids_for(stage_text)
            exact = len(stage_ids) > len(prefix_ids) and stage_ids[:len(prefix_ids)] == prefix_ids
            full = last_logits_full(stage_ids)
            entry: dict[str, Any] = {'variant': variant['variant_id'], 'probe': label,
                                     'prefix_tokens': len(prefix_ids), 'stage_tokens': len(stage_ids),
                                     'token_prefix_exact': exact}
            if exact:
                entry['restored_vs_full'] = compare(full, last_logits_restored(prefix_ids, stage_ids[len(prefix_ids):]))
            if wrong_msgs is not None:
                wrong_ids = ids_for(tokenizer.apply_chat_template(
                    wrong_msgs, tools=tools, tokenize=False, add_generation_prompt=False, enable_thinking=False))
                entry['changed_tool_response_prefix_tokens'] = len(wrong_ids)
                entry['changed_tool_response_detected_by_exact_check'] = stage_ids[:len(wrong_ids)] != wrong_ids
                if len(wrong_ids) < len(stage_ids):
                    entry['stale_reuse_vs_full'] = compare(full, last_logits_restored(wrong_ids, stage_ids[len(wrong_ids):]))
            probes.append(entry)
            mx.clear_cache()
    write_json_exclusive(args.output, {'schema_version': 2, 'probes': probes,
                                       'runtime': {'mlx': _version('mlx'), 'mlx_lm': _version('mlx-lm')}})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
