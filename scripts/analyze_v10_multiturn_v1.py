#!/usr/bin/env python3
"""Paired analysis of the v9/v10 multi-turn ABBA campaign with t and bootstrap intervals.

Reads only retained raw artifacts (trajectory, guard, grade, probe) and the
frozen protocol. Applies the preregistered decision rule mechanically.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
         18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def t_critical(df: int) -> float:
    if df in T_975:
        return T_975[df]
    keys = sorted(T_975)
    return T_975[max(k for k in keys if k <= df)] if df < 30 else 1.96


def interval(values: list[float], *, seed: int, resamples: int) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {'n': 0}
    mean = statistics.fmean(values)
    out: dict[str, Any] = {'n': n, 'mean': mean, 'min': min(values), 'max': max(values)}
    if n >= 2:
        sd = statistics.stdev(values)
        half = t_critical(n - 1) * sd / math.sqrt(n)
        out.update({'sd': sd, 't95_low': mean - half, 't95_high': mean + half})
        rng = random.Random(seed)
        boots = sorted(statistics.fmean(rng.choice(values) for _ in range(n)) for _ in range(resamples))
        out.update({'bootstrap95_low': boots[int(0.025 * resamples)],
                    'bootstrap95_high': boots[int(0.975 * resamples) - 1]})
    return out


def load_runs(run_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted((run_dir / 'runs').iterdir()):
        guard = json.loads((path / 'guard.json').read_text())
        trajectory = path / 'trajectory.json'
        grade = path / 'grade.json'
        runs.append({'label': path.name, 'guard': guard,
                     'trajectory': json.loads(trajectory.read_text()) if trajectory.is_file() else None,
                     'grade': json.loads(grade.read_text()) if grade.is_file() else None})
    return runs


def trajectory_metrics(run: dict[str, Any]) -> dict[str, Any]:
    record, guard = run['trajectory'], run['guard']
    metrics: dict[str, Any] = {
        'label': run['label'], 'arm': guard['arm'], 'variant': guard['variant'], 'horizon': guard['horizon'],
        'block': guard['block'], 'position': guard['position'], 'abort': guard['abort'],
        'exit_code': guard['exit_code'], 'guard_wall_seconds': guard['guard_wall_seconds'],
        'guard_peak_rss_bytes': guard['peak_process_tree_rss_bytes'],
        'swap_growth_bytes': guard['swap_growth_during_run_bytes'],
    }
    if record is None:
        metrics.update({'outcome': 'no_trajectory_record', 'completed': False})
        return metrics
    turns = [turn for turn in record['turns'] if turn.get('status') == 'completed']
    fallback_reasons: dict[str, int] = {}
    for turn in turns:
        for reason in turn.get('kv_fallback_reasons') or []:
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
    base_active = record.get('metal_active_after_load_bytes', 0)
    metrics.update({
        'outcome': record['outcome'],
        'completed': record['outcome'] == 'horizon_completed' and not guard['abort'],
        'turns_completed': len(turns),
        'model_load_seconds': record.get('model_load_seconds'),
        'trajectory_wall_seconds': record.get('trajectory_wall_seconds'),
        'generation_wall_seconds': sum(t['generation_wall_seconds'] for t in turns),
        'prefill_seconds': sum(t['stage_prefill_seconds'] + t['shared_prefix_build_seconds'] for t in turns),
        'decode_seconds': sum(t['decode_seconds'] for t in turns),
        'tool_seconds': sum(t['tool_seconds'] for t in turns),
        'full_prompt_tokens': sum(t['full_prompt_tokens_sum'] for t in turns),
        'amortized_prefill_tokens': sum(t['amortized_prefill_tokens'] for t in turns),
        'per_turn_prefill_tokens': [t['amortized_prefill_tokens'] for t in turns],
        'per_turn_generation_seconds': [t['generation_wall_seconds'] for t in turns],
        'generated_tokens': sum(t['generated_tokens'] for t in turns),
        'metal_peak_bytes': record.get('metal_peak_bytes_trajectory'),
        'process_peak_rss_bytes': record.get('process_peak_rss_bytes'),
        'parser_accepted_turns': sum(1 for t in turns if t['parse_status'] == 'complete_tool_call'),
        'stages': sum(t['stage_count'] for t in turns),
        'kv_reuse_stages': sum(t['kv_reuse_stages'] for t in turns),
        'kv_fallback_reasons': fallback_reasons,
        'max_active_growth_after_turn_bytes': max(
            [t['metal_active_after_turn_bytes'] - base_active for t in turns] or [0]),
        'audit_violations': sum(len((t.get('prefix_audit') or {}).get('violations') or []) for t in turns),
        'edits_applied': record.get('edits_applied'),
        'actions': [[t['action']['name'], t['action']['arguments']] if t.get('action') else ['<rejected>', t['raw_model_output']]
                    for t in turns],
        'token_ids': [t['stage_generated_token_ids'] for t in turns],
        'correct': bool(run['grade'] and run['grade'].get('correct')),
    })
    return metrics


def first_divergence(left: list[Any], right: list[Any]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right), start=1):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right)) + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    stats_cfg = protocol['analysis']
    rules = protocol['decision_rule']['thresholds']
    runs = [trajectory_metrics(run) for run in load_runs(args.run_dir)]
    by_block: dict[int, list[dict[str, Any]]] = {}
    for item in runs:
        by_block.setdefault(item['block'], []).append(item)
    pairs = []
    determinism = []
    for block, items in sorted(by_block.items()):
        position = {item['position']: item for item in items}
        for pair_positions in ((1, 2), (3, 4)):
            members = [position[p] for p in pair_positions if p in position]
            if len(members) != 2:
                continue
            v9 = next((m for m in members if m['arm'] == 'schema_slots_v9'), None)
            v10 = next((m for m in members if m['arm'] == 'schema_slots_v10'), None)
            if v9 is None or v10 is None:
                continue
            pair = {'block': block, 'variant': v9['variant'], 'horizon': v9['horizon'],
                    'v9': v9['label'], 'v10': v10['label'], 'both_completed': v9['completed'] and v10['completed']}
            if v9.get('actions') is not None and v10.get('actions') is not None:
                pair.update({
                    'actions_equal': v9['actions'] == v10['actions'],
                    'token_ids_equal': v9['token_ids'] == v10['token_ids'],
                    'first_action_divergence_turn': first_divergence(v9['actions'], v10['actions']),
                    'first_token_divergence_turn': first_divergence(v9['token_ids'], v10['token_ids']),
                    'correct_v9': v9['correct'], 'correct_v10': v10['correct'],
                })
            if pair['both_completed']:
                for key in ('trajectory_wall_seconds', 'generation_wall_seconds', 'prefill_seconds',
                            'decode_seconds', 'tool_seconds', 'amortized_prefill_tokens', 'full_prompt_tokens',
                            'metal_peak_bytes', 'process_peak_rss_bytes', 'model_load_seconds', 'guard_wall_seconds',
                            'swap_growth_bytes'):
                    pair[f'{key}_v9'] = v9[key]
                    pair[f'{key}_v10'] = v10[key]
            pairs.append(pair)
        for arm in ('schema_slots_v9', 'schema_slots_v10'):
            same = [item for item in items if item['arm'] == arm and item.get('token_ids') is not None]
            if len(same) == 2:
                determinism.append({'block': block, 'arm': arm, 'token_ids_equal': same[0]['token_ids'] == same[1]['token_ids']})

    def summarize(selected: list[dict[str, Any]], key: str, *, relative: bool) -> dict[str, Any]:
        values = []
        for pair in selected:
            a, b = pair.get(f'{key}_v9'), pair.get(f'{key}_v10')
            if a is None or b is None:
                continue
            values.append((1 - b / a) * 100 if relative and a else b - a)
        return interval(values, seed=stats_cfg['bootstrap_seed'], resamples=stats_cfg['bootstrap_resamples'])

    summaries: dict[str, Any] = {}
    for horizon in (8, 16, 'pooled'):
        selected = [p for p in pairs if p['both_completed'] and (horizon == 'pooled' or p['horizon'] == horizon)]
        summaries[str(horizon)] = {
            'completed_pairs': len(selected),
            'trajectory_wall_reduction_percent': summarize(selected, 'trajectory_wall_seconds', relative=True),
            'generation_wall_reduction_percent': summarize(selected, 'generation_wall_seconds', relative=True),
            'prefill_seconds_reduction_percent': summarize(selected, 'prefill_seconds', relative=True),
            'amortized_prefill_token_reduction_percent': summarize(selected, 'amortized_prefill_tokens', relative=True),
            'decode_seconds_difference_v10_minus_v9': summarize(selected, 'decode_seconds', relative=False),
            'tool_seconds_difference_v10_minus_v9': summarize(selected, 'tool_seconds', relative=False),
            'model_load_seconds_difference_v10_minus_v9': summarize(selected, 'model_load_seconds', relative=False),
            'metal_peak_increase_bytes': summarize(selected, 'metal_peak_bytes', relative=False),
            'process_peak_rss_increase_bytes': summarize(selected, 'process_peak_rss_bytes', relative=False),
            'trajectory_wall_seconds_difference': summarize(selected, 'trajectory_wall_seconds', relative=False),
            'equivalent_pairs_only_trajectory_wall_reduction_percent': summarize(
                [p for p in selected if p.get('actions_equal')], 'trajectory_wall_seconds', relative=True),
        }

    per_turn: dict[str, Any] = {}
    for item in runs:
        if not item.get('completed'):
            continue
        key = f"h{item['horizon']}:{item['arm']}"
        per_turn.setdefault(key, []).append(item['per_turn_prefill_tokens'])
    per_turn_means = {key: [round(statistics.fmean(col), 1) for col in zip(*rows)] for key, rows in per_turn.items()}

    probe_path = args.run_dir / 'probe' / 'probe.json'
    probe = json.loads(probe_path.read_text()) if probe_path.is_file() else None
    probe_ok = bool(probe) and all(
        p.get('token_prefix_exact') and p['restored_vs_full']['argmax_equal']
        and p['restored_vs_full']['top5_overlap'] >= rules['probe_min_top5_overlap']
        for p in probe['probes'])
    stale_detected = bool(probe) and all(
        p.get('changed_tool_response_detected_by_exact_check', True) for p in probe['probes'])

    def arm_items(arm: str) -> list[dict[str, Any]]:
        return [item for item in runs if item['arm'] == arm]

    failures = {arm: sum(1 for item in arm_items(arm) if not item.get('completed')) for arm in
                ('schema_slots_v9', 'schema_slots_v10')}
    acceptance = {arm: (sum(item.get('parser_accepted_turns', 0) for item in arm_items(arm)),
                        sum(item.get('turns_completed', 0) for item in arm_items(arm)))
                  for arm in ('schema_slots_v9', 'schema_slots_v10')}
    correct = {arm: (sum(1 for item in arm_items(arm) if item.get('correct')), len(arm_items(arm)))
               for arm in ('schema_slots_v9', 'schema_slots_v10')}
    v10_items = arm_items('schema_slots_v10')
    gates = {
        'kv_correctness': (sum(item.get('audit_violations', 0) for item in runs) == 0 and probe_ok and stale_detected),
        'action_equivalence': bool(pairs) and all(p.get('actions_equal') for p in pairs if 'actions_equal' in p),
        'reliability': failures['schema_slots_v10'] <= failures['schema_slots_v9']
        and acceptance['schema_slots_v10'][0] * max(acceptance['schema_slots_v9'][1], 1)
        >= acceptance['schema_slots_v9'][0] * max(acceptance['schema_slots_v10'][1], 1),
        'retention_bounded': all(item.get('max_active_growth_after_turn_bytes', 0) <= rules['max_active_growth_after_turn_bytes']
                                 for item in runs if item.get('completed')),
        'efficiency': all(
            summaries[h]['completed_pairs'] >= 2
            and summaries[h]['trajectory_wall_reduction_percent']['mean'] >= rules['min_mean_wall_reduction_percent']
            and summaries[h]['trajectory_wall_reduction_percent']['t95_low'] > 0
            for h in ('8', '16')),
        'memory': all(
            summaries[h]['completed_pairs'] >= 1
            and summaries[h]['metal_peak_increase_bytes']['mean'] <= rules['max_mean_metal_peak_increase_bytes']
            for h in ('8', '16')) and all(
            (item.get('process_peak_rss_bytes') or 0) <= rules['max_v10_process_peak_rss_bytes'] for item in v10_items),
    }
    if not gates['kv_correctness'] or not gates['retention_bounded']:
        decision = 'NO_GO_V10_KEEP_V9_DEFAULT'
    elif not gates['action_equivalence'] or not gates['reliability'] or not gates['efficiency']:
        decision = 'KEEP_V9_DEFAULT_V10_EXPLICIT_OPT_IN'
    elif not gates['memory']:
        decision = 'KEEP_EXPLICIT_FALLBACK_V10_OPT_IN_UNDER_MEMORY_HEADROOM'
    else:
        decision = 'PROMOTE_V10_DEFAULT_RETAIN_EXPLICIT_V9_FALLBACK'
    output = {
        'protocol_id': protocol['protocol_id'],
        'runs': [{k: v for k, v in item.items() if k not in ('actions', 'token_ids')} for item in runs],
        'pairs': pairs, 'within_arm_determinism': determinism, 'summaries': summaries,
        'per_turn_mean_amortized_prefill_tokens': per_turn_means,
        'probe': probe, 'probe_ok': probe_ok, 'stale_prefix_detected_by_exact_check': stale_detected,
        'failures': failures, 'parser_acceptance': acceptance, 'task_correct': correct,
        'kv_fallback_reasons_total': {
            arm: {r: sum(item.get('kv_fallback_reasons', {}).get(r, 0) for item in arm_items(arm))
                  for r in {r for item in arm_items(arm) for r in item.get('kv_fallback_reasons', {})}}
            for arm in ('schema_slots_v9', 'schema_slots_v10')},
        'kv_reuse_stages_v10': sum(item.get('kv_reuse_stages', 0) for item in v10_items),
        'stages_v10': sum(item.get('stages', 0) for item in v10_items),
        'gates': gates, 'decision': decision,
        'limitations': protocol['limitations'],
    }
    args.output.write_text(json.dumps(output, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps({'decision': decision, 'gates': gates}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
