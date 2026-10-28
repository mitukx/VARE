#!/usr/bin/env python3
"""Deterministic validation and descriptive analysis of the frozen run."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'run-002'
OUT = ROOT / 'analysis.json'
source = json.loads((ROOT / 'input.json').read_text())
summary_files = sorted((RUN / 'model-output').glob('attempt-*/attempt-summary.json'))
assert len(summary_files) == 4
attempts = []
for summary_path in summary_files:
    summary = json.loads(summary_path.read_text())
    arm = summary['arm']
    directory = summary_path.parent
    read = json.loads((directory / 'turn-01-read-lines.json').read_text())
    finish = json.loads((directory / 'turn-02-finish.json').read_text())
    assert summary['outcome'] == 'two_turn_tool_interaction_complete'
    assert read['parse_status'] == finish['parse_status'] == 'complete_tool_call'
    assert read['parsed_call']['name'] == 'read_lines'
    assert read['parsed_call']['arguments'] == {
        'path': source['fixture_path'], 'start': source['read_start'], 'end': source['read_end'],
    }
    assert summary['tool_execution']['result']['status'] == 'read_complete'
    assert summary['tool_execution']['result']['content'] == source['fixture_source']
    assert finish['parsed_call']['name'] == 'finish'
    turn_details = [read['generation_protocol_detail'], finish['generation_protocol_detail']]
    all_stages = [stage for detail in turn_details for stage in detail['stages']]
    assert len(all_stages) == 6
    cache_tokens = [int(stage['kv_cache_reused_prefix_tokens']) for stage in all_stages]
    fallback_reasons = [stage.get('kv_cache_fallback_reason') for stage in all_stages]
    attempts.append({
        'attempt_index': summary['attempt_index'],
        'arm': arm,
        'outcome': summary['outcome'],
        'wall_seconds': summary['wall_seconds'],
        'tool_calls': [
            {'name': 'read_lines', 'parse_status': read['parse_status'],
             'generated_tokens': read['generated_tokens'], 'prompt_tokens': read['prompt_tokens']},
            {'name': 'finish', 'parse_status': finish['parse_status'],
             'generated_tokens': finish['generated_tokens'], 'prompt_tokens': finish['prompt_tokens']},
        ],
        'source_read_status': summary['tool_execution']['result']['status'],
        'source_read_arguments': summary['tool_execution']['arguments'],
        'full_prompt_tokens_across_stages': sum(int(s['full_prompt_tokens']) for s in all_stages),
        'prefill_tokens_from_generation': sum(int(s['prefill_input_tokens']) for s in all_stages),
        'shared_prefix_tokens_built_once_per_turn': sum(
            int(d.get('shared_prefix_cache', {}).get('prefix_tokens', 0)) for d in turn_details
        ),
        'amortized_prefill_tokens': (
            sum(int(s['prefill_input_tokens']) for s in all_stages)
            + sum(int(d.get('shared_prefix_cache', {}).get('prefix_tokens', 0)) for d in turn_details)
        ),
        'measured_prefill_seconds': round(sum(float(s['prompt_prefill_seconds'] or 0) for s in all_stages), 6),
        'shared_prefix_build_seconds': round(sum(float(d.get('shared_prefix_cache', {}).get('build_wall_seconds', 0)) for d in turn_details), 6),
        'cache_restore_seconds': round(sum(float(s['kv_cache_restore_wall_seconds']) for s in all_stages), 6),
        'cached_prefix_tokens_per_stage': cache_tokens,
        'cache_fallback_reasons': fallback_reasons,
        'max_metal_stage_peak_bytes': max(int(s['metal_peak_memory_bytes_this_stage']) for s in all_stages),
    })

comparison = json.loads((RUN / 'model-output' / 'comparison.json').read_text())
pairs = comparison['paired_repetitions']
assert len(pairs) == 2 and all(pair['raw_turn_outputs_and_token_ids_equal'] for pair in pairs)
assert all(a['outcome'] == 'two_turn_tool_interaction_complete' for a in attempts)
assert all(a['tool_calls'][0]['parse_status'] == a['tool_calls'][1]['parse_status'] == 'complete_tool_call' for a in attempts)
v10 = [a for a in attempts if a['arm'] == 'schema_slots_v10']
assert len(v10) == 2
assert all(a['cache_fallback_reasons'] == [None] * 6 for a in v10)
assert all(a['cached_prefix_tokens_per_stage'] == [2980] * 4 + [3092] * 2 for a in v10)
resource = json.loads((RUN / 'resource-summary.json').read_text())
assert resource['status'] == 'COMPLETED' and resource['maximum_swap_growth_bytes'] == 0
pair_results=[]
for pair in pairs:
    a=pair['schema_slots_v9_metrics']; b=pair['schema_slots_v10_metrics']
    pair_results.append({
        'replicate':pair['replicate'],
        'raw_output_and_generated_token_ids_equal':pair['raw_turn_outputs_and_token_ids_equal'],
        'v9_wall_seconds':pair['schema_slots_v9_wall_seconds'],
        'v10_wall_seconds':pair['schema_slots_v10_wall_seconds'],
        'wall_time_reduction_percent':round(100*(1-pair['schema_slots_v10_wall_seconds']/pair['schema_slots_v9_wall_seconds']),3),
        'v9_amortized_prefill_tokens':a['amortized_prefill_tokens_including_shared_prefix'],
        'v10_amortized_prefill_tokens':b['amortized_prefill_tokens_including_shared_prefix'],
        'prefill_token_reduction_percent':round(100*(1-b['amortized_prefill_tokens_including_shared_prefix']/a['amortized_prefill_tokens_including_shared_prefix']),3),
        'v9_measured_prefill_seconds':a['measured_prefill_seconds_including_shared_prefix_build'],
        'v10_measured_prefill_seconds':b['measured_prefill_seconds_including_shared_prefix_build'],
        'measured_prefill_reduction_percent':round(100*(1-b['measured_prefill_seconds_including_shared_prefix_build']/a['measured_prefill_seconds_including_shared_prefix_build']),3),
    })
result={
  'status':'ANALYSIS_VALIDATED',
  'protocol_id':'vare_sprint_x_schema_slots_kv_cache_v2',
  'attempts':attempts,
  'paired_results':pair_results,
  'aggregate':{
    'attempts':len(attempts),'completed_two_turn_interactions':sum(a['outcome']=='two_turn_tool_interaction_complete' for a in attempts),
    'accepted_tool_calls':sum(len(a['tool_calls']) for a in attempts),
    'unique_synthetic_families':1,'external_development_families':0,'independently_verified_repairs':0,
    'model_load_seconds':comparison['model_load_seconds'],
    'mean_v9_wall_seconds':round(sum(p['v9_wall_seconds'] for p in pair_results)/2,6),
    'mean_v10_wall_seconds':round(sum(p['v10_wall_seconds'] for p in pair_results)/2,6),
    'mean_pairwise_wall_reduction_percent':round(sum(p['wall_time_reduction_percent'] for p in pair_results)/2,3),
    'mean_pairwise_prefill_token_reduction_percent':round(sum(p['prefill_token_reduction_percent'] for p in pair_results)/2,3),
    'mean_pairwise_prefill_seconds_reduction_percent':round(sum(p['measured_prefill_reduction_percent'] for p in pair_results)/2,3),
    'resource_summary':resource,
    'confidence_interval':'not estimated: two repetitions on one synthetic fixture are not independent task families',
  },
  'interpretation':'Exact strict tool outputs and generated token IDs matched in both pairs; the shared-prefix arm reduced measured prefill work and wall time on this synthetic long-context interaction. No external repair or general coding-capability conclusion follows.'
}
OUT.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(result['aggregate'],ensure_ascii=False,sort_keys=True,indent=2))
