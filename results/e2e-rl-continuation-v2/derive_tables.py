"""Derive descriptive tables from the audited v2 raw campaign; no model execution."""
import argparse
import csv
import json
from pathlib import Path
import statistics as stats


def read(path):
    return json.loads(path.read_text())


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def csv_write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def resource_metrics(case_dir):
    samples = [json.loads(line) for line in (case_dir/'resources.jsonl').read_text().splitlines()]
    if len(samples) < 2:
        raise ValueError('resource samples are incomplete')
    window = samples[-1]['time'] - samples[0]['time']
    if window <= 0:
        raise ValueError('resource sample window is invalid')
    return {
        'sampled_cpu_core_percent': 100*sum(row['cpu_delta_s'] for row in samples)/window,
        'sampled_tree_peak_rss_bytes': max(row['rss_bytes'] for row in samples),
    }


def derive(root, output):
    rows = read(root / 'progress.json')
    resources = {row['directory']: resource_metrics(root/row['directory']) for row in rows}
    normal = [r for r in rows if r['cutpoint'] == 'none']
    comparison, stages, recovery = [], [], []
    for workers in sorted({r['workers'] for r in normal}):
        for mode in ('minimal', 'guarded'):
            group = [r for r in normal if r['workers'] == workers and r['mode'] == mode]
            per_case = [[json.loads(line) for line in (root / row['directory'] / 'trace.jsonl').read_text().splitlines()]
                        for row in group]
            trace = [span for case_trace in per_case for span in case_trace]
            rollouts = [r for r in trace if r['stage'] == 'rollout']
            events = sorted([(r['service_start'], 1) for r in rollouts]
                            + [(r['service_end'], -1) for r in rollouts])
            current = peak = 0
            for _, delta in events:
                current += delta
                peak = max(peak, current)
            row = {'workers': workers, 'mode': mode, 'seeds': len(group)}
            for key, values in {
                'tasks_s': [r['tasks_s'] for r in group],
                'output_tokens_s': [r['tokens_s'] for r in group],
                'round_ms': [1000*r['round_wall_s']/r['rounds'] for r in group],
                'case_wall_s': [r['case_wall_s'] for r in group],
                'sampled_cpu_core_percent': [resources[r['directory']]['sampled_cpu_core_percent'] for r in group],
            }.items():
                row[key+'_mean'] = stats.fmean(values)
                row[key+'_sample_sd'] = stats.stdev(values)
            row['sampled_peak_tree_rss_mib_max'] = max(resources[r['directory']]['sampled_tree_peak_rss_bytes'] for r in group)/2**20
            row['observed_service_concurrency_max'] = peak
            row['participating_worker_pids_per_case'] = '/'.join(str(len({r['worker_pid'] for r in case_trace
                                                                        if r['stage']=='rollout'})) for case_trace in per_case)
            row['normal_policy_lag_max'] = max(r['policy_lag'] for r in rollouts)
            rejected = sum(r['stale_replay_rejected'] for r in group)
            row['stale_canary_rejected'] = rejected
            row['replay_candidates'] = len(rollouts)+rejected
            row['replay_rejection_rate'] = rejected / row['replay_candidates']
            setup = [read(root/g['directory']/'setup-attempt0.json') for g in group]
            row['rollout_pool_start_s_mean'] = stats.fmean(s['rollout_pool_start_s'] for s in setup)
            row['evaluation_pool_start_s_mean'] = stats.fmean(s['evaluation_pool_start_s'] for s in setup)
            paired = [next(r for r in normal if r['workers']==workers and r['seed']==g['seed'] and r['mode']=='minimal') for g in group]
            overhead = [100*(g['round_wall_s']/b['round_wall_s']-1) for g, b in zip(group, paired, strict=True)]
            row['paired_wall_overhead_percent_mean'] = stats.fmean(overhead)
            row['paired_wall_overhead_percent_sample_sd'] = stats.stdev(overhead)
            comparison.append(row)
            for stage in sorted({r['stage'] for r in trace if r.get('round_index', -1) >= 0}):
                values = [r['duration_s'] for r in trace if r['stage']==stage and r.get('round_index', -1)>=0]
                stages.append({'workers':workers, 'mode':mode, 'stage':stage, 'n':len(values),
                               'p50_ms':1000*quantile(values, .5), 'p95_ms':1000*quantile(values, .95),
                               'span_sum_s':sum(values)})
    for cut in read(root/'sources/benchmarks/e2e_rl_continuation_v2/protocol.json')['cutpoints']:
        group = [r for r in rows if r['cutpoint']==cut]
        values = [r['recovery_to_next_update_s'] for r in group]
        recovery.append({'cutpoint':cut, 'n':len(group), 'mean_s':stats.fmean(values), 'sample_sd_s':stats.stdev(values),
                         'min_s':min(values), 'max_s':max(values),
                         'physical_optimizer_attempts':sum(r['physical_optimizer_attempts'] for r in group),
                         'logical_rounds':sum(r['completed_rounds'] for r in group),
                         'exact_reference_cases':sum(r['exact_reference_match'] for r in group),
                         'live_old_descendants':sum(r['live_old_descendants'] for r in group)})
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output/'comparison.csv', comparison)
    csv_write(output/'stage-latency.csv', stages)
    csv_write(output/'recovery.csv', recovery)
    aggregate = {'comparison':comparison, 'recovery':recovery,
                 'campaign_sampled_peak_tree_rss_mib':max(resources[r['directory']]['sampled_tree_peak_rss_bytes'] for r in rows)/2**20,
                 'recovery_mean_s':stats.fmean(r['recovery_to_next_update_s'] for r in rows if r['cutpoint']!='none')}
    (output/'derived-metrics.json').write_text(json.dumps(aggregate, indent=2, sort_keys=True)+'\n')
    return aggregate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.raw, args.output), sort_keys=True))
