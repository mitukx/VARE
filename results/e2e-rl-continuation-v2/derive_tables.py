"""Derive descriptive tables from the audited v2 raw campaign; no model execution."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as stats
from collections import Counter


def read(path):
    return json.loads(path.read_text())


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def sample_sd(values):
    """A sample with one observation has no defined sample standard deviation."""
    return stats.stdev(values) if len(values) >= 2 else None


def csv_write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def resource_metrics(case_dir):
    samples = [json.loads(line) for line in (case_dir/'resources.jsonl').read_text().splitlines()]
    if len(samples) < 2:
        raise ValueError('resource samples are incomplete')
    times = [row['time'] for row in samples]
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in times)
            or any(right <= left for left, right in zip(times, times[1:]))):
        raise ValueError('resource sample timestamps are invalid')
    window = times[-1] - times[0]
    if window <= 0:
        raise ValueError('resource sample window is invalid')
    return {
        'sampled_resource_window_s': window,
        'sampled_cpu_core_percent': 100*sum(row['cpu_delta_s'] for row in samples)/window,
        'sampled_tree_peak_rss_bytes': max(row['rss_bytes'] for row in samples),
    }


def exact_recovery_match(root, row, reference):
    observed = [read(root/row['directory']/'rounds'/f'r{i:04d}'/'committed.json')
                for i in range(row['rounds'])]
    if len(observed) != len(reference):
        return False
    for actual, expected in zip(observed, reference, strict=True):
        actual_update = {k: v for k, v in actual['update'].items() if k != 'optimizer_steps'}
        expected_update = {k: v for k, v in expected['update'].items() if k != 'optimizer_steps'}
        if (actual['active_state_sha256'] != expected['active_state_sha256']
                or actual['candidate_state_sha256'] != expected['candidate_state_sha256']
                or actual['accepted'] != expected['accepted']
                or actual_update != expected_update):
            return False
    return True


def validated_normal_trace(root, row):
    """Check latency evidence and required stage coverage before publishing tables."""
    case = root / row['directory']
    trace = [json.loads(line) for line in (case/'trace.jsonl').read_text().splitlines()]
    tasks = read(case/'config.json')['tasks']
    rounds = row['rounds']
    known = {'checkpoint_hash', 'checkpoint_save', 'checkpoint_recovery', 'evaluation',
             'freshness', 'promotion', 'replay', 'rollback', 'rollout',
             'rollout_verified', 'round', 'training', 'verification'}
    counts = Counter()
    for span in trace:
        stage = span.get('stage')
        round_index = span.get('round_index')
        learner_attempt = span.get('learner_attempt')
        start, end, duration = (span.get(key) for key in ('start', 'end', 'duration_s'))
        if (stage not in known or type(round_index) is not int or
                not -1 <= round_index < rounds or type(learner_attempt) is not int or
                learner_attempt != 0 or type(start) not in (int, float) or
                not math.isfinite(start) or type(end) not in (int, float) or
                not math.isfinite(end) or end < start or type(duration) not in (int, float) or
                not math.isfinite(duration) or duration != end-start):
            raise ValueError('trace span timestamps or identity are invalid')
        if round_index == -1 and stage not in ('checkpoint_hash', 'checkpoint_save'):
            raise ValueError('unexpected setup-stage trace span')
        counts[(stage, round_index, learner_attempt)] += 1
        if stage == 'rollout':
            service_start, service_end = span.get('service_start'), span.get('service_end')
            if (type(service_start) not in (int, float) or not math.isfinite(service_start) or
                    type(service_end) not in (int, float) or not math.isfinite(service_end) or
                    not span['start'] <= service_start <= service_end <= span['end']):
                raise ValueError('rollout service interval is outside its trace span')

    attempts = [json.loads(line) for line in (case/'optimizer_attempts.jsonl').read_text().splitlines()]
    if {(attempt['round_index'], attempt['learner_attempt']) for attempt in attempts} != {
            (index, 0) for index in range(rounds)}:
        raise ValueError('normal trace optimizer attempts are incomplete')
    commits = [read(case/'rounds'/f'r{index:04d}'/'committed.json') for index in range(rounds)]
    policy_lags = []
    for index in range(rounds):
        active_version = commits[index-1]['accepted_updates'] if index else 0
        round_rollouts = [json.loads(line) for line in
                          (case/'rounds'/f'r{index:04d}'/'rollouts-attempt0.jsonl').read_text().splitlines()]
        expected_lags = []
        for rollout in round_rollouts:
            version = rollout.get('policy_version')
            if type(version) is not int or version < 0:
                raise ValueError('rollout policy version is invalid')
            expected_lags.append(max(0, active_version-version))
        trace_lags = [span.get('policy_lag') for span in trace
                      if span['stage'] == 'rollout' and span['round_index'] == index]
        if (len(expected_lags) != tasks or any(type(lag) is not int or lag < 0 for lag in trace_lags) or
                sorted(trace_lags) != sorted(expected_lags)):
            raise ValueError('rollout trace policy lag differs from rollout provenance')
        policy_lags.extend(expected_lags)
    canary = read(case/'canary.json')['attempt']
    for index in range(rounds):
        stale_canary = int(index > 0 and canary['policy_version'] < commits[index-1]['accepted_updates'])
        for stage, expected in (
                ('rollout', tasks), ('rollout_verified', tasks), ('verification', tasks),
                ('freshness', 2*tasks + (tasks if row['mode'] == 'guarded' else 0) + stale_canary),
                ('replay', 1), ('training', 1), ('evaluation', 2), ('promotion', 1), ('round', 1),
                ('checkpoint_save', 1)):
            if counts[(stage, index, 0)] != expected:
                raise ValueError(f'{stage} trace coverage differs from retained round evidence')
        evaluation_ids = {path.stem.removeprefix('evaluation-').removesuffix('-attempt0')
                          for path in (case/'rounds'/f'r{index:04d}').glob('evaluation-*-attempt0.json')}
        traced_ids = [span.get('policy_id') for span in trace
                      if span['stage'] == 'evaluation' and span['round_index'] == index]
        if len(evaluation_ids) != 2 or sorted(traced_ids) != sorted(evaluation_ids):
            raise ValueError('evaluation trace identities differ from retained evaluations')
    if counts[('checkpoint_save', -1, 0)] != 1:
        raise ValueError('initial checkpoint-save trace coverage differs')
    expected_hashes = (2 + 4*rounds if row['mode'] == 'minimal'
                       else 3 + 11*rounds + int(rounds > 0))
    if sum(count for (stage, _, _), count in counts.items() if stage == 'checkpoint_hash') != expected_hashes:
        raise ValueError('checkpoint-hash trace coverage differs from audited operations')
    for stage in ('rollback', 'checkpoint_recovery'):
        if sum(count for (name, _, _), count in counts.items() if name == stage):
            raise ValueError(f'unexpected {stage} span in a normal case')
    return trace, max(policy_lags)


def derive(root, output):
    rows = read(root / 'progress.json')
    resources = {row['directory']: resource_metrics(root/row['directory']) for row in rows}
    normal = [r for r in rows if r['cutpoint'] == 'none']
    recovery_references = {
        row['seed']: [read(root/row['directory']/'rounds'/f'r{i:04d}'/'committed.json')
                      for i in range(row['rounds'])]
        for row in normal if row['workers'] == 2 and row['mode'] == 'guarded'
    }
    comparison, stages, recovery = [], [], []
    for workers in sorted({r['workers'] for r in normal}):
        for mode in ('minimal', 'guarded'):
            group = [r for r in normal if r['workers'] == workers and r['mode'] == mode]
            validated = [validated_normal_trace(root, row) for row in group]
            per_case = [case_trace for case_trace, _ in validated]
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
                'sampled_resource_window_s': [resources[r['directory']]['sampled_resource_window_s'] for r in group],
                'sampled_cpu_core_percent': [resources[r['directory']]['sampled_cpu_core_percent'] for r in group],
            }.items():
                row[key+'_mean'] = stats.fmean(values)
                row[key+'_sample_sd'] = sample_sd(values)
            row['sampled_peak_tree_rss_mib_max'] = max(resources[r['directory']]['sampled_tree_peak_rss_bytes'] for r in group)/2**20
            row['observed_service_concurrency_max'] = peak
            row['participating_worker_pids_per_case'] = '/'.join(str(len({r['worker_pid'] for r in case_trace
                                                                        if r['stage']=='rollout'})) for case_trace in per_case)
            row['normal_policy_lag_max'] = max(policy_lag for _, policy_lag in validated)
            rejected = sum(sum(read(root/r['directory']/'rounds'/f'r{i:04d}'/'committed.json')
                                   ['stale_replay_rejected'] for i in range(r['rounds'])) for r in group)
            row['stale_canary_rejected'] = rejected
            row['replay_candidates'] = len(rollouts)+rejected
            row['replay_rejection_rate'] = rejected / row['replay_candidates']
            paired = [next(r for r in normal if r['workers']==workers and r['seed']==g['seed'] and r['mode']=='minimal') for g in group]
            overhead = [100*(g['round_wall_s']/b['round_wall_s']-1) for g, b in zip(group, paired, strict=True)]
            row['paired_wall_overhead_percent_mean'] = stats.fmean(overhead)
            row['paired_wall_overhead_percent_sample_sd'] = sample_sd(overhead)
            comparison.append(row)
            for stage in sorted({r['stage'] for r in trace if r.get('round_index', -1) >= 0}):
                values = [r['duration_s'] for r in trace if r['stage']==stage and r.get('round_index', -1)>=0]
                stages.append({'workers':workers, 'mode':mode, 'stage':stage, 'n':len(values),
                               'p50_ms':1000*quantile(values, .5), 'p95_ms':1000*quantile(values, .95),
                               'span_sum_s':sum(values)})
    for cut in read(root/'sources/benchmarks/e2e_rl_continuation_v2/protocol.json')['cutpoints']:
        group = [r for r in rows if r['cutpoint']==cut]
        values = [r['recovery_to_next_update_s'] for r in group]
        recovery.append({'cutpoint':cut, 'n':len(group), 'mean_s':stats.fmean(values), 'sample_sd_s':sample_sd(values),
                         'min_s':min(values), 'max_s':max(values),
                         'physical_optimizer_attempts':sum(r['physical_optimizer_attempts'] for r in group),
                         'logical_rounds':sum(r['completed_rounds'] for r in group),
                         'exact_reference_cases':sum(exact_recovery_match(
                             root, r, recovery_references[r['seed']]) for r in group),
                         'live_old_descendants':sum(read(root/r['directory']/'kill_record.json')
                                                    ['live_old_descendants'] for r in group)})
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
