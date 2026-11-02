"""Inspect duplicate rollout cancellation and equal-adapter eval consistency."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = Path('<HOME>/vare-format-vs-correctness-v2')
blob = (RUN / 'progress.jsonl').read_bytes()
rows = [json.loads(line) for line in blob.decode().splitlines() if line.strip()]
audit = json.loads((HERE / 'audit.json').read_text())
assert hashlib.sha256(blob).hexdigest() == audit['progress_sha256']
groups = defaultdict(Counter)
mixed_cases = []
for row in rows:
    if row['kind'] != 'train' or row['arm'] == 'format_sft': continue
    same = row['outputs'][0] == row['outputs'][1]
    mixed = row['rewards'][0] != row['rewards'][1]
    groups[row['arm']].update(groups=1, exact_duplicate_outputs=int(same),
                              mixed_groups=int(mixed), mixed_duplicate_outputs=int(same and mixed),
                              mixed_duplicate_nonzero_loss=int(same and mixed and row['loss'] != 0))
    if row['arm'] == 'random_reward_rl' and mixed:
        mixed_cases.append({'run_seed': row['run_seed'], 'step': row['step'],
                            'identical_output_strings': same, 'loss': row['loss'],
                            'rewards': row['rewards']})

eval_rows = {}
for row in rows:
    if row['kind'] == 'eval':
        key = (row['arm'], row['run_seed'], row['checkpoint_step'], row['split'],
               row['task_id'], row['sample_index'])
        eval_rows[key] = row
checks = []
for entry in audit['cross_arm_checkpoints']:
    seed = entry['run_seed']
    for step in (8, 16):
        if step not in entry['format_and_correctness_adapter_equal_steps']: continue
        pairs = []
        for key, row in eval_rows.items():
            arm, run_seed, checkpoint_step, split, task, sample = key
            if arm == 'format_reward_rl' and run_seed == seed and checkpoint_step == step:
                other = eval_rows['correctness_reward_rl', seed, step, split, task, sample]
                assert row['sample_seed'] == other['sample_seed']
                pairs.append(row['output'] == other['output'])
        checks.append({'run_seed': seed, 'checkpoint_step': step, 'pairs': len(pairs),
                       'identical_outputs': sum(pairs), 'all_identical': all(pairs)})

out = {'progress_sha256': hashlib.sha256(blob).hexdigest(),
       'groups': {arm: dict(counts) for arm, counts in groups.items()},
       'random_mixed_groups': mixed_cases,
       'equal_format_correctness_adapter_same_sampling_seed_output_checks': checks,
       'notes': ['Identical output strings imply identical encoded completion under the same deterministic tokenizer.',
                 'Opposite group advantages then cancel the same completion NLL in the current zero-dropout objective.',
                 'Mixed rewards alone do not prove a nonzero gradient.',
                 'This is a retrospective record consistency check, not a controlled experiment.']}
(HERE / 'duplicate-rollouts.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
print(json.dumps(out, indent=2, sort_keys=True))
