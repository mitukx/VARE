"""Read-only audit of an existing run with unknown executor/source revision."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np
from safetensors.numpy import load as load_safetensors

SOURCE = Path('/Users/user/Documents/Codex/2026-10-13/var/work/VARE-eval-stats')
RUN = Path('/Users/user/vare-format-vs-correctness-v2')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE / 'src'))
from vare import format_vs_correctness_v2 as frozen
from vare.answer_scoring import score_output
from vare.integer_seed_split import derive_seeds, task_from_seed


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def exact_truth(expression: str) -> int:
    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Fraction(node.value)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
        raise ValueError(f'unsupported syntax: {ast.dump(node)}')
    result = visit(ast.parse(expression, mode='eval').body)
    assert result.denominator == 1
    return result.numerator


def strict_integer(text: str) -> int | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines: return None
    match = re.fullmatch(r'####\s+(-?\d+)\s*', lines[-1])
    return int(match.group(1)) if match else None


def train_key(row: dict) -> tuple:
    return row['arm'], row['run_seed'], row['step']


def eval_key(row: dict) -> tuple:
    return (row['arm'], row['run_seed'], row['checkpoint_step'], row['split'],
            row['task_id'], row['sample_index'])


files = sorted(path for path in RUN.rglob('*') if path.is_file())
before = {str(path.relative_to(RUN)): digest(path.read_bytes()) for path in files}
blob = (RUN / 'progress.jsonl').read_bytes()
rows = [json.loads(line) for line in blob.decode().splitlines() if line.strip()]
train = [row for row in rows if row['kind'] == 'train']
evaluation = [row for row in rows if row['kind'] == 'eval']
environment = json.loads((RUN / 'environment.json').read_bytes())
stored_summary = json.loads((RUN / 'summary.json').read_bytes())

# Reconstruct task identities from the current frozen definitions. This does
# not prove which prompts or source code produced the existing run.
confirm = {}
for seed in derive_seeds('confirm'):
    task = task_from_seed(seed, 'confirm')
    confirm[f'confirm-{seed}'] = task.expression
pack_path = SOURCE / 'benchmarks/heldout/exact_integer_v1/tasks.jsonl'
pack_blob = pack_path.read_bytes()
pack_rows = [json.loads(line) for line in pack_blob.decode().splitlines() if line.strip()]
pack = {row['task_id']: row['expression'] for row in pack_rows}
assert len(pack) == len(pack_rows) == 64
for row in pack_rows:
    assert digest(str(exact_truth(row['expression'])).encode()) == row['value_sha256']
tasks = {'confirm': confirm, 'pack': pack}
training_seeds = [seed for seed in derive_seeds('train') if seed != 1029953][:16]
training_truth = {seed: exact_truth(task_from_seed(seed, 'train').expression) for seed in training_seeds}

expected_train = {(arm, seed, step) for arm in frozen.UPDATED_ARMS
                  for seed in frozen.RUN_SEEDS for step in range(1, 17)}
expected_eval = set()
sample_seed_errors = []
for arm in frozen.ARMS:
    for seed in frozen.RUN_SEEDS:
        for step in ((0,) if arm == 'base' else (8, 16)):
            for split in ('confirm', 'pack'):
                if split == 'pack' and step != (0 if arm == 'base' else 16): continue
                for task_index, task_id in enumerate(tasks[split]):
                    for sample in range(4):
                        expected_eval.add((arm, seed, step, split, task_id, sample))
train_counts = Counter(train_key(row) for row in train)
eval_counts = Counter(eval_key(row) for row in evaluation)
assert set(train_counts) == expected_train and set(train_counts.values()) == {1}
assert set(eval_counts) == expected_eval and set(eval_counts.values()) == {1}

eval_totals = defaultdict(Counter)
score_mismatches = []
for row in evaluation:
    expressions = tasks[row['split']]
    expression = expressions[row['task_id']]
    parsed = strict_integer(row['output'])
    truth = exact_truth(expression)
    own = {'strict_format': parsed is not None, 'strict_correct': parsed == truth}
    reference = score_output(expression, row['output'])
    if any(own[key] != reference[key] for key in own): score_mismatches.append(eval_key(row))
    key = row['arm'], row['checkpoint_step'], row['split']
    eval_totals[key].update(n=1, strict_format=int(own['strict_format']), strict_correct=int(own['strict_correct']))
    task_index = list(expressions).index(row['task_id'])
    expected_seed = 80_061_016 + row['run_seed'] * 100_000 + row['checkpoint_step'] * 1_000 + task_index * 10 + row['sample_index']
    if expected_seed != row['sample_seed']: sample_seed_errors.append(eval_key(row))

training = {}
reward_mismatches, advantage_mismatches, training_seed_errors = [], [], []
for arm in frozen.UPDATED_ARMS:
    arm_rows = [row for row in train if row['arm'] == arm]
    counts = Counter()
    mixed_steps = {str(seed): [] for seed in frozen.RUN_SEEDS}
    for row in arm_rows:
        if row['train_seed'] != training_seeds[row['step'] - 1]: training_seed_errors.append(train_key(row))
        if arm == 'format_sft':
            assert row['outputs'] is row['rewards'] is row['advantages'] is None
            continue
        assert len(row['outputs']) == 2
        parsed = [strict_integer(text) for text in row['outputs']]
        correct = [answer == training_truth[row['train_seed']] for answer in parsed]
        counts.update(outputs=2, strict_format=sum(answer is not None for answer in parsed), strict_correct=sum(correct))
        if arm == 'format_reward_rl': expected = [float(answer is not None) for answer in parsed]
        elif arm == 'correctness_reward_rl': expected = list(map(float, correct))
        else:
            rng = random.Random(f"format-vs-correctness-v2-random-reward:{row['run_seed']}:{row['step']}")
            expected = [float(rng.random() < .5) for _ in range(2)]
        if row['rewards'] != expected: reward_mismatches.append(train_key(row))
        mean = sum(expected) / 2
        std = math.sqrt(sum((reward - mean) ** 2 for reward in expected) / 2)
        advantages = [(reward - mean) / (std + 1e-8) for reward in expected] if std else [0., 0.]
        if any(abs(a - b) > 1e-12 for a, b in zip(row['advantages'], advantages, strict=True)):
            advantage_mismatches.append(train_key(row))
        if expected[0] != expected[1]: mixed_steps[str(row['run_seed'])].append(row['step'])
        if all(value == 0 for value in row['advantages']):
            counts['zero_advantage_groups'] += 1
            counts['zero_advantage_nonzero_loss'] += int(row['loss'] != 0)
        counts[str(expected)] += 1
    training[arm] = {'groups': len(arm_rows), 'output_counts': dict(counts),
                     'mixed_reward_steps_by_seed': mixed_steps,
                     'loss_min': min(row['loss'] for row in arm_rows),
                     'loss_max': max(row['loss'] for row in arm_rows)}

checkpoint_results = {}
tensor_by_arm_seed = {}
checkpoint_manifest = []
for arm in frozen.UPDATED_ARMS:
    seed_results = []
    for seed in frozen.RUN_SEEDS:
        previous = None
        first = None
        zero_b_steps, changed_steps, zero_moment_steps = [], [], []
        optimizer_step_errors, nonfinite_steps = [], []
        snapshots = {}
        for step in range(1, 17):
            directory = RUN / 'checkpoints' / arm / str(seed) / f'step-{step:04d}'
            adapters_blob = (directory / 'adapters.safetensors').read_bytes()
            optimizer_blob = (directory / 'optimizer.safetensors').read_bytes()
            config_blob = (directory / 'adapter_config.json').read_bytes()
            tensors = load_safetensors(adapters_blob)
            optimizer = load_safetensors(optimizer_blob)
            assert len(tensors) == 4 and all(key.endswith(('.lora_a', '.lora_b')) for key in tensors)
            assert json.loads(config_blob)['lora_parameters']['dropout'] == 0
            if any(not np.isfinite(value).all() for value in [*tensors.values(), *optimizer.values()]): nonfinite_steps.append(step)
            if all(np.count_nonzero(value) == 0 for key, value in tensors.items() if key.endswith('.lora_b')):
                zero_b_steps.append(step)
            moments = [value for key, value in optimizer.items() if key.endswith(('.m', '.v'))]
            assert len(moments) == 8
            if all(np.count_nonzero(value) == 0 for value in moments): zero_moment_steps.append(step)
            if int(optimizer['step']) != step: optimizer_step_errors.append(step)
            if previous is not None and any(not np.array_equal(tensors[key], previous[key]) for key in tensors): changed_steps.append(step)
            if first is None: first = tensors
            snapshots[step] = tensors
            checkpoint_manifest.append({'arm': arm, 'run_seed': seed, 'step': step,
                                        'adapter_sha256': digest(adapters_blob),
                                        'optimizer_sha256': digest(optimizer_blob),
                                        'config_sha256': digest(config_blob)})
            previous = tensors
        tensor_by_arm_seed[arm, seed] = snapshots
        seed_results.append({'run_seed': seed, 'zero_lora_b_steps': zero_b_steps,
                             'changed_from_previous_saved_step': changed_steps,
                             'zero_optimizer_moment_steps': zero_moment_steps,
                             'optimizer_step_errors': optimizer_step_errors,
                             'nonfinite_steps': nonfinite_steps,
                             'all_lora_a_equal_step1': all(np.array_equal(first[key], value[key]) for value in snapshots.values() for key in first if key.endswith('.lora_a'))})
    checkpoint_results[arm] = seed_results

cross_arm = []
for seed in frozen.RUN_SEEDS:
    format_snapshots = tensor_by_arm_seed['format_reward_rl', seed]
    correct_snapshots = tensor_by_arm_seed['correctness_reward_rl', seed]
    equal_steps = [step for step in range(1, 17) if all(np.array_equal(format_snapshots[step][key], correct_snapshots[step][key]) for key in format_snapshots[step])]
    cross_arm.append({'run_seed': seed, 'format_and_correctness_adapter_equal_steps': equal_steps})

primary_left = eval_totals['format_sft', 16, 'confirm']
primary_right = eval_totals['correctness_reward_rl', 16, 'confirm']
primary_difference = Fraction(primary_right['strict_correct'], primary_right['n']) - Fraction(primary_left['strict_correct'], primary_left['n'])
after = {str(path.relative_to(RUN)): digest(path.read_bytes()) for path in files}
assert before == after
assert digest(blob) == before['progress.jsonl']
assert not reward_mismatches and not advantage_mismatches and not score_mismatches
assert not sample_seed_errors and not training_seed_errors

result = {
    'description': 'Retrospective read-only inspection of existing records; executor and run source revision unknown',
    'run_path': str(RUN), 'progress_sha256': digest(blob), 'environment': environment,
    'source_used_for_reconstruction': '3a5a808e71d948cbe531a952344f8d5dee91f625',
    'source_module_sha256': digest((SOURCE / 'src/vare/format_vs_correctness_v2.py').read_bytes()),
    'task_pack_sha256': digest(pack_blob),
    'files_read': len(files), 'checkpoint_file_counts': dict(Counter(path.name for path in files if 'checkpoints' in path.parts)),
    'input_hashes_unchanged_during_audit': before == after,
    'record_counts': dict(Counter(row['kind'] for row in rows)),
    'record_backend_values': sorted({row['backend'] for row in rows}),
    'record_model_id_values': sorted({row['model_id'] for row in rows}),
    'record_protocol_sha256_values': sorted({row['protocol_sha256'] for row in rows}),
    'analysis_sha256_tagged_rows': sum('analysis_sha256' in row for row in rows),
    'complete_grid_unique': True, 'sample_seed_errors': sample_seed_errors,
    'training_seed_errors': training_seed_errors,
    'independent_score_mismatches': score_mismatches,
    'reward_mismatches': reward_mismatches, 'advantage_mismatches': advantage_mismatches,
    'eval_counts': [{'arm': arm, 'checkpoint_step': step, 'split': split, **counts} for (arm, step, split), counts in sorted(eval_totals.items())],
    'training': training, 'checkpoints': checkpoint_results, 'cross_arm_checkpoints': cross_arm,
    'primary_direct_count_difference': {'fraction': str(primary_difference), 'float': float(primary_difference),
                                        'matches_stored_mean': math.isclose(float(primary_difference), stored_summary['paired_bootstrap']['mean'], rel_tol=0, abs_tol=1e-15)},
    'bootstrap_rerun': False, 'model_loading_or_generation': False, 'training_executed': False,
    'limitations': ['Run executor and source revision are unknown.',
                    'Task expressions are reconstructed from current frozen definitions, not captured run prompts.',
                    'No step-zero adapter checkpoint exists; unchanged comparisons begin at saved step one.',
                    'Checkpoint consistency and raw-record consistency do not establish a controlled experiment or historical execution integrity.'],
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'input-manifest.json').write_text(json.dumps(before, indent=2, sort_keys=True) + '\n')
(OUT / 'checkpoint-manifest.json').write_text(json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + '\n')
(OUT / 'audit.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, indent=2, sort_keys=True))
