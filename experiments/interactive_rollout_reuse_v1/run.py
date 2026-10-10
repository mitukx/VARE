"""Exact finite-law prerequisites; pinned-source acceptance sanity, not LM training."""
from fractions import Fraction as F
from itertools import product
from collections import defaultdict
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time

START = time.perf_counter()
Q = (F(9, 10), F(1, 10))
P = Q[::-1]


def tv(a, b):
    return sum(abs(a.get(x, 0) - b.get(x, 0)) for x in set(a) | set(b)) / 2


def kernel(x, q, p, residual=True):
    accept = min(F(1), p[x] / q[x])
    d = sum(max(F(0), p[z] - q[z]) for z in (0, 1))
    fallback = tuple(max(F(0), p[z] - q[z]) / d for z in (0, 1)) if residual and d else p
    return tuple((accept if z == x else 0) + (1 - accept) * fallback[z] for z in (0, 1))


def transport(q, p, residual):
    return {z: sum(q[x] * kernel(x, q, p, residual)[z] for x in (0, 1)) for z in (0, 1)}


def iid_group(p):
    return {a: p[a[0]] * p[a[1]] for a in product((0, 1), repeat=2)}


def grpo_update(law):
    # For two independent Bernoulli-action samples, binary rewards r=a,
    # population standard deviation normalization without epsilon, and a common
    # logit parameter, the mean score*advantage is 1/2 for mixed groups, 0 otherwise.
    return sum(prob * F(1, 2) * (a[0] != a[1]) for a, prob in law.items())


def assert_law(law):
    assert sum(law.values()) == 1 and min(law.values()) >= 0


naive = transport(Q, P, False)
residual = transport(Q, P, True)
exogenous = {
    'naive_law': naive, 'residual_law': residual,
    'naive_tv': tv(naive, dict(enumerate(P))),
    'residual_tv': tv(residual, dict(enumerate(P))),
    'naive_group_update': grpo_update(iid_group(tuple(naive.values()))),
    'fresh_group_update': grpo_update(iid_group(P)),
}

q = (F(1, 2), F(1, 2))
adaptive = []
for selected in (0, 1):
    p = (F(1, 10), F(9, 10)) if selected else (F(9, 10), F(1, 10))
    consumed, independent = defaultdict(F), defaultdict(F)
    # Conditional on target selection, consumed first action is known. Second
    # action is still independent uniform. Independent donor has q x q law.
    for draft in product((0, 1), repeat=2):
        consumed_prob = F(1, 2) if draft[0] == selected else F(0)
        for out in product((0, 1), repeat=2):
            k = kernel(draft[0], q, p)[out[0]] * kernel(draft[1], q, p)[out[1]]
            consumed[out] += consumed_prob * k
            independent[out] += F(1, 4) * k
    target = iid_group(p)
    for law in (consumed, independent, target): assert_law(law)
    adaptive.append({'selected':selected, 'p':p, 'consumed':consumed,
        'independent':independent, 'target':target,
        'consumed_tv':tv(consumed, target), 'independent_tv':tv(independent, target),
        'consumed_group_update':grpo_update(consumed),
        'independent_group_update':grpo_update(independent),
        'fresh_group_update':grpo_update(target)})

old, fresh, repaired = defaultdict(F), defaultdict(F), defaultdict(F)
regenerated = F(0)
for a, b in product((0, 1), repeat=2):
    oldprob = F(1, 2) * (F(1, 5) if b else F(4, 5))
    old[(a, 0, b)] += oldprob
    current_p = F(9, 10) if a else F(1, 5)
    fresh[(a, a, b)] += F(1, 2) * (current_p if b else 1 - current_p)
    if not a:
        repaired[(a, 0, b)] += oldprob
    else:
        regenerated += oldprob
        for newb in (0, 1):
            repaired[(a, 1, newb)] += oldprob * (current_p if newb else 1 - current_p)
for law in (old, fresh, repaired): assert_law(law)
feedback = {'old':old, 'fresh':fresh, 'repaired':repaired,
    'relabel_tv':tv(old, fresh), 'repair_tv':tv(repaired, fresh),
    'relabel_reward':sum(k[-1]*v for k,v in old.items()),
    'fresh_reward':sum(k[-1]*v for k,v in fresh.items()),
    'repair_reward':sum(k[-1]*v for k,v in repaired.items()),
    'expected_regenerated_action_calls':regenerated,
    'fresh_action_calls':2,
    'note':'Action calls only; excludes feedback re-execution, snapshots, prefill, donor generation, batching. No wall-clock speedup.'}

# Run the actual upstream acceptance function. Continuation is a categorical
# target sampler, emulating the published algorithm, not the vLLM/Ray runtime.
import torch
source = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('pinned_speculative_decoding', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(2)
g = torch.Generator().manual_seed(20261011)
n = 200000
qt = torch.tensor([.9, .1], dtype=torch.float64)
pt = qt.flip(0)
x = torch.multinomial(qt, n, replacement=True, generator=g)
out = module.spec_cut(qt[x].log()[:,None], pt[x].log()[:,None], torch.ones(n,1,dtype=torch.bool), seed=20261012)
y = torch.multinomial(pt, n, replacement=True, generator=torch.Generator().manual_seed(20261013))
y[out['cut_idx'] == 1] = x[out['cut_idx'] == 1]
empirical = torch.bincount(y, minlength=2).double() / n
source_check = {'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'function':'spec_cut', 'samples':n, 'seeds':{'draft':20261011,'acceptance':20261012,'continuation':20261013},
    'conditional_acceptance':{str(z):float((out['cut_idx'][x==z]==1).double().mean()) for z in (0,1)},
    'empirical_law':empirical.tolist(), 'predicted_law':[float(naive[z]) for z in (0,1)],
    'absolute_error':max(abs(float(empirical[z])-float(naive[z])) for z in (0,1)),
    'limitation':'Imported acceptance only; target continuation simulated; no model, learner, Ray or vLLM.'}
checks = {'residual_exact':exogenous['residual_tv']==0,
    'independent_donor_exact':all(r['independent_tv']==0 for r in adaptive),
    'feedback_repair_exact':feedback['repair_tv']==0,
    'adaptive_counterexample':all(r['consumed_tv']>F(1,20) for r in adaptive),
    'naive_counterexample':exogenous['naive_tv']>F(1,20),
    'pinned_source_sanity':source_check['absolute_error']<.005}

def serialize(obj):
    if isinstance(obj, F): return {'fraction':str(obj),'value':float(obj)}
    if isinstance(obj, dict): return {str(k):serialize(v) for k,v in obj.items()}
    if isinstance(obj, (tuple,list)): return [serialize(v) for v in obj]
    return obj
result = {'exogenous':exogenous,'adaptive':adaptive,'feedback':feedback,
    'source_sanity':source_check,'checks':checks,'pass':all(checks.values()),
    'runtime':{'python':platform.python_version(),'torch':torch.__version__,
    'device':'cpu','elapsed_seconds':time.perf_counter()-START}}
print(json.dumps(serialize(result), indent=2, sort_keys=True))
assert all(checks.values()), checks
