import itertools,json,math,sys
from fractions import Fraction
sys.path.insert(0,'/Users/user/Documents/Codex/2026-10-13/var/work/VARE-eval-stats/src')
from vare.format_vs_correctness_v2 import _pass_at,RUN_SEEDS,EVAL_SAMPLES
from vare.eval_stats import unbiased_pass_at_k
TASK={'task_id':'hand-task','expression':'1 + 1','prompt':'compute 1 + 1'}
def pooled(counts,k):
    grouped={('format_sft',16,'confirm',TASK['task_id'],seed): [{'output':'#### 2' if i<c else 'nope'} for i in range(EVAL_SAMPLES)] for seed,c in zip(RUN_SEEDS,counts,strict=True)}
    return _pass_at(grouped,[TASK],arm='format_sft',checkpoint_step=16,split='confirm',k=k)
def per_seed(counts,k): return sum(unbiased_pass_at_k(4,c,k) for c in counts)/5
vectors={'concentrated':[4,0,0,0,0],'spread_same_total':[1,1,1,1,0],'spread_less_total':[1,1,1,0,0]}
observed={name:{'counts':c,'legacy_pooled_pass1':pooled(c,1),'seed_mean_pass1':per_seed(c,1),'legacy_pooled_pass4':pooled(c,4),'seed_mean_pass4':per_seed(c,4)} for name,c in vectors.items()}
max_diff=0.
for counts in itertools.product(range(5),repeat=5):
    exact_seed=Fraction(sum(counts),20)
    exact_pool=1-Fraction(20-sum(counts),20)
    assert exact_seed==exact_pool
    max_diff=max(max_diff,abs(unbiased_pass_at_k(20,sum(counts),1)-per_seed(counts,1)))
print(json.dumps({'handmade_task_only':True,'cases':observed,'vectors_checked':5**5,'pass1_exact_rational_equality':True,'maximum_float_rounding_difference':max_diff,'pass4_rank_reversal':observed['concentrated']['legacy_pooled_pass4']>observed['spread_less_total']['legacy_pooled_pass4'] and observed['concentrated']['seed_mean_pass4']<observed['spread_less_total']['seed_mean_pass4']},indent=2))
