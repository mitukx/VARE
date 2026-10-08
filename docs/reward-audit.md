# Held-out verifier false-acceptance audit (experimental)

This optional post-training control checks whether proxy-verifier approvals survive an
**independent trusted evaluation**. It focuses on a failure mode relevant to
RL with imperfect reward signals: a candidate's apparent gain may come from
exploiting the verifier rather than gaining capability.

## Data contract

Each held-out example has a stable task ID, task family, a Boolean proxy
accept/reject and an independently determined trusted accept/reject. Collect
these labels on a disjoint, frozen evaluation split; retain the raw evaluator
outputs, hashes, policy/verifier revisions, task IDs and experiment protocol
separately. The audit cannot infer or certify independence from labels alone.
Never train on these examples or use the trusted labels to tune the candidate.

`audit_reward_labels` estimates the conditional false-acceptance frequency
`P(trusted_fail | proxy_pass)`. It reports a one-sided Hoeffding upper bound
for all proxy-positive examples and for every observed task family with a
Bonferroni adjustment over the observed slices. The bound requires independent
representative examples and a frozen proxy; adaptive task selection, correlated
rollouts, repeated peeking and model-selection on the same holdout invalidate
the nominal confidence level. In such settings, collect a new sealed
evaluation set or use a justified sequential / clustered inference method.

## Strict promotion

```python
from vare.config import PromotionConfig
from vare.promotion import PromotionGate
from vare.types import EvaluationReport

gate = PromotionGate(PromotionConfig(
    paired_confidence_gate=True,
    require_complete_slices=True,
    require_measured_disagreement=True,
    require_reward_audit=True,
    reward_audit_max_false_accept_ucb=0.25,
    reward_audit_min_proxy_positives=32,
    reward_audit_alpha=0.05,
))
# On the *candidate* EvaluationReport.metadata, attach:
# "per_task_scores": {"heldout-1": 1.0, ...}  # same IDs on incumbent
# "verifier_disagreement_measured": True       # only if actually measured
# "reward_audit": [
#   {"task_id": "heldout-1", "family": "coding",
#    "proxy_pass": True, "trusted_pass": True},
#   ...
# ]
# decision = gate.decide(incumbent_report, candidate_report)
```

Strict mode is opt-in because existing examples do not yet collect independent
trusted/proxy paired labels. Setting `verifier_disagreement_measured=True`
without a measurement is not evidence and violates this contract. The current
RVL-GRPO example cannot enable strict mode unchanged: its evaluation adapter
uses a trusted task score only and returns a placeholder disagreement. A future
adapter must explicitly collect both independent signals.

## Research protocol

1. Freeze the task split, reward proxy, reference checker, IDs, seeds, decision
   policy, wall/compute budget, and minimum detectable effect before running.
2. Compare a fixed-verifier baseline and an audit-constrained candidate under
   matched budget and independent evaluation.
3. Report false accepts, false rejects, proxy-positive count, family slices,
   trusted held-out success, sampling costs, and rejected promotions.
4. Keep negative runs and raw trajectories. Do not call audit acceptance a
   capability gain; only downstream held-out evaluation can support that.
5. Never repeatedly tune and re-test on the same sealed evaluation set.

This is **CPU-testable statistical plumbing**, not evidence of improved RL
training. A positive system claim requires actual independent model/agent
outcomes with a fixed-compute comparison.
