# Partial-label identification of a clipped GRPO update (synthetic v1)

## Finding

The frozen finite witness passes its exact checks. Two worlds have the same policy action distribution and the same clean-label distribution for every member **conditional on the complete four-action group**. Therefore any audit that observes the whole action group but only one clean label has identical observation laws in both worlds, including an audit that selects a member based on the actions. Yet the expected local gradient of the clipped GRPO surrogate differs.

This is a mathematical existence counterexample, not an estimate of how often real verifiers have such dependence. The construction was selected after an exploratory finite linear-program search. The protocol lock discloses that selection; the result is an exact verification of a chosen witness, not a preregistered confirmatory study.

## Setup

Each group has four independent Bernoulli actions with $p=1/4$. For binary clean rewards $y_i$, define population-standardized group advantages

\[
A_i(y)=\frac{y_i-\bar y}{\sqrt{\frac14\sum_j(y_j-\bar y)^2}},
\]

and set all $A_i=0$ when the group reward variance is zero. The per-group score gradient for a shared Bernoulli-logit parameter is

\[
G(a,y)=\sum_i A_i(y)(a_i-p).
\]

At the old policy, the importance ratio is 1. With clipping ε=0.2, that point is inside the clip interval, so the local derivative of the **maximized** clipped surrogate is $G$. A minimized loss uses the opposite sign. This is a local derivative, not a finite optimizer step.

For every action vector except `0001`, both worlds flip each action bit independently with probability 1/5 to produce clean rewards. Conditional on action vector `0001`, the worlds differ:

| Clean reward vector | World A | World B |
| --- | ---: | ---: |
| `0000` | 1/5 | 0 |
| `0001` | 1/5 | 7/10 |
| `0011` | 1/5 | 0 |
| `0101` | 1/5 | 0 |
| `1001` | 1/5 | 0 |
| `0110` | 0 | 1/10 |
| `1010` | 0 | 1/10 |
| `1101` | 0 | 1/10 |

In both worlds, for every complete action vector and member, $P(y_i=a_i\mid a)=4/5$. Thus per-member audits cannot see the dependence difference even after conditioning on all actions.

## Exact outcome

| Quantity | World A | World B | Difference B − A |
| --- | ---: | ---: | ---: |
| Expected (G) | 0.726819846 | 0.739872704 | **0.013052858** |
| Exact form | $10881/32000 + (3573/16000)\sqrt{3}$ | $8181/32000 + (4473/16000)\sqrt{3}$ | $-27/320 + (9/160)\sqrt{3}$ |

The target action group occurs with probability $27/256$. Conditional expected gradients there are $3/5+\sqrt{3}/5$ in World A and $-1/5+11\sqrt{3}/15$ in World B. The gap is small, but strictly nonzero: $9(2\sqrt{3}-3)/320$.

The total-variation distance between one-item audit laws that include the complete action vector and item index is exactly 0. The total-variation distance for the full group observation is $27/320$. This proves that item-only observations do not identify this expected update for the constructed law, while full-group observations contain information about it. It does not show that full-group audits are generally more cost-effective or that a practical auditor should always buy them.

The exact enumerator uses rational probabilities and $a+b\sqrt{3}$ arithmetic. A separately implemented direct enumerator reconstructed the metrics and marginal constraints; it ran on the same machine, so this is not outside replication. Details and hashes are retained in [`run-1`](../results/grpo-expected-update-audit-identifiability-v1/run-1/).

## Reproduction

From the repository root, using only the Python standard library:

```bash
python scripts/run_grpo_expected_update_audit_identifiability_v1.py
python scripts/audit_grpo_expected_update_audit_identifiability_v1.py
```

The [protocol](../protocols/grpo_expected_update_audit_identifiability_v1.lock.json) records the construction, estimand, selection disclosure, pass rule, and limitations.

## Relation to prior work and limits

This result sits close to existing research. Plesner et al. analyze several group-level reward-noise structures and their effects on conditional advantages; Yang et al. study finite-group relative-advantage bias; recent paired-rollout work distinguishes reward-contrast variance from gradient variance; VStress studies information- and cost-aware auditing of correlated verifiers. Reward-hacking work also shows that verifier reward and intended capability can diverge. The present construction is narrower: a partial-label observation channel with **the full action vector visible**, identical per-member conditional label laws, and a non-identifiable expected local clipped-GRPO update. A focused literature scan cannot certify novelty, and no broad algorithmic novelty is claimed.

The difference is small and comes from a deliberately chosen synthetic reward dependence. No model, optimizer state, task-success metric, verifier-training process, reward-model drift, or real-world verifier was evaluated. The result does not establish a policy improvement, capability gain, audit-allocation policy, or deployment recommendation. Independent outside reproduction and a non-synthetic follow-up remain open.

Related primary sources:

- [An Imperfect Verifier is Good Enough: Learning with Noisy Rewards](https://arxiv.org/abs/2604.07666)
- [Your Group-Relative Advantage Is Biased](https://arxiv.org/abs/2601.08521)
- [Luck Is Not Skill: When Do Paired Rollouts Help Group-Relative RL of LLM Agents?](https://arxiv.org/abs/2609.24144)
- [VStress: Correlation-Aware Auditing and Adaptive Budget Allocation for Repeated Verifiers](https://arxiv.org/abs/2609.36958)
- [A Pre-Registered Causal Partition of Self-Consistency Elicitation and Reward Design in RLVR](https://arxiv.org/abs/2606.05932)
