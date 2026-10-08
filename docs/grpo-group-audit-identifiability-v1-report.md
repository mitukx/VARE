# GRPO group-audit identifiability v1

## Result

**The frozen mechanism prediction passed.** With the same 200 trusted label calls per batch, item-only audits could not distinguish two worlds with different clean group-normalized advantage signals; auditing complete two-member groups distinguished them perfectly. This is an exact synthetic result about the observability of the per-group signal. It is not an LLM training, optimizer, or capability result.

## Hypothesis and construction

The [protocol](../protocols/grpo_group_audit_identifiability_v1.lock.json) was committed before execution. Every group has proxy rewards `(1,0)`. In world A the clean rewards are `(Z,Z)`; in world B they are `(Z,1-Z)`, with fair `Z`. In both worlds either fixed member has clean-reward marginal 0.5. Thus uniform one-item audits and audits of the proxy-positive member observe the same Bernoulli(0.5) law. The exact total-variation distance between the 200-label count distributions is zero.

For binary group rewards, the centered population-standardized advantage is zero when both labels match. It is `(1,-1)` or `(-1,1)` when labels differ. Therefore the clean per-group advantage is always zero in A and always nonzero in B. Item-level inverse-propensity weighting can recover item marginals, but no function of observations with the same distribution can identify which joint law generated them. A complete group audit observes the dependence directly.

## Frozen experiment

- 2,000 seeds; one batch per world per seed.
- 200 clean-label calls per batch in every arm.
- Uniform item audit: one randomly chosen member from each of 200 groups.
- Proxy-positive item audit: the verifier-positive member from each of 200 groups.
- Group-atomic audit: both members from 100 randomly chosen groups.
- Primary: balanced accuracy for distinguishing A versus B, plus exact total-variation distance for item-only observation laws.
- The training policy and proxy rewards are identical between worlds; the audit classifier sees audit data only.

## Outcome

| Audit design | World A accuracy | World B accuracy | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Uniform item | 51.95% | 46.45% | 49.20% |
| Proxy-positive item | 51.40% | 48.60% | 50.00% |
| Group atomic | 100.00% | 100.00% | 100.00% |

The theoretical one-item observation-law TV was exactly 0. The group-atomic classifier was correct on all 4,000 world-seed batches. The 2,000-seed raw bundle and hash manifest are retained in [`results/grpo-group-audit-identifiability-v1/run-1`](../results/grpo-group-audit-identifiability-v1/run-1/). The initial auditor replay attempt failed because it used a different seeded-random primitive; that failed development attempt and cause are retained in [`audit-development-failure-v1.json`](../results/grpo-group-audit-identifiability-v1/audit-development-failure-v1.json). The corrected independent replay passed and is recorded in the bundle's `audit.json`.

The first CI run also exposed a test-writing error: a 1,000-sample Bernoulli mean was incorrectly asserted to equal its expectation exactly (observed 497/1,000). That test was replaced with exact finite-law enumeration without changing the protocol or experiment bundle; the failure is retained in [`ci-development-failure-v1.json`](../results/grpo-group-audit-identifiability-v1/ci-development-failure-v1.json). The final GitHub CI run [37852234082](https://github.com/mitukx/VARE/actions/runs/37852234082) passed with 161 tests passed and 12 skipped.

## Interpretation and limits

This proves an information limitation for the constructed two-member setting: marginal reward labels do not determine the joint reward law or the realized normalized group signal. It does **not** prove that a population-expected policy gradient is unidentifiable in every setting; for example, symmetric World B has canceling advantage signs under additional exchangeability assumptions. It does not establish a general advantage for group auditing, a cost-optimal audit policy, or a language-model capability improvement.

The closest overlap is substantial. Recent RLVR work already studies asymmetric and group-correlated reward corruption, and a September 2026 paper studies adaptive allocation among correlated repeated verifier channels. This report's narrower target is partial *clean-label observation of within-group dependence*. Novelty beyond that precise separation is unproven. The next meaningful step, if pursued, is to instantiate the construction with actual score-function vectors and a frozen clipped GRPO update, then check whether the joint-label distinction changes update direction under non-canceling policy features.

## Reproduction

From the repository root:

```bash
python scripts/run_grpo_group_audit_identifiability_v1.py
python scripts/audit_grpo_group_audit_identifiability_v1.py results/grpo-group-audit-identifiability-v1/run-1
```

The simulation and independent audit use only Python's standard library. `pytest` was not installed in the available local Python environments; the two focused test functions were invoked directly, all three files compiled, and the independent bundle audit passed.
