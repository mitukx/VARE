# Partial clean-label audits for a four-member GRPO update (v1)

## Question and result

Does observing a random subset of a four-member reward group recover the fixed local GRPO score update more efficiently than auditing every label, when the clean-label-call budget is held fixed?

For this exact four-member binary-reward construction, a three-label Horvitz–Thompson/Walsh estimator is unbiased for every joint reward law. That fact did **not** translate into lower MSE per label call: at the fixed budget of 12 clean labels, its sample-mean MSE exceeded full-group auditing in all 387 law cases where the full-audit denominator was nonzero. The median ratio was 1.0103 and the range was 1.0103–1.2229. Lower-order one- and two-label Walsh projections had slightly smaller MSE in 385/387 cases, but they are biased and miss the constructed third-order signal; their MSE advantage is not evidence that they preserve the update direction.

This is a negative result for a generic “three labels are cheaper and equally informative” recommendation. It is exact evidence for a small, specified estimand, not a general audit policy or model result.

## Frozen estimand and comparison

The policy samples four Bernoulli(1/4) actions independently. For binary group rewards $y\in\{0,1\}^4$, the target is the local score update

$$G(a,y)=\sum_{i=1}^4 A_i(y)(a_i-1/4),$$

where $A(y)$ is the population-standardized reward with constant-reward groups mapped to zero. The action vector is observed. Audit size $m$ uniformly samples an $m$-coordinate subset and uses the inverse-inclusion estimator for Walsh terms of degree at most $m$. Its coefficients are supplied exactly from the target truth table. This oracle-coefficient assumption isolates audit information and is not a deployable learned estimator.

The frozen 390 conditional reward laws comprise three independent-Bernoulli cases, three common-mode cases, and 384 third-order parity cases. We exactly enumerate 1,560 law-by-audit-size rows. The matched budget is 12 labels: 12, 6, 4, and 3 independent groups for $m=1,2,3,4$. The primary metric is the fixed-budget sample-mean MSE ratio

$$\frac{\operatorname{Var}(\widehat G_m)/(12/m)+\operatorname{Bias}(\widehat G_m)^2}{\operatorname{Var}(G)/3}.$$

The exact enumeration uses rational plus $\sqrt 3$ arithmetic. Separately, a fixed six-seed simulation estimates wrong-or-zero update direction on the two frozen parity worlds. Computation cost is recorded as the number of subset choices $\binom 4m$ and Walsh terms $2^m$ per group; end-to-end latency is not compared.

## Quantitative evidence

| Labels per group | Exact unbiased for all laws? | Cases with lower fixed-budget MSE than full audit | Median MSE ratio | Range |
|---:|:---:|---:|---:|---:|
| 1 | No | 385 / 387 | 0.99487 | 0.99487–1.47845 |
| 2 | No | 385 / 387 | 0.99487 | 0.99487–1.30104 |
| 3 | Yes | 0 / 387 | 1.01026 | 1.01026–1.22291 |
| 4 | Yes; reference | 387 / 387 equal | 1.00000 | 1.00000 |

Three zero-variance full-audit cases have undefined ratios and are excluded from the denominators. The 3-label estimator's identity is checked pointwise for all action/reward vectors and audited triples. For every law, its variance equals the full-update variance plus the expected subset-randomization variance. The 1- and 2-label estimators are truncated Walsh projections, so bias is retained in MSE rather than hidden.

For the fixed 12-label budget on the two parity worlds, the simulated wrong-or-zero direction rates were:

| Labels per group | Parity world $-$ | Parity world $+$ |
|---:|---:|---:|
| 1 | 52.80% [52.17%, 53.42%] | 53.18% [52.55%, 53.80%] |
| 2 | 58.04% [57.43%, 58.66%] | 58.02% [57.41%, 58.64%] |
| 3 | 52.47% [51.85%, 53.10%] | 52.01% [51.39%, 52.63%] |
| 4 | 57.63% [57.01%, 58.24%] | 57.91% [57.30%, 58.53%] |

Each interval is a pooled Wilson 95% interval over 24,576 independent simulated replicates (six seeds × 4,096). Direction estimates are descriptive; no direction-superiority decision rule was frozen. The small underlying policy-weighted target magnitude is about 0.00306, which makes this an intentionally difficult sign diagnostic. In this one synthetic case, three-label sampling had a lower sign-error rate than full audits but a higher exact MSE; this does not define a robust operating advantage.

The lower-order identifiability witness also gives a precise limit. At action `0001`, two reward laws agree on every observation using at most two coordinates but their conditional update means differ by $-3/8+\sqrt3/4\approx0.05801$. Any estimator with that common observation law has worst-case conditional absolute bias at least half the gap, $-3/16+\sqrt3/8\approx0.02901$. Under the frozen action probability $27/256$, the policy-weighted gap is approximately 0.00612 and the corresponding lower bound is approximately 0.00306.

## Novelty and relation to prior work

The estimator is the direct inverse-inclusion (Horvitz–Thompson) application to the already derived Walsh coefficients; it is not a new estimator. The potentially useful evidence is the explicit finite cost/MSE and direction comparison for this particular group-relative update, including cases where MSE and sign accuracy disagree. The study is too narrow, uses oracle coefficients, and has no real verifier or learner; it does not establish a publishable general method or practical audit recommendation.

The novelty bar is high. *Noise-corrected GRPO* already analyzes reward-noise correction and unbiased group-policy gradients ([arXiv](https://arxiv.org/abs/2510.18924)). *Your Group-Relative Advantage Is Biased* studies finite-group advantage bias ([arXiv](https://arxiv.org/abs/2601.08521)). *VStress* explicitly compares fixed-budget verifier breadth, repeated calls, correlation-aware allocation, shift fallback, and downstream RLVR ([arXiv](https://arxiv.org/abs/2609.36958)). *Audit-First VAPO* treats finite-budget verification as update-direction admission and compares against matched-random, confidence, and noise-correction baselines ([arXiv](https://arxiv.org/abs/2609.33662)). The review checked the primary arXiv records and VStress full text available on 2026-10-09; it is focused, not exhaustive. Those works make generic claims about noisy-verifier auditing, budget allocation, or update direction unsuitable as a novelty claim here.

Repository PR #1 was also inspected without merging or modifying it. It adds a held-out proxy false-accept audit and opt-in promotion checks; it does not implement this within-group partial-label estimator. The two efforts address adjacent but distinct boundaries.

## Reproduction and retained failures

```bash
python scripts/run_grpo_partial_audit_estimator_v1.py
python scripts/audit_grpo_partial_audit_estimator_v1.py
```

The exact experiment protocol is [`protocols/grpo_partial_audit_estimator_v1.json`](../protocols/grpo_partial_audit_estimator_v1.json). Hash-bound implementation errata record a missing order-4 inclusion-table entry and an unnecessarily slow first implementation; neither attempt generated outcome metrics. Both failed attempts remain under [`results/grpo-partial-audit-estimator-v1/`](../results/grpo-partial-audit-estimator-v1/). The successful raw enumeration, same-host independent audit, post-hoc summary/interval audit, and hash manifest are retained in [`run-1/`](../results/grpo-partial-audit-estimator-v1/run-1/). The independent implementation recomputes all 1,560 exact rows and the seed-level direction counts. This is a same-host audit, not outside reproduction. [GitHub Actions run 37860143100](https://github.com/mitukx/VARE/actions/runs/37860143100) passed on commit `825381d`: 187 tests passed, 12 skipped, and all three exact GRPO reproduction/audit steps plus the demo passed. Local pytest is not installed in the Python 3.9.6 environment.

No GPU, model, paid API, external compute, or additional dependency was used. No actual optimizer step or capability measure was evaluated.

## Decision

**STOP** treating random three-of-four audits as a generally more efficient estimator for this fixed update: exact unbiasedness came with worse fixed-budget MSE in every nondegenerate case in the frozen grid. **PIVOT** the active research question to the remaining highest-value gap: a feasible, independently evaluated real-model policy update. Before freezing such a study, identify a cached base model, task with nontrivial base success, CPU update/save/reload path, and untouched task-success evaluation that all fit a strict no-spend runtime cap. If those gates fail, retain the null/negative feasibility result and return to a concrete upstream correctness defect rather than extending synthetic audit variants.
