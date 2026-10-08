# Audit order needed to identify a finite GRPO update (v1)

## Result

For the four-member Bernoulli policy and binary, population-standardized rewards in the preceding exact-update witness, the local clipped-GRPO score update is a Boolean function of the four clean labels. Exact Walsh expansion shows that its conditional update function has degree three for every nonconstant action vector and is identically zero for the two constant action vectors.

This gives an exact audit-order result for this estimand:

- Knowing every action-conditioned one- and two-label joint marginal does **not** identify the expected update under arbitrary within-group reward dependence.
- Knowing every action-conditioned joint marginal through order three **does** identify it.
- Therefore the minimum marginal order is three for this particular four-member local update. This does not imply that all four labels must be audited together.

The result is a finite structural-identifiability statement. It says which population distributions could be distinguished in principle; it says nothing about the sample count needed to estimate their marginals accurately.

## Identification criterion

For a fixed action vector $a$, let $f_a(y)=G(a,y)$ for $y\in\{0,1\}^4$. Use the Walsh characters

$$
\chi_T(y)=\prod_{i\in T}(1-2y_i), \qquad
\alpha_T(a)=2^{-4}\sum_{y\in\{0,1\}^4}f_a(y)\chi_T(y).
$$

Then $f_a(y)=\sum_T\alpha_T(a)\chi_T(y)$. If all conditional label marginals through order $k$ are known, every $\mathbb{E}[\chi_T(Y)\mid a]$ with $|T|\le k$ is known. Thus if all coefficients of order greater than $k$ vanish, $\mathbb{E}[f_a(Y)\mid a]$ is identified.

Conversely, suppose $\alpha_T(a)\ne0$ for some $|T|>k$. For $0<\epsilon\le1$, define

$$
P_{\pm}(y\mid a)=2^{-4}(1\pm\epsilon\chi_T(y)).
$$

These are valid distributions. Summing out any coordinate outside an observed subset of size at most $k$ cancels the signed term, so the two laws have identical marginals up to order $k$. But their expected update values differ by $2\epsilon\alpha_T(a)$. This proves necessity. The action vector is observed, and the audit subset may depend on actions and public randomization, but not on unobserved labels. The sufficiency statement assumes every required action/subset stratum has positive audit probability; an arbitrary action-dependent sampler does not guarantee that coverage by itself.

This proof concerns identifiability from exact population marginals. Estimation error and cost are separate questions.

## Exact finite result

The runner enumerates all 16 action vectors and all 16 clean-label vectors per action, using exact rational plus $\sqrt{3}$ arithmetic. For target action `0001` and three-label subset `{0,1,2}`:

| Quantity | Exact value |
| --- | ---: |
| Walsh coefficient $\alpha_{012}$ | $-3/8+\sqrt{3}/4$ |
| Perturbation $\epsilon$ | $1/2$ |
| Conditional expected-update difference | $-3/8+\sqrt{3}/4\approx0.0580127$ |
| Probability of action `0001` | $27/256$ |
| Unconditional expected-update difference | $-81/2048+27\sqrt{3}/1024\approx0.00611853$ |
| TV of every observed marginal of order 0, 1, or 2 | $0$ |
| TV of the selected three-label marginal | $1/2$ |

The constructed reward laws differ only in the target action stratum; all other action strata may use the same law. Since the target action has positive probability, the conditional discrepancy induces the stated nonzero unconditional update difference. The independent exact auditor recomputes the Walsh degrees, marginal laws, and expected-update differences without importing the runner.

## Reproduction

From the repository root, with the Python standard library only:

```bash
python scripts/run_grpo_audit_order_characterization_v1.py
python scripts/audit_grpo_audit_order_characterization_v1.py
```

The locked protocol is [`protocols/grpo_audit_order_characterization_v1.lock.json`](../protocols/grpo_audit_order_characterization_v1.lock.json). Raw exact output and the same-host independent audit are retained in [`results/grpo-audit-order-characterization-v1/run-1/`](../results/grpo-audit-order-characterization-v1/run-1/). Both commands are added to CI.

## Relation to prior work and limits

Recent work already studies group-correlated verifier reward noise and its effect on GRPO advantages, finite-group advantage bias, risk-controlled selective updates, and correlation-aware audit allocation. This result should be treated as a narrow extension of VARE's earlier synthetic witness: it characterizes the order of joint clean-label information needed for one precisely defined local update. It does not introduce a learning method or audit allocator. Whether this estimand-specific characterization is sufficiently distinct from prior work remains unresolved.

No trained model, optimizer step, verifier-learning process, finite-sample audit budget, language task, or capability measure was evaluated. The outcome is synthetic exact theory; it is not a real-world prevalence estimate, policy-improvement result, external reproduction, or novelty certification. A practical study would need to measure marginal-estimation variance and audit cost, then test a distinct real-verifier setting without treating this synthetic construction as evidence of model capability.

## Primary sources reviewed

- [An Imperfect Verifier is Good Enough: Learning with Noisy Rewards](https://arxiv.org/abs/2604.07666)
- [Your Group-Relative Advantage Is Biased](https://arxiv.org/abs/2601.08521)
- [VStress: Correlation-Aware Auditing and Adaptive Budget Allocation for Repeated Verifiers](https://arxiv.org/abs/2609.36958)
- [Audit-First VAPO: Risk-Certified Selective Updates under Imperfect Verification](https://arxiv.org/abs/2609.33662)
- [Verifier Errors in RLVR: Reward Hacking, Limits of Feedback, and Selective Control](https://arxiv.org/abs/2609.35677)
