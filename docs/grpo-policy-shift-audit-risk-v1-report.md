# Policy shift and finite-budget GRPO audit risk (v1)

## Result

This study extends VARE's four-member, degree-three GRPO audit witness to an action-distribution shift. At an equal budget of 144 clean-label calls, uniform three-of-four audits of behavior-policy groups had an exact optimal wrong-sign risk of 46.71% on the frozen two-world construction, versus 41.27% for full-group audits. Auditing only the known action stratum and its active three-label subset reduced that risk to 0.011%, but required an expected 4,096 behavior rollout groups to obtain 48 usable groups. Fresh target-policy groups with uniform triple audits reduced risk to 30.64% under the explicit assumption that the conditional clean-label law is unchanged.

The result isolates a cost interaction: action overlap determines how often old-policy data reaches a target-relevant action, while label-subset inclusion determines whether that record reveals the third-order verifier dependence. Label-call matching alone hides the rollout cost of target-stratified auditing. No general audit method or model-level improvement is established.

## Question and estimand

The behavior policy samples each of four binary actions independently with probability $1/4$. The target policy uses probability $3/4$. The action stratum is $a^*=1110$, whose mass changes from

$$
\mu(a^*)=3/256,\qquad \pi(a^*)=27/256,\qquad \pi(a^*)/\mu(a^*)=9.
$$

For the existing population-standardized binary group reward convention, let

$$
G(a,y)=\sum_i A_i(y)(a_i-1/4),\qquad
\chi_T(y)=(-1)^{\sum_{i\in T}y_i},\quad T=\{0,1,2\}.
$$

Because $\sum_i A_i(y)=0$, the local update $G$ is unchanged if $1/4$ in the score factor is replaced by the target Bernoulli probability. The exact coefficient at $(a^*,T)$ is $\alpha_T(a^*)=3/8-\sqrt{3}/4\approx-0.0580127$.

At $a^*$, define two clean-label laws

$$
P_s(y\mid a^*)=2^{-4}\bigl(1+s\epsilon\chi_T(y)\bigr),\qquad
s\in\{-1,+1\},\quad \epsilon=1/2,
$$

and let every other action stratum have uniform labels in both worlds. These worlds agree on every observed label subset that omits at least one member of $T$. Their target expected updates have opposite signs and magnitude

$$
|\theta_\pi|=\pi(a^*)\epsilon|\alpha_T(a^*)|\approx0.0030593.
$$

This is a two-point sign-decision experiment with an exactly known adversarial signal, not an estimate of real verifier-error prevalence.

## Exact decision-risk calculation

An audit record is informative about the world only if it both lands in $a^*$ and observes all of $T$. Let $r$ be this per-group probability. Conditional on an informative record, the parity has probabilities $(1+\epsilon)/2$ versus $(1-\epsilon)/2$ in the two worlds. For $n$ independent groups, the complete transcript reduces without loss to counts of positive parity, negative parity, and uninformative records. We enumerate these trinomial laws exactly and calculate

$$
\operatorname{TV}(P_+^n,P_-^n),\qquad
R^*_{\mathrm{sign}}=\frac{1-\operatorname{TV}(P_+^n,P_-^n)}{2}.
$$

The second quantity is the equal-prior Bayes error and, by the symmetry of this pair, the minimax wrong-sign probability. This is stronger than the union-bound lower bound $[1-nr\epsilon]_+/2$ for the same finite experiment.

For the active Walsh component $\theta=\pi(a^*)\alpha_T\mathbb{E}[\chi_T\mid a^*]$, the behavior-data Horvitz–Thompson observation is

$$
X=\frac{\pi(a^*)}{\mu(a^*)}\,\alpha_T\,\mathbf{1}[A=a^*]\,
\frac{\mathbf{1}[T\subseteq S]}{e_T(a^*)}\chi_T(Y).
$$

It is unbiased when $\mu(a^*)e_T(a^*)>0$ and the conditional label law transports. Under the two-world fixture, its one-group second moment is $\pi(a^*)^2\alpha_T^2/(\mu(a^*)e_T)$, hence

$$
\operatorname{MSE}(\bar X_n)=\frac{1}{n}\left[
\frac{\pi(a^*)^2\alpha_T^2}{\mu(a^*)e_T}-\theta^2\right].
$$

Fresh target-policy observations replace this second moment by $\pi(a^*)\alpha_T^2/e_T$; conditioning on the known action stratum replaces it by $\pi(a^*)^2\alpha_T^2$ while increasing rollout demand. These formulas quantify why propensity correction preserves expectation but cannot remove overlap-driven variance.

| Design | Groups in estimate | Exact informative-record rate | Optimal wrong-sign risk | Active-component MSE | Expected source groups |
|---|---:|---:|---:|---:|---:|
| Item-only behavior audit | 144 | 0 | 50.00% | $9.36\times10^{-6}$ | 144 |
| Pair-only behavior audit | 72 | 0 | 50.00% | $9.36\times10^{-6}$ | 72 |
| Full-group behavior audit | 36 | $3/256$ | 41.27% | $8.85\times10^{-5}$ | 36 |
| Uniform triple, behavior data | 48 | $3/1024$ | 46.71% | $2.66\times10^{-4}$ | 48 |
| Known action and subset, behavior data (oracle) | 48 | 1, conditional sample | 0.011% | $5.85\times10^{-7}$ | 4,096 expected |
| Uniform triple, fresh target data | 48 | $27/1024$ | 30.64% | $2.94\times10^{-5}$ | 48 |
| Known action and subset, fresh target data (oracle) | 48 | 1, conditional sample | 0.011% | $5.85\times10^{-7}$ | 455.1 expected |

All rows use 144 clean-label calls. The source-group count is separate: the oracle-stratum rows condition on seeing $a^*$ before spending labels, so they have low label cost only by consuming many rollout groups. Their allocation knows the hidden action and subset in advance and is an unattainable upper bound for an unknown verifier failure.

The MSE is for the target-weighted active Walsh component under the frozen parity pair. It is not full-update MSE over arbitrary reward laws. The item/pair zero estimator has lower MSE than full auditing because the target signal is small, while its sign risk stays at 50%. Thus MSE alone can favor making no directional claim when the practical decision is whether to accept an update.

## Mathematical boundary

The exact observation distinction factors into two coverage terms:

$$
r_{\mathrm{behavior}}=\mu(a^*)\,e_T(a^*),\qquad
r_{\mathrm{target}}=\pi(a^*)\,e_T(a^*),
$$

where $e_T(a)=\Pr(T\subseteq S\mid a)$ is the probability that an audit observes the full active subset. For uniform triples, $e_T=1/4$. Item and pair audits have $e_T=0$, so their transcript laws are identical for all sample sizes. Inverse action and audit-propensity weighting can make a component estimator unbiased when both terms are positive, but do not create information: small $\mu(a^*)e_T$ still produces high finite-budget sign risk. Historical reweighting also assumes $P(Y\mid A)$ is stable; if policy changes response semantics or verifier error conditional on action, fresh target audits or an explicit transport bound are required.

## Prior-art distinction and novelty assessment

This report does **not** claim a new importance-weighted estimator. Its estimator is the direct composition of action importance weighting and inverse audit-inclusion weighting. The narrow addition is an exact finite-sample GRPO sign-risk comparison where the behavior/target action ratio and the order-three label inclusion are varied together, with rollout demand reported beside clean-label cost.

Closest work limits the claim:

- [Verifier Errors in RLVR](https://www.elliott-thornley.com/src/writing/verifier-errors-rlvr/) uses known audit propensities to estimate reward-hacking gradients and explicitly discusses variance inflation and policy-distribution shift. Its target is the hack gradient under verifier feedback; this study uses a group-normalized GRPO update with a higher-order missing-label interaction.
- [Audit-First VAPO](https://arxiv.org/abs/2609.33662) freezes finite-budget audit decisions and certifies selected harmful-update risk; it renews its certificate when rollout or verifier conditions change. A broad claim about audit-certified update direction would overlap.
- [VStress](https://arxiv.org/html/2609.36958) allocates verifier calls across repeated channels using conditional information, cost, and a dependence-shift fallback. This study concerns coordinates within one group rather than verifier channels.
- [Noise-corrected GRPO](https://arxiv.org/abs/2510.18924) studies gradient correction under specified reward-noise models, while [Your Group-Relative Advantage Is Biased](https://arxiv.org/abs/2601.08521) studies finite-group advantage bias.

**Novelty assessment: narrow and unproven.** The exact result is useful as a risk/cost counterexample to treating matched clean-label count as matched information, but it is a Le Cam calculation on a constructed finite witness. The math is not a new general theory, there is no empirically selected allocator, and the literature overlap is close. Do not claim publication-level novelty, real verifier impact, or model improvement from this artifact.

## Reproduction and independent audit

The frozen protocol is [`protocols/grpo_policy_shift_audit_risk_v1.lock.json`](../protocols/grpo_policy_shift_audit_risk_v1.lock.json). Run:

```bash
python scripts/run_grpo_policy_shift_audit_risk_v1.py
python scripts/audit_grpo_policy_shift_audit_risk_v1.py
```

The runner uses a binomial-mixture decomposition for transcript TV and a direct truth-table coefficient calculation. The independent auditor recomputes the coefficient with 60-digit decimal arithmetic and sums the plus/minus/silent multinomial transcript laws directly. It also recomputes the finite-sample MSE formulas. Raw values and the independent audit are retained in [`results/grpo-policy-shift-audit-risk-v1/run-1/`](../results/grpo-policy-shift-audit-risk-v1/run-1/). The protocol was frozen before these scripts and the independent audit were run; this deterministic stress case was selected from earlier exact evidence and is not a preregistered confirmation study.

No GPU, paid API, external compute, or non-standard Python dependency was used. No model, optimizer step, real verifier, held-out capability task, or external reproduction was evaluated.

## Separate upstream engineering opportunity

The closest concrete VARE evidence is the compatibility-preserving optimizer configuration change in [RVL draft PR #89](https://github.com/mitukx/Recursive-Verification-Lag/pull/89): expose GRPO `weight_decay` while preserving the effective default. The local v2 report and all seven upstream CI checks are already retained. However, the PR currently has no maintainer comments or reviews, so it is **not** a maintainer-requested gap and this report does not present it as one. The targeted check of current VARE/RVL/TRL records surfaced no open maintainer request that this new result directly resolves; the existing TRL requests inspected are already covered by active proposals or prior VARE reproductions. No external PR or message was sent.

## Decision

**PIVOT the research claim; retain this as a bounded negative/diagnostic result.** It establishes that behavior/target action overlap and third-order audit inclusion jointly control finite-budget sign risk, and that label-cost-only comparisons can hide very large rollout demand. It does not establish a novel allocator or a usable guarantee under conditional verifier drift. The next useful theoretical step is a non-oracle optimal allocation rule under a declared rollout-plus-audit cost and unknown active interaction; stop if it reduces to standard importance-sampling variance allocation or is already covered. Do not run another synthetic variant merely to strengthen the same constructed witness.
