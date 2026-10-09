# Non-oracle GRPO audit under policy shift: reduction result (v1)

## Decision

**STOP this method line for novelty.** Once rollout generation and clean-label acquisition are priced together, the non-oracle design is a standard unequal-probability sampling problem for a finite-population update functional. Its unbiased estimator is a Horvitz–Thompson (HT) estimator with joint inclusion probabilities; allocation by known or pilot-estimated stratum variance is Neyman allocation / optimal design; sign certification is a sequential testing problem. The order-three GRPO term changes the required inclusion event, but does not create a new estimator or allocation principle.

The CPU calculation adds a useful boundary to VARE's prior diagnostic: label-call matching can make a target-action-stratified audit look attractive while hiding hundreds or thousands of generated groups. Full-group auditing beats non-oracle random triples at matched label-only cost in the frozen behavior-policy fixture. Neither result shows an efficient practical method or a model improvement.

## Precise target and observables

Let \(Z\) denote the complete rollout record available before clean-label auditing (including action/configuration \(A\), prompt, and logged policy provenance), \(Y\in\{0,1\}^G\) its clean verifier labels, \(q_b(Z)\) the behavior rollout distribution, and \(q_t(Z)\) the target-policy distribution. Let \(g(Z,Y)\in\mathbb{R}^d\) be the unclipped clean group-relative score update at the declared policy point. The target update is

\[
\theta_t=\mathbb{E}_{Z\sim q_t,\,Y\sim P_t(\cdot\mid Z)}[g(Z,Y)].
\]

An audit observes \(Z\), behavior-verifier outputs contained in \(Z\), a selected subset \(S\subseteq[G]\), and clean labels \(Y_S\). The selection law must be logged before the labels in \(S\) are revealed. In an adaptive design, it may depend on the clean audit history \(H\) available before the current selection. Write \(e_T(Z,H)=\Pr(T\subseteq S\mid Z,H)\) for joint inclusion of a label subset \(T\). Cost is

\[
C=c_R N_{\rm generated}+c_L\sum_i |S_i|,
\]

where \(c_R\) is the cost of generating one group and \(c_L\) is the cost of obtaining one clean label. Under rejection sampling, obtaining one audited group conditional on action \(a\) costs \(c_R/q(a)+c_L|S|\) in expectation. This accounts for rollout overlap as well as audit calls.

## Reduction to established design theory

For binary labels, expand each projected update direction \(u^\top g(Z,Y)\) in the Walsh basis:

\[
u^\top g(Z,Y)=\sum_{T\subseteq[G]}\alpha_T(Z)\chi_T(Y),\qquad
\chi_T(Y)=(-1)^{\sum_{j\in T}Y_j}.
\]

Under **conditional transport** \(P_t(Y\mid Z)=P_b(Y\mid Z)\) and overlap \(q_b(Z)>0\) wherever \(q_t(Z)>0\), the direct unbiased estimator is

\[
\widehat\theta_u=\frac1n\sum_{i=1}^n\frac{q_t(Z_i)}{q_b(Z_i)}
\sum_T\alpha_T(Z_i)\frac{\mathbf 1[T\subseteq S_i]}{e_T(Z_i,H_i)}\chi_T(Y_i).
\]

For a design whose conditional inclusion probabilities are logged, \(\mathbb{E}[\mathbf 1[T\subseteq S]\chi_T/e_T\mid Z,H,Y]=\chi_T(Y)\). With adaptive history, this is a sequential HT/martingale argument. Covariances between two terms \(T,U\) depend on joint inclusion probabilities for \(T\cup U\), as in ordinary unequal-probability sampling. The degree-three interaction only says a clean order-three marginal is needed; it does not change the HT identity. The required overlap is \(q_b(Z)>0\) wherever \(q_t(Z)>0\); action-only weights are sufficient only when the remaining target-relevant record distribution is unchanged conditional on action.

For strata (h) with target mass (W_h), per-group variance (\sigma_h^2), and all-in cost (c_h), minimizing

\[
\sum_h \frac{W_h^2\sigma_h^2}{n_h}
\quad\text{subject to}\quad \sum_h c_h n_h\le C
\]

gives by a Lagrange multiplier

\[
n_h=\frac{C\,W_h\sigma_h/\sqrt{c_h}}
{\sum_j W_j\sigma_j\sqrt{c_j}}.
\]

This is cost-adjusted Neyman allocation. With vector updates, replacing scalar variance by (u^\top\Sigma_h u) is standard (c)-optimal design for a declared decision direction; minimizing trace is the usual A-optimal objective. If the proposal distribution itself is selected, minimizing importance-sampling second moment gives the standard variance-optimal proposal proportional to target density times the square root of conditional second moment. Unknown variances require pilot/adaptive estimation and inherit the familiar exploration, positivity, and estimation-error tradeoffs.

For a two-world sign decision, the optimal error is (R^*=(1-\operatorname{TV}(P_+,P_-))/2). Adaptive audits are sequential experiments: the chain rule adds conditional KL information per selected query, while standard sequential tests convert accumulated evidence into a decision. In VARE's parity witness, an informative observation occurs with probability (q(A=a^*)e_T(a^*)), and its conditional parity KL is

\[
D_{\rm KL}(P_+\Vert P_-)=\epsilon\log\frac{1+\epsilon}{1-\epsilon}.
\]

Thus third-order inclusion and policy overlap multiply the rate of information acquisition; this is a problem-specific application of standard experiment design and testing theory, not a new lower-bound technique.

### Non-oracle adaptive baseline

Suppose the active third-order subset is unknown among the four triples and every available pre-audit proxy observes only strict subsets whose distributions are identical across those alternatives. Then the proxy carries zero mutual information about which triple is active. For one triple per audit, any proxy-only allocation has inclusion probabilities (p_T) with \(\sum_Tp_T=1\), so \(\min_Tp_T\le1/4\); the uniform triple design attains this maximin bound. It is the non-oracle adaptive baseline for this information structure. After clean triples are observed, adaptation can use the history; that is ordinary sequential experimental design and is not claimed to be dominated by the static rule.

## Prior-art comparison

- **PAIR (2026):** Represents the unclipped leave-one-out group gradient as a pairwise (U)-statistic and corrects adaptive rollout completion with logged joint edge-inclusion probabilities. PAIR concerns the cost of generating unfinished rollout endpoints; VARE's question concerns acquiring clean labels for already generated groups. The mechanism is therefore not identical, but PAIR establishes the same design-unbiasedness principle and makes an RLVR-specific novelty claim for inclusion correction untenable without a stronger result. [Paper](https://arxiv.org/abs/2608.11368)
- **VIP (2026):** Allocates rollout counts across prompts by predicted gradient variance under a rollout budget, with a convex allocation rule. This is the prompt-stratified analogue of cost-aware Neyman allocation. [Paper](https://arxiv.org/abs/2602.01601)
- **Audit-First VAPO (2026):** Freezes an observation-only accept/appeal/abstain trace before joining clean labels and certifies selected harmful-update risk under a finite verification budget; it renews the certificate after rollout or verifier changes. Its decision unit differs from within-group label subsets, but risk certification and pre-label audit policies are direct overlap. [Paper](https://arxiv.org/abs/2609.33662)
- **VStress (2026):** Allocates calls across repeated verifier channels using dependence/information and cost, and includes a dependence-shift fallback. It studies channel redundancy rather than higher-order label subsets, but the adaptive information-per-cost objective is already present. [Paper](https://arxiv.org/abs/2609.36958)
- **Imperfect/noisy verifier RLVR:** Existing work derives reward/gradient corrections under explicit verifier noise models and uses online audits to estimate error rates. Such corrections do not identify an arbitrary higher-order conditional label law from behavior data, but composing them with HT does not by itself make a novel audit allocator. [Noisy-reward RLVR](https://arxiv.org/abs/2510.00915)
- **Classical design and sign testing:** HT inclusion weighting, Neyman allocation, finite-sample two-point testing, and sequential sign identification predate RLVR. The VARE construction inherits these tools; it does not establish a new statistical primitive.

## Frozen CPU calculation

The protocol was locked before the runner and independent audit: [`grpo_audit_cost_reduction_v1.lock.json`](../protocols/grpo_audit_cost_reduction_v1.lock.json). It reuses the existing four-member witness only to test the distinct claim about **total rollout-plus-label cost and non-oracle allocation**. The target action has mass (27/256\); behavior masses are (27/256,9/256,3/256\), giving target-to-behavior ratios (1,3,9\). The unknown active triple is not given to non-oracle designs. One unit of clean-label acquisition costs 1; rollout cost is varied between 0 and 1 label-call unit; total budget is 144.

At overlap ratio 9 and (c_R/c_L=0), exact two-world optimal sign errors were:

| Design | Audited groups | Expected generated groups | Clean labels | Optimal sign error | HT active-component MSE |
|---|---:|---:|---:|---:|---:|
| Behavior, uniform triple | 48 | 48 | 144 | 46.71% | \(2.660\times10^{-4}\) |
| Behavior, full group | 36 | 36 | 144 | 41.27% | \(8.848\times10^{-5}\) |
| Target-policy, uniform triple | 48 | 48 | 144 | 30.64% | \(2.938\times10^{-5}\) |
| Target action stratum, uniform triple | 48 | 455 | 144 | 3.65% | \(2.925\times10^{-6}\) |
| Target action stratum, full group | 36 | 341 | 144 | 0.070% | \(7.799\times10^{-7}\) |
| Target action stratum, oracle active triple | 48 | 455 | 144 | 0.011% | \(5.849\times10^{-7}\) |

The last row is an unattainable upper bound because it is told which third-order subset carries the signal. Target action stratification is observable and non-oracle, but rejection sampling makes its rollout demand explicit. Full-group auditing has a lower sign error than non-oracle random triples at the same label-only budget in this fixture.

When a generated group costs one label-call unit, behavior uniform-triple and full-group errors at overlap ratios (1,3,9\) become (33.90/19.72\%, 43.16/33.47\%, 47.49/42.93\%\), respectively. Target-policy uniform triples stay at 33.90% across those behavior-overlap settings because they are sampled from the target distribution. Under the same total budget, target-action-stratified uniform triples use 11 audited groups (33 clean labels) and 104 expected generated groups, with 20.32% error; the oracle active-triple rule has 3.43% error. Unused budget from integer group rounding is retained in the raw record rather than silently spent.

The raw table covers 42 design/cost/overlap rows. A separate auditor enumerates the positive/negative/silent transcript multinomial directly, independently of the runner's binomial-mixture TV computation; all 42 sign risks and active-component HT MSEs agree (maximum MSE difference (3.3\times10^{-19}\)). See [raw results](../results/grpo-audit-cost-reduction-v1/run-1/raw.json) and [audit](../results/grpo-audit-cost-reduction-v1/run-1/independent_audit.json). Reproduce with:

```bash
python scripts/run_grpo_audit_cost_reduction_v1.py
python scripts/audit_grpo_audit_cost_reduction_v1.py
```

## Conditional verifier/task drift is a separate failure mode

Let (m_b(A)=\mathbb{E}_{P_b}[\chi_T(Y)\mid A]) and (m_t(A)=\mathbb{E}_{P_t}[\chi_T(Y)\mid A]). Behavior-policy importance weighting transports the action marginal, not the conditional label law. For the active component,

\[
\widehat\theta_{\rm transport}-\theta_t
=q_t(a^*)\alpha_T(a^*)[m_b(a^*)-m_t(a^*)].
\]

More generally, for (\|u^\top g\|_\infty\le B), the conditional-transport bias is bounded by

\[
2B\,\mathbb{E}_{A\sim q_t}\!\left[\operatorname{TV}\{P_b(\cdot\mid A),P_t(\cdot\mid A)\}\right].
\]

This is separate from covariate/action shift (q_b(A)\ne q_t(A)). In the frozen example, behavior parity mean is (0.5). A conditional parity shift of (-0.75) changes the target mean to (-0.25), reversing the update sign while behavior-only transport still predicts the old sign; the resulting transport bias is (0.00459\), larger than the target update magnitude (0.00153\). At a shift of (-0.5), the target component is zero. Independent target-policy clean audits or an explicit conditional-shift assumption/bound are necessary; action importance weights cannot solve this.

## Evidence status and limitations

- **Established:** exact design identities, exact two-world sign risks for the frozen fixture, matched expected rollout-plus-label cost under the declared ratios, and the conditional-transport bias identity.
- **Not established:** a better non-oracle adaptive audit policy; a general finite-sample guarantee under unknown conditional shift; real verifier prevalence; a real optimizer update; or an independently measured capability gain.
- **Novelty:** no distinct algorithmic or mathematical contribution survives the reduction. The defensible contribution is a precise negative novelty result plus a reproducible cost accounting boundary for this witness.
- **Decision:** stop this allocation-method branch. Preserve the v1 and v1-cost results. Do not add further variants of this synthetic witness. Return to the repository's separate evidence gap: independently evaluated task success after a real policy update or a genuinely actionable upstream defect.
