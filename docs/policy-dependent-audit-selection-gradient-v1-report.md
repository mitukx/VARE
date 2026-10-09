# Policy-dependent audit selection changes verifier-response gradients

## Result in one sentence

For a verifier trained on policy-generated examples selected by a policy-dependent audit rule, differentiating the verifier's population optimum includes the audit-selection score. In a fully enumerable two-outcome example, omitting that term reverses the outer policy-score gradient even though every label is clean and audit inclusion has full support. The formula is the ordinary score-function derivative of a selected distribution combined with implicit differentiation; this report does **not** establish a novel algorithm.

## Question and scope

The focused question is whether a policy-dependent clean-audit selection rule changes the verifier-response term in iterative policy/reward-model co-optimization, and whether the omitted term can alter an update decision. This extends VARE's earlier work on estimating one GRPO update from selectively audited groups: here selection changes the data distribution used to fit the verifier itself.

This is deliberately not an RLVR capability experiment. It uses two outcomes, a misspecified scalar verifier, and deterministic clean labels. It isolates selection-induced covariate shift. It does not model a change in the conditional verifier-error law, real human audit behavior, text generation, an optimizer, or an LLM task.

## Closest prior art and novelty boundary

- [Foresighted Policy Optimization](https://arxiv.org/abs/2605.04266) derives a Stackelberg policy gradient containing the policy's effect on future reward-model parameters. The chain rule below is the same general object, specialized to a reward model fit on selectively audited data. This report does not show that FPO omits this term in its stated setting.
- [Cooper](https://arxiv.org/abs/2508.05613) dynamically selects positive/negative examples while updating a reward model and policy. Its existence rules out novelty claims for the broad idea of jointly adapting the policy and the reward-model data.
- [Cost-Effective Proxy Reward Model Construction with On-Policy and Active Learning](https://arxiv.org/abs/2407.02119) combines on-policy sampling and active preference selection under limited label budgets. It rules out broad novelty claims about active on-policy reward-model labeling.
- [RL Tango](https://arxiv.org/abs/2505.15034) jointly trains generator and verifier, but does not by itself establish the selective-audit derivative studied here.

The derivation is a direct combination of the likelihood-ratio identity, inverse-propensity weighting, and the implicit function theorem. That combination is mathematically useful as an interface check, but no distinct method or theorem-level novelty survives this prior-art comparison. The result is a counterexample to a particular stop-gradient approximation, not evidence that any named method uses that approximation.

## Formal claim

Let outcomes be (z\in\mathcal Z), with policy mass (p_\theta(z)>0) and audit inclusion probability (s_\theta(z)>0). The distribution of audited outcomes is

\[
q_\theta(z)=\frac{p_\theta(z)s_\theta(z)}{Z_\theta},\qquad
Z_\theta=\sum_zp_\theta(z)s_\theta(z).
\]

Let (\ell(z,y;\phi)) be verifier loss for clean label (y), and let

\[
L(\theta,\phi)=\mathbb E_{z\sim q_\theta,\,y\sim P(y\mid z)}[\ell(z,y;\phi)]+\Omega(\phi).
\]

Assume the conditional label law (P(y\mid z)) is fixed in (\theta), (L) is twice differentiable, and the Hessian (H=\nabla^2_{\phi\phi}L) is nonsingular at its minimizer (\phi^*(\theta)). Write (g=\nabla_\phi\ell), (a_\theta(z)=\nabla_\theta\log p_\theta(z)), and (b_\theta(z)=\nabla_\theta\log s_\theta(z)). Then

\[
\nabla_\theta\nabla_\phi L
=\mathbb E_{q_\theta}[\nabla_\theta g]
+\operatorname{Cov}_{q_\theta}(g, a_\theta+b_\theta),
\qquad
\frac{d\phi^*}{d\theta}
=-H^{-1}\nabla_\theta\nabla_\phi L.
\]

For (J(\theta)=\mathbb E_{z\sim p_\theta}[r_{\phi^*(\theta)}(z)]),

\[
\nabla_\theta J
=\mathbb E_{p_\theta}[r_{\phi^*}(z)a_\theta(z)]
+\left(\mathbb E_{p_\theta}[\nabla_\phi r_{\phi^*}(z)]\right)^T
\frac{d\phi^*}{d\theta}.
\]

The added covariance with (b_\theta=\nabla_\theta\log s_\theta) is the selection-propensity score. If the audit policy is fixed conditional on the outcome, (b_\theta=0): policy-induced covariate shift remains through (a_\theta), but there is no separate selection-rule derivative. If (P(y\mid z,\theta)) changes, its derivative is an additional term; this experiment holds that conditional law fixed and makes no claim about verifier-error drift.

### Exact two-outcome counterexample

Let (p_\theta(A)=\sigma(\theta)), (p_\theta(B)=1-p_\theta(A)), and (\theta_0=\log(7/3)). Audit with

\[
s_\theta(A)=\sigma(-4(\theta-\theta_0)),\qquad s_\theta(B)=1-s_\theta(A).
\]

At (\theta_0), (p(A)=7/10), (p'(A)=21/100), (s(A)=s(B)=1/2), (s'(A)=-1), and (s'(B)=1). Thus (q(A)=7/10) but

\[
q'(A)=\frac{(p's_A+ps_A')Z-(ps_A)Z'}{Z^2}=-\frac{63}{100}.
\]

Freezing the selection propensities locally instead gives (q'_{\mathrm{stop}}(A)=21/100). The verifier has one parameter, (r_\phi(A)=\phi), (r_\phi(B)=-\phi), and receives the same clean label (y=1) for both outcomes under squared loss. Its selected-data optimum is (\phi^*=2q(A)-1=2/5), so (d\phi^*/d\theta=-63/50); the stop-selection ablation gives (21/50).

For the outer score (J=(2p(A)-1)\phi^*\), the full derivative is

\[
J'=2p'\phi^*+(2p-1)\phi^{*'}=-\frac{42}{125}=-0.336,
\]

while omitting the selection score gives (J'_{\mathrm{stop}}=+42/125=+0.336). This is a sign reversal with positive audit support and no label noise. It arises from the deliberately misspecified one-parameter verifier fit to an outcome-skewed sample.

As a control, normalized inverse-propensity weighting satisfies

\[
\frac{\mathbb E_{q_\theta}[\ell(z,y;\phi)/s_\theta(z)]}
{\mathbb E_{q_\theta}[1/s_\theta(z)]}
=\mathbb E_{p_\theta,P(y\mid z)}[\ell(z,y;\phi)].
\]

So known positive propensities remove this selection distortion at the population-objective level. That is standard IPW; it does not resolve finite-budget variance or unknown propensities.

## Executable check

The frozen protocol is [`policy_dependent_audit_selection_gradient_v1.lock.json`](../protocols/policy_dependent_audit_selection_gradient_v1.lock.json). Run:

```bash
python scripts/validate_policy_dependent_audit_selection_gradient_v1.py \\
  --output results/policy-dependent-audit-selection-gradient-v1/run-1.json
```

The runner derives exact rational derivatives, compares them with a central finite-difference oracle using step (10^{-5}), and checks the normalized IPW identity. It passed all three checks:

| Quantity | Exact | Independent finite difference |
| --- | ---: | ---: |
| (q'(A)), full selection derivative | (-0.63) | (-0.629999999968156) |
| (q'(A)), stop-gradient selection | (+0.21) | (+0.210000000000488) |
| (J'), full selection derivative | (-0.336) | (-0.335999999995229) |
| (J'\), stop-gradient selection | (+0.336) | (+0.335999999993841) |
| normalized IPW (q(A)) | (0.7) | exact identity |

The raw output is [`run-1.json`](../results/policy-dependent-audit-selection-gradient-v1/run-1.json). This is a deterministic mathematical fixture, not model or trainer evidence.

## Implication for VARE and decision

The VARE/RVL path does not cache parameter-gradient vectors. At pinned RVL revision `c7e646b043cb56e5ea3c2623bb8a61e065451f72`, `HFCausalLMGRPOTrainer._metadata` reads token IDs and behavior log-probabilities, `_sample_objective` performs a fresh model forward and computes current log-probabilities before applying the behavior/current ratio, and `train_step` backpropagates that objective. VARE supplies clean-verification rewards and group-relative scalar advantages. The earlier cached-gradient witness therefore does not identify a defect in this path.

**Decision: close the cached-parameter-gradient direction as inapplicable to current VARE/RVL; do not claim a novel selective-audit co-evolution method from this derivative identity.** Reopen only for a concrete trainer that changes its audit-selection propensity with policy parameters and omits that propensity in its verifier-response update, followed by a real implementation-path reproduction and a practical variance-controlled correction. A real-model co-evolution result would additionally need a trainable verifier, an independent task grader, policy-dependent audits, and multi-seed evaluation; VARE currently has no such validated setup.

The separate externally reviewable engineering artifact remains the [pinned-RVL module-mode rollback result](rvl-grpo-module-mode-rollback-v2-report.md): a real trainer optimizer step under injected mid-step failure changed from 12/14 to 14/14 frozen rollback checks after the adapter fix, with same-host clean-clone reproduction. Independent reviewer reproduction and RVL maintainer assessment are still needed. No external PR or contact was made.
