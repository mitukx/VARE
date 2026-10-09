# Stale audit-gradient control under policy overlap

## Question

Can policy-distribution overlap alone justify reusing an audited policy-gradient correction after the policy has changed?

**No.** A one-parameter counterexample has exactly identical old and new action distributions (zero KL and total variation), yet the old audited correction increases the current policy's hack probability. The result concerns reuse of a parameter-space gradient vector. It does not apply when an implementation retains audited examples and recomputes their current-policy score gradients.

## Relation to prior work

RL Tango and Cooper study training the generator and verifier together. Foresighted Policy Optimization (FPO) studies the policy's effect on reward-model retraining and adds a steering term. Projected Audit Correction (PAC) studies a fixed verifier and uses the current audited hack-probability gradient; its theorem assumes the audited gradient remains representative over the training interval. VARE's previous policy-shift studies concern action-distribution overlap and estimators of target-policy updates.

The result below isolates a different object: **the parameter gradient itself**. It does not establish that any cited method caches stale gradients, and it is not a new policy-training algorithm. It is a limitation on using KL or action overlap as the sole freshness test for a cached correction direction. Its novelty is modest: it is a direct derivative counterexample, not a paper-level contribution by itself.

## Exact counterexample

There are two outcomes: a correct answer $G$ and a verifier-accepted hack $H$. The verifier accepts both, so its reward is constant and its gradient is zero. Let the policy be parameterized by

$$
\pi_\theta(H)=\frac12+a\sin(\theta/\tau),\qquad
\pi_\theta(G)=1-\pi_\theta(H),
$$

where $0<a<1/2$ and $\tau>0$. The clean task-success rate is $J_C(\theta)=\pi_\theta(G)$ and the hack rate is $p_H(\theta)=\pi_\theta(H)$.

At $\theta_0=0$ and $\theta_1=\pi\tau$, both policies assign probability $1/2$ to each outcome. Therefore

$$D_{\mathrm{KL}}(\pi_{\theta_0}\|\pi_{\theta_1})=0,\qquad
\mathrm{TV}(\pi_{\theta_0},\pi_{\theta_1})=0.$$

Their hack gradients have opposite signs:

$$p'_H(\theta_0)=a/\tau,\qquad p'_H(\theta_1)=-a/\tau.$$

Suppose exact clean audits at $\theta_0$ produce a correction vector and that this parameter-space vector is reused at $\theta_1$. Gradient descent on audited hack probability gives $u_0=-\lambda p'_H(\theta_0)=-\lambda a/\tau$. For a finite parameter step $\eta>0$, write $\delta=\eta\lambda a/\tau^2$. Then

$$
p_H(\theta_0+\eta u_0)=\tfrac12-a\sin\delta,
\qquad
p_H(\theta_1+\eta u_0)=\tfrac12+a\sin\delta.
$$

For $0<\delta<\pi$, the old vector lowers the hack rate at its source policy and raises it at the new policy, despite zero policy KL. Because verifier reward is constant, PAC's projection convention reduces to the identity here; the reversal comes only from the stale parameter gradient. All labels and verifier behavior are fixed, so this is not conditional verifier-error drift.

## What follows, and what does not

- KL/TV overlap of policy outputs does not bound the alignment of cached parameter gradients without additional assumptions on score/Jacobian smoothness and parameterization.
- This does not imply that stored audited trajectories are unusable. Recomputing current log-probability scores on those trajectories changes the estimator and avoids this particular stale-vector failure, subject to support and importance-weighting limits.
- A useful delayed-control guarantee would need a bound on current-vs-cached audit-gradient error (or a fresh gradient estimate), not only an action-level KL threshold. Deriving a non-vacuous, finite-sample threshold for actual GRPO/optimizer updates remains open.
- The witness is mathematical and deliberately minimal. It gives no evidence about the frequency of this failure in LLM trainers, model quality, or task success.

## Decision

Retain this as a precise diagnostic for the next scientific gate: **measure current-to-stale audited-gradient alignment under policy updates, while separately testing whether gradients are recomputed from retained trajectories**. Do not claim a new mitigation or capability gain from this counterexample. A publishable contribution requires a nontrivial robustness bound or a measured failure on an actual training path, compared with fresh-gradient and importance-weighted baselines.

## Closest sources

- [RL Tango](https://arxiv.org/abs/2505.15034)
- [Cooper](https://arxiv.org/abs/2508.05613)
- [Explaining and Preventing Alignment Collapse in Iterative RLHF](https://arxiv.org/abs/2605.04266)
- [Verifier Errors in RLVR: Reward Hacking, Limits of Feedback, and Selective Control](https://www.elliott-thornley.com/src/writing/verifier-errors-rlvr/)
