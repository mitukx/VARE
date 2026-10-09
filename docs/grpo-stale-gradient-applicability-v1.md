# Stale parameter-gradient applicability audit

**Decision: retire this as a VARE/RVL post-training failure direction.** The zero-policy-KL counterexample is mathematically valid, including for a nontrivial neural parameterization, but the inspected RVL and TRL GRPO paths do not apply parameter gradients computed at earlier policy versions. They retain rollout data and behavior-policy log-probabilities, then form the current-policy objective and gradient at training time. The remaining counterexample is a familiar parameter-symmetry/coordinate-alignment issue, not evidence of a consequential RL trainer failure.

## Question and distinction

The claim under review is narrow: can a trainer apply a stored parameter-space gradient vector computed under policy version `v` to parameters for policy version `v+1`, without recomputing it? This differs from retaining an old trajectory, reward, advantage, or behavior log-probability. PPO/GRPO routinely retain those rollout quantities and recompute the current policy's log-probability and gradient; an importance ratio corrects part of the policy-distribution difference. That is not cached-gradient reuse.

## Source-path audit

The audit used VARE at `7f9d27c7862b8aba26c0f26c53579ec8e506becd`, RVL at `c7e646b043cb56e5ea3c2623bb8a61e065451f72`, and the TRL source snapshot at `f4526e10e25c8618855932528c232e2284c7801c` (2026-10-09).

### RVL and VARE adapter

- RVL reads stored prompt/response token IDs and behavior token log-probabilities in [`hf_trainer.py`](https://github.com/mitukx/Recursive-Verification-Lag/blob/c7e646b043cb56e5ea3c2623bb8a61e065451f72/src/rvl_systems/hf_trainer.py#L50-L67). It forwards the stored sequence through the current model, computes current token log-probabilities, and forms the clipped GRPO surrogate against the stored behavior log-probabilities at lines 69–118. This is fresh current-policy differentiation over retained rollout data.
- `train_step` zeros gradients, calls that objective for each sample, performs `loss.backward()`, clips, then steps the optimizer at lines 149–196. No parameter-gradient vector crosses a policy-version boundary.
- VARE's [`rvl_grpo.py`](../src/vare/integrations/rvl_grpo.py) turns retained `Experience` records into verified generations, computes scalar group-relative advantages (lines 198–260), and passes them to `trainer.train_step` (lines 262–287). It does not cache gradients.
- RVL's separate tabular lab learner uses a local `gradients` accumulator, but its source comment says it computes all gradients against one fixed target policy before mutation. It evaluates probabilities and per-row score derivatives using the current logits, accumulates them, then applies the update in the same `train` call (`src/rvl_systems/lab/agent.py`, lines 41–64). This is batch accumulation, not a stale gradient consumer.

### TRL GRPO upstream path

- The inspected [`GRPOTrainer._compute_loss`](https://github.com/huggingface/trl/blob/f4526e10e25c8618855932528c232e2284c7801c/trl/trainer/grpo_trainer.py#L2946-L3019) reconstructs inputs from retained prompt/completion token IDs and calls `_get_per_token_logps_and_entropies(model, ...)` on the current model (lines 2946–2973). It then uses stored `old_per_token_logps` only as the denominator/behavior reference for the importance ratio (lines 2986–3019); the policy loss uses the newly computed differentiable `per_token_logps` (lines 3033 onward).
- Its rollout path explicitly says that misaligned generation/optimization may make samples come from an earlier policy version, so it computes `old_per_token_logps` for importance sampling. It also computes current-model log-probabilities under `no_grad` for the reference statistics (lines 2560–2585). The retained object is old log-probability data, not a gradient.
- In the inspected paths, gradients are created from the current loss by autograd during training. No evidence was found of serializing, queueing, or applying old parameter-gradient vectors. This audit is limited to pinned RVL and TRL paths; it does not establish that no such code exists anywhere in the ecosystem.

## Neural parameter-symmetry check

To answer whether the original effect survives beyond the sine/two-state parameterization, this report includes a direct CPU autograd calculation, not a trainer. Define a Bernoulli policy with

\[
f_\theta=a_1\sigma(w_1)+a_2\sigma(w_2),\qquad
\pi_\theta(1)=\sigma(f_\theta),
\]

where each `(a_i, w_i)` is one hidden unit. At `\theta=(10,0,-10,0)`, `f=0` and `\pi(1)=1/2`. Swapping the two complete hidden units gives `T\theta=(-10,0,10,0)`, exactly the same function and action distribution, hence zero policy KL. For `J(\theta)=\pi_\theta(1)`, autograd gives

\[
g(\theta)=(1/8,5/8,1/8,-5/8),\quad
g(T\theta)=(1/8,-5/8,1/8,5/8).
\]

Thus `cos(g(\theta),g(T\theta))=-12/13\approx-0.9231`, and the directional derivative from applying the untransformed old ascent vector at the swapped representation is `g(T\theta)^T g(\theta)=-3/4`. Fresh ascent has derivative `\|g(T\theta)\|^2=13/16`. At step size `0.001`, the action probability changes by `-0.0007499994` under the old vector and `+0.0008124993` under the fresh vector. The executable is [`reproduce_stale_gradient_neural_symmetry_v1.py`](../scripts/reproduce_stale_gradient_neural_symmetry_v1.py); its captured output is [`run-1.json`](../results/stale-gradient-neural-symmetry-v1/run-1.json).

This establishes that output-space equality alone cannot identify a parameter-space vector across a changed parameter alignment. But the transformation here is an exact hidden-unit permutation, with Jacobian a coordinate permutation. The counterexample is precisely a missing gradient-coordinate transport. Neural-network permutation symmetries and the dependence of Euclidean optimization on parameter-space geometry are established topics; stale-gradient reuse itself is also a long-studied asynchronous-SGD problem. This report does not identify a new mechanism, bound, or correction.

## Interpretation and limits

The original zero-KL result remains a valid warning against using policy KL alone to certify reuse of an already computed gradient vector. It does **not** imply that stored trajectories, rewards, advantages, or old log-probabilities are invalid. Those are distinct objects; the inspected trainers recompute current-policy derivatives on them, with their own standard off-policy approximations and limitations.

No real trainer path consuming a stale parameter gradient was found. Therefore this audit does not justify a stale-vs-fresh-vs-importance-weighted training comparison: such a comparison would need an actual consumer, and inventing a trainer would add no evidence about existing post-training systems. No task-success effect, incidence rate, RLVR-specific novelty, or model-capability effect is established. No code fix or upstream patch is warranted for this direction. The previously completed RVL module-mode rollback finding remains separate and unchanged.

## Reproduction

From the repository root, with PyTorch available:

```sh
python scripts/reproduce_stale_gradient_neural_symmetry_v1.py
```

The script is deterministic, uses CPU float64, no model download, no GPU, and no paid service. It is a direct derivative sanity check; it is not a frozen capability experiment.

## Prior-art comparison

- Asynchronous SGD explicitly studies gradients computed on an older parameter snapshot and later applied to a newer one; staleness-aware methods and convergence analysis predate this VARE example ([Faster Asynchronous SGD](https://arxiv.org/abs/1601.04033)).
- Neural parameter spaces have exact hidden-unit permutation symmetries ([Hidden Symmetries of ReLU Networks, ICML 2023](https://proceedings.mlr.press/v202/grigsby23a.html)); Euclidean gradient dynamics also depend on how parameter coordinates and their metric are represented ([The Geometry of Neural Nets’ Parameter Spaces, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/file/395371f778ebd4854b88521100af30ad-Paper-Conference.pdf)).
- GRPO implementations distinguish rollout behavior log-probabilities from the current model's differentiable log-probabilities and compute current-policy losses from fresh forward passes, as the pinned TRL source above shows.

**Novelty assessment:** no defensible original research contribution survives. The VARE witness is a compact diagnostic, not a new stale-gradient theory or evidence of a failure in deployed training code.

## Decision

**STOP / RETIRE** the stale-parameter-gradient direction for VARE and the inspected RVL/TRL implementations. Keep the original mathematical report and this applicability audit as a negative result. Reopen only if source-level evidence identifies a maintained post-training path that applies parameter gradients from an older policy version without recomputation or explicit coordinate transport. Preserve the independent RVL module-mode rollback evidence as a separate systems result; this investigation neither changes nor extends it.
