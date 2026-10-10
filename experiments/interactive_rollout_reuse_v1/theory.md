# Conditional exactness of interactive rollout repair

These are explicit correctness prerequisites and finite counterexamples, not a
claim of a new general speculative-sampling theorem.

## 1. Rejection must use the residual law

For a fixed target categorical law p and an independent proposal X~q, let
m(z)=min(q(z),p(z)) and d=1-sum_z m(z)=TV(p,q). Accept X with
min(1,p(X)/q(X)). If rejected and resampled from p, the output law is

    s(z) = m(z) + d p(z).

Thus acceptance alone does not give p. The standard correction samples the
rejected token from (p-q)+/d, giving m(z)+(p(z)-q(z))+ = p(z).
This correction is established speculative sampling (Chen et al., 2023):
https://arxiv.org/abs/2302.01318 . It is not our contribution.

With q=(.9,.1) and p=(.1,.9), target resampling yields (.18,.82), TV .08.
A two-member binary-reward group with population-std normalization has mean
logit-score ascent update .1476 rather than .09. The epsilon-free objective is
specified in the protocol; the supplemental actual-trainer check uses RVL's
1e-6 variance epsilon.

## 2. A correct residual kernel does not resolve adaptive cache dependence

Let H be history before a donor group D is generated under behavior policy q
and old feedback F. Let the learner/verifier target Z=(theta,phi) later depend on
other data. Ordinary transport exactness K_Z(q)=p_Z does not imply
K_Z(Law(D|H,Z))=p_Z when Z depends on D.

Concrete two-action, group-size-two example: q(action1)=.5. Select current
p(action1)=.9 if D_1=1, else .1. Run the **correct residual kernel** independently
on each cached group member, using logged q. Conditional on the selected target,
the first member is known and always accepted; only the second has law p.
Conditional group TV from p x p is .1. The expected group-normalized ascent
score is .05; fresh p x p gives .09, for either target. Averaging across targets
does not remove this gradient difference. This is an adaptive target-selection
rule, not a measured neural-network SGD trajectory.

An independent unconsumed donor group D'~q x q, conditionally independent of Z
given H, has the correct p x p output under the residual kernel. “Unconsumed”
must exclude influence on **both policy and feedback-verifier** updates,
hyperparameter selection, curriculum, and target-selection decisions. Lineage
exclusion is a practical sufficient condition under independent random streams;
it is not a proof that every consumed draft is biased or that every biased
sampler harms task performance. These facts specialize conditional sampling and
adaptive-data principles, rather than establishing a new general lower bound.
Related primary foundation: https://arxiv.org/abs/1506.02629 .

## 3. Changing agent-visible feedback changes the trajectory law

Write an interactive law as

    P_(theta,phi)(tau) = mu(x) product_t pi_theta(a_t|h_t) F_phi(o_t|h_t,a_t).

Policy ratios alone cancel the F factors only when the feedback law is unchanged
(or a different estimand explicitly conditions on historical contexts). Exact
terminal reward relabeling changes R(tau), not the observations that caused later
actions. Deterministic feedback changes can put positive target mass outside old
trajectory support. This is an off-dynamics issue, not a new observation about
ordinary reward-model rescoring. DARC already studies changing dynamics:
https://arxiv.org/abs/2006.13916 .

In the locked example a_1~Bernoulli(.5), old feedback=0, new feedback=a_1;
P(a_2=1|feedback0)=.2 and P(a_2=1|feedback1)=.9. Reward is a_2.
Relabel-only has expected reward .2; current collection has .55, with trajectory
TV .5 despite identical policy weights. Replaying the new feedback and
regenerating the suffix at its first change exactly produces the current law.
Expected regenerated action calls are .5 vs two for fresh collection. This is
only an action-count calculation, excluding donor generation, verification,
prefill, state restoration, and batching. It is not a measured speedup.

## 4. Sufficient contract for the selected method

Assume: (a) fixed initial tasks and resettable deterministic tool state; verifier
changes only agent-visible observations, not unrecorded state; (b) old donor
trajectories are iid under pinned q,F_old, conditionally independent of the
frozen target theta,phi given pre-donor history; (c) no reward/length-based donor
selection, complete groups and separate auxiliary RNGs; (d) exact old/current
sampling distributions including temperature/truncation; (e) exact snapshots,
no irreversible side effects; (f) target remains pinned throughout repair.

Walk the old trace causally. At each policy token apply the standard residual
coupling. On first rejection, sample its residual token and regenerate the
remaining trajectory under the target. After each accepted tool action, replay
the tool/verifier under the target. If its observation differs, discard the old
suffix and continue under the target from that state. Otherwise retain the
prefix and continue validation. Re-evaluate terminal reward under the target.

Proof sketch: at a common history the residual mixture emits exactly the target
policy token; the tool emits the target observation. After a divergence all
later variables are freshly target-generated. Induction over finite histories
gives the target trajectory marginal. Independent donor members and auxiliary
randomness give the target product group law. Therefore any deterministic
complete-group advantage function has the same distribution as fresh collection.
This says nothing about optimality of GRPO or exact gradients of expected reward.

This contract combines known coupling and causal replay principles. It is **not
yet an implemented scalable algorithm or a demonstrated novel research result**.
The potentially substantial contribution is enforcing its lineage/feedback
conditions in an asynchronous agent learner while showing a net cost advantage.

## Actual RVL path check

`trainer_check.py` runs unchanged RVL `HFCausalLMGRPOTrainer.train_step` on all
four binary groups for each selected target, then weights gradients by the exact
laws. It instantiates a one-parameter tensor policy, not a pretrained LM.
The stored token probabilities equal current-policy scores by construction;
these are **not** the true conditional sampling probabilities of the transported
consumed group. Reported ratio/KL near zero therefore cannot certify the data law.
The numerical gradients match .05/sqrt(1+4e-6) and .09/sqrt(1+4e-6).
