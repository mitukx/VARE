# Selected research question: distribution-correct reuse of interactive rollout groups

Decision: **CONTINUE**, for one integration-and-cost feasibility cycle. Do not
start a new model-training campaign from the current finite-law success. The
current result is not yet a major contribution under deliverable A/B/C/D.

## Question and technical hypothesis

Can an asynchronous agent-RL learner reuse **unconsumed**, complete rollout groups
after both policy and agent-visible verifier feedback change, while matching the
current joint group distribution and reducing total cost to a fixed independently
measured capability?

Hypothesis: a donor group excluded from the ancestry of both target policy and
feedback-verifier, exact residual policy coupling, and replay to the first changed
feedback observation together preserve the current group law. A bounded pool of
already-in-flight, unused groups may then cost less than discarding all stale
work. “Already generated” does not make the donor free: initial generation,
historical-policy reconstruction, target prefill, tool replay, and restoration
must all be charged. The effect is only plausible when feedback changes are
localized and large rollout tails remain reusable.

Scope: deterministic resettable coding/tool environments, fixed tokenizer and
action schema, pure feedback changes, and pinned targets during repair. General
live-web nonstationarity and arbitrary harness/schema changes are excluded.

## Why this is important, rather than chosen for the available computer

Long agent trajectories make discarding stale work expensive. Weight/version
provenance and reward rescoring alone do not specify the law of histories after
feedback changes or after a cache has influenced the learner. A system that
reduces that wasted work without corrupting group-relative updates would directly
serve asynchronous long-horizon post-training. Its relevance is conditional on
these events actually occurring at useful frequency; we have not measured their
incidence in a frontier lab or assumed any private xAI implementation.

AReaL establishes asynchronous rollout/learner decoupling as a practical problem:
https://arxiv.org/abs/2505.24298 . RollVerify already repairs stale suffixes and
reports tool-assisted experiments; general prefix repair is therefore not new:
https://arxiv.org/html/2610.09914 .

## Closest prior work and actual proposed difference

| Primary work | Overlap | Specific boundary for this question |
| --- | --- | --- |
| [SPEC-RL](https://arxiv.org/html/2509.23232) | Cached old-policy prefixes and continuation are already its central method. | We target conditional group-law fidelity when cache ancestry and in-rollout feedback change. Its published target-resampling rule is distinguished from the standard residual correction; our finite result does not refute its reported empirical gains. |
| [Chen et al., speculative sampling](https://arxiv.org/abs/2302.01318) | Exact token acceptance/residual correction. | Reuse this known primitive; no novelty claim for it. Check donor independence and tool-state/feedback boundaries around it. |
| [ReSpec](https://arxiv.org/abs/2510.26475) | RL-specific speculative serving, drafter alignment, batching and training effects. | A major systems baseline. Our target is an unused historical trace, not simply learning a fresh drafter. Net overhead may eliminate our advantage; no generic “speculative decoding for RL is new” claim. |
| [RollVerify](https://arxiv.org/html/2610.09914) | Policy-shift-guided suffix truncation/regeneration before training. | We seek a declared exact conditional group-law contract with verifier-generated observation changes; its OPS heuristic and partial-rollout throughput must be compared, not ignored. |
| [PEBBLE](https://arxiv.org/abs/2106.05091) | Relabel replay as learned reward changes. | Terminal relabeling cannot change observations already used to choose later actions. No novelty claim for reward relabeling. |
| [DARC](https://arxiv.org/abs/2006.13916) | Changing transition dynamics and correction. | Deterministic feedback can create support mismatch; resettable tools permit causal replay rather than just estimated dynamics ratios. The general off-dynamics issue is known. |
| [HarnessX v3](https://arxiv.org/html/2606.14249v3) | Replay across evolving harnesses with task-level grouping. | It explicitly replays each sample in its own harness context under a fixed verifier. That mixed-context objective is not the current-feedback product law targeted here; we do not declare its stated objective incorrect. |
| [RL Tango](https://arxiv.org/abs/2505.15034) | Generator/verifier co-training. | Co-evolution alone is not our contribution; the selected problem concerns unused asynchronous interactive groups and their sampling law. |
| [CAPF](https://arxiv.org/abs/2606.01830) | Verifier information enters the trajectory as agent-visible feedback. | Establishes a concrete intra-rollout use case. It addresses privileged-feedback credit and deployment transfer, not this cache transport contract. |
| [Adaptive data analysis](https://arxiv.org/abs/1506.02629) | Data-dependent targets and reuse require conditional reasoning. | Our counterexample is a specialization, not a new general adaptive-data theorem. |

IMPALA/off-policy correction, asynchronous PPO/GRPO, PNPO, delayed feedback,
nonstationary rewards, verifier exploitation and long-horizon credit assignment
remain important surrounding problems. This proposal does not replace their
methods or solve them all. It changes the collector's input distribution before
the existing learner; it does not assert that a clipped GRPO surrogate is an
unbiased gradient of expected task reward.

**Distinctive contribution sought, not established:** an integrated collector
that enforces a joint learner/verifier ancestry boundary and causally repairs
agent-visible feedback at complete-group granularity, with a demonstrated net
cost/capability trade-off against strong existing collectors. This would be a
systems/method contribution (C, possibly B). Known coupling proofs and tiny
finite examples alone do not satisfy the project's novelty bar. A bounded scan
found no matching complete implementation in the primary sources inspected; that
is not proof of priority or an exhaustive literature claim.

## Completed decisive prerequisites

The frozen protocol and exact rational runner check three distinct assumptions
of this **single collector question**:

1. Fixed target, q=(.9,.1), p=(.1,.9): accept-and-resample-from-p outputs (.18,.82),
   TV=.08. The pinned upstream acceptance function with simulated target
   continuation gives (.18024,.81976) in 200,000 samples. The residual control
   exactly matches p. This executes the acceptance function, not Ray/vLLM or a
   pretrained model.
2. Adaptive target chosen from a consumed two-member draft group: even corrected
   residual coupling gives conditional group TV=.1 and expected ascent update
   .05 instead of .09. Independent unconsumed donors give TV=0 and .09. This
   rejects “versioned draft probabilities alone certify on-policy groups.”
3. Identical policy, changed deterministic feedback: terminal relabeling leaves
   expected reward .2 rather than .55 and trajectory TV=.5. First-changed-feedback
   replay gives TV=0 and .55. The .5 versus 2 action-call count is not speedup.

The supplemental unchanged RVL `train_step` connection produces weighted ascent
gradients **.0499998986721** versus **.0899998176098**, matching its epsilon-aware
oracles within 2.4e-9. Its reported token-ratio/KL diagnostics are zero by the
specified metadata construction. It uses a tensor policy and analytically
assigned group probabilities, not adaptive pretrained-model training. See
`theory.md`, `trainer-protocol.json`, and `trainer-raw.json`.

The first Monte Carlo source sanity failed because the runner initialized draft
and acceptance randomness with the same seed. `attempt-1-*` and `rng-repair.json`
retain that failure and the reason for repair. After freezing separate random
streams, all six primary checks and both trainer checks passed. No tolerance,
case, or acceptance rule changed. These are same-author, same-host checks, not
external reproduction or publicly timestamped preregistration.

## Falsifiable prediction and next minimal decisive experiment

Next, implement this contract in **one** existing multi-turn collector rather
than inventing another training stack. Candidate boundary: RVL trainer consumes
complete token-exact action groups; the prototype must add per-turn policy and
feedback digests, state snapshot IDs, and donor-consumption lineage at the
collector. RVL's current scalar-generation metadata and deployment service do
not already constitute this interactive path. Do not claim integration done.

Use a pinned resettable coding environment with fresh task IDs and a capable
7B-class coder. Freeze train/dev/confirmation repository splits before opening
outputs; use a small unused format/reward pilot only to establish mixed outcomes,
never reuse retired VARE cohorts or lower a failed gate. The initial experiment
collects 32 fresh tasks x 4 trajectories under each of two verifier-feedback
versions, with three prespecified checkpoint gaps (0,1,3). Tool semantics and
independent hidden final tests stay fixed; only training-visible feedback is
changed, including a real weak-to-strong checker update and a same-output control.

At frozen checkpoints compare:
- fresh current collection;
- ordinary stale replay with policy correction and terminal relabeling;
- cached prefix acceptance/target continuation baseline;
- residual correction plus feedback repair using consumed drafts (ancestry control);
- the same repair using preassigned unconsumed donors (candidate).

Do not use success-filtered or fastest-completing donors. Wait for full groups;
log discarded/unfinished work and all calls. Pair random streams where valid,
while keeping acceptance/continuation randomness independent. A snapshot must
reproduce state and tool output before any repaired trajectory is admitted.

Prediction: target-independent donors plus causal repair pass token/sampling
conformance and produce statistically equivalent target outcomes, while the
conditional-law counterexamples disappear. This is a reliability prerequisite,
not a claim that every biased collector loses task success.

The decisive **practical** gate is at least 15% lower total collector GPU-seconds
and no >1 percentage point loss in independently checked task success, with
paired task-cluster intervals supporting both. The 32-task integration pilot
cannot establish the 1-point equivalence margin; use it to falsify savings or
correctness first. A separate, adequately powered untouched evaluation must
establish non-inferiority before any claimed efficiency win. Stop if net savings
are absent after charging all donor generation and old-policy reconstruction.

## Real learning and frontier-scale evaluation

Only after the collector gate, run matched GRPO comparisons at 7B, then 32B/70B
or a frontier-capable sparse model, with coding and one resettable long-horizon
tool/research environment. Vary independently policy lag (0,1,3,8 updates),
feedback update rate, observed disagreement rate, and 8K/32K/128K horizon. Avoid
a full Cartesian campaign: freeze zero-lag, one practical lag, and one stress
setting; add scaling only if the practical setting passes.

Primary endpoints: independently executed hidden-test task success versus total
training/collection GPU-seconds and wall-clock to a fixed success level. Include
three training seeds, fixed SFT/no-update references for capability interpretation,
strong async baseline, current-fresh baseline, SPEC-RL/RollVerify-compatible
collector baselines and the ancestry control. Report learner processed tokens,
regenerated tokens, donor creation, policy logit reconstruction, verifier cost,
restore/prefill, snapshot storage, utilization, failed jobs, and group completion
latency. Training reward, clip fraction, or tokens saved alone cannot pass.

Do not infer learning improvement from exact marginal laws: reused trajectories
can remain temporally correlated, independent-donor pools can reduce utilization,
and sample-law fidelity can be unnecessary for a useful biased method. The
criterion is the measured capability/cost frontier.

## Compute requirements (planning limits, not measured estimates)

- Completed here: CPU rational arithmetic, 200,000 categorical acceptance calls,
  eight tensor-policy train steps; 0 GPU hours and 0 paid inference calls.
- Integration pilot: two 7B bf16 inference snapshots plus KV state usually require
  separate serving slots or sequential reload/offload; historical weights and
  first-rejection logits must be reconstructable. Plan access to 2 x 80GB GPUs,
  128GB host RAM, resettable CPU sandboxes, and a **16 GPU-hour cap**. This is a
  requested future resource envelope, not an availability or runtime claim.
  Record a 10-minute throughput/profile before fixing a full-run allocation.
- A small learning comparison with 30 updates x 16 prompts x 4 trajectories x
  2048 action tokens is 3,932,160 action tokens per arm/seed before donor and
  repair overhead. Five arms x three seeds is 58,982,400 action tokens. At 7B,
  the rough dense-training 6PN arithmetic for those tokens is 2.48e18 FLOPs;
  it excludes rollout, reference/old-policy forwards, context tokens, optimizer,
  padding and tool costs. It is not a GPU-hour forecast. LoRA does not eliminate
  backbone activation or inference costs. Budget only after pilot measurement.
- Frontier design: 100 updates x 64 prompts x 8 trajectories x a 16,384 mean
  action-token planning length = 838,860,800 action tokens per arm/seed. Three
  prespecified arms x three seeds would process 7.55B action tokens before
  overhead; dense 70B 6PN is roughly 3.17e21 training FLOPs. A 64–256 GPU cluster
  is a plausible allocation envelope for profiling this campaign, not a
  demonstrated requirement or duration. Longer histories, sparse models and
  tool waits change it materially. Choose hardware/count from measured memory
  and throughput, and set a hard resource cap before launch.

A free Colab session can test a smaller inference adapter or snapshot restore;
it cannot validate this frontier claim. No paid infrastructure has been started.

## Stop conditions

1. Any conditional sampling, feedback-state restoration or complete-group
   conformance failure blocks training.
2. No auditable unused donor lineage or no resettable feedback boundary in the
   selected real collector: stop this implementation attempt; do not substitute
   another toy demonstration.
3. No >=15% net cost reduction in the practical pilot, or savings require omitting
   donor/logit/restore costs: stop the efficiency claim and campaign.
4. Non-inferiority fails, independent task success declines, or task gains exist
   only in optimization-verifier scores: do not promote.
5. A closest implementation already enforces the same contract and achieves the
   same effect: contribute a focused extension/reproduction or retire novelty.
6. The current finite-law success licenses one real integration feasibility
   cycle, not indefinite synthetic variants or new bug hunting around PR #91.
