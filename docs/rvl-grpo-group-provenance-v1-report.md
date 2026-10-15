# Direct RVL GRPO group provenance boundary v1

**Status:** fixed and locally verified on 2026-10-15. This is bounded E0
trainer-boundary evidence; it does not establish model capability or defect
prevalence.

## Finding

At baseline `7801f2415c5193c03a48b3a1a0fa28d317ec51d1`,
`RVLGRPOHooks.train_candidate` accepted a complete declared rollout group by
ID and member count alone. The adapter normalized its rewards and called the
trainer even when the group combined different prompt identities, policy or
behavior distributions, or evaluator identities. The ordinary
`CapabilityLoop` replay path already rejects mixed prompt and policy/verifier
provenance; this gap was limited to direct adapter training calls.

The regression uses a CPU fixture trainer and a two-member group. Ten controlled
mutations cover prompt ID/text, policy ID/version, model, sampling temperature,
log-probability distribution, verifier name/version, and verifier-configuration
hash. On the pinned baseline all ten cases reached `train_step` instead of
raising. The fixture is a counterexample, not an estimate of how often callers
produce such batches.

## Correction and checks

Before computing group-relative advantages, the adapter now checks that each
member's stored prompt matches its task and that all members of a declared group
agree on prompt, policy ID/version, behavior model/temperature/log-probability
distribution, verifier name/version, and any recorded verifier-configuration
hash. This detects within-group mixtures; it does not turn absent provenance
fields into independently verified values. Mixed groups fail before the
trainer call. Separate complete groups may still contain different prompts and
provenance.

The same ten regressions now reject the batch with no fixture trainer step or
state change. Existing homogeneous-group advantage checks, separate-group
normalization, reward-attestation/freshness cases, trajectory checks, and the
full CPU suite passed. The full local suite excluded the two optional Math500
grader modules; it passed with Python 3.12 on `PATH`. Ruff and the targeted mypy
check also passed.

This closes one directly reproduced integration-boundary correctness defect.
It does not demonstrate a failure in normal replay selection, a real optimizer
or model update, downstream task improvement, real-world frequency, or an RL
method contribution.
