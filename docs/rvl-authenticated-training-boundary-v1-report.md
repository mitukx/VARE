# Authenticated reward checks at the RVL trainer boundary

Date: 2026-10-15  
Evidence tier: E0 code-path regression. No model inference or real optimizer
update was run; the tests use a recording stub trainer.

## Finding

`CapabilityLoop` already supports use-time reward authorization, including the
operator-owned `RewardEvidenceManifest.verified` check. `RVLGRPOHooks` also
exposes `train_candidate()` directly, though, and that method previously
converted mutable `Experience` values into RVL training samples without
checking the manifest. A caller using the hook directly could therefore skip
the outer-loop admission boundary.

## Change

`RVLGRPOHooks` now accepts an optional
`training_evidence_authorizer: Callable[[Experience], bool]`. When provided,
the hook checks every experience immediately before conversion and before the
trainer call. An exception, a non-boolean result, or any rejection stops the
whole batch. Setting `require_authenticated_training=True` also makes
construction fail unless an authorizer is configured. This strict option
works for direct hook calls; `CapabilityLoop` keeps its independent
consumption-time check. Applications that can reach both paths should
configure both with the same operator-owned manifest.

At both boundaries, an authenticated replay-backed RVL batch also has to
contain each declared group member index exactly once. Per-member receipts
alone would authorize sending the same signed member twice to fill a group;
the batch check rejects duplicates, missing members, inconsistent IDs/sizes,
and mixed replay/non-replay rows before authorization or training. Completeness
uses the count of distinct, in-range indices, so an untrusted huge size
declaration cannot trigger a proportional allocation. Live RVL
rollouts carry `rvl_generation` but have no replay member indices yet, so their
own configured authorizer decides whether they are eligible.

The option defaults off for compatibility with diagnostic and legacy runs.
Without it, the hook does not claim that RVL's mutable `trusted` field or
unkeyed SQLite behavior hash authenticates a reward.

## Verification

The targeted CPU suite passed 51 tests:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3.12 -m pytest -q \
  tests/test_rvl_reward_attestation.py \
  tests/test_rvl_trajectory.py \
  tests/test_training_evidence_admission.py
51 passed
```

A separate fresh-interpreter import smoke for `vare.engine` followed by
`vare.integrations.rvl_grpo` passed; the reward-evidence helper uses a lazy RVL
reader import to avoid a package-initialization cycle.

New direct-hook tests cover valid signed evidence, a missing strict-mode
authorizer, empty batches, duplicate or missing group members, and
post-signature edits to reward, verifier configuration, policy version, and
generation token IDs. A huge group-size declaration is rejected without
allocating a correspondingly large range. They use a recording test trainer
to assert that invalid evidence is rejected before `train_step`; the success
case confirms the authorized batch reaches the hook. An engine regression exercises a repeated
signed member through replay sampling and confirms the outer-loop check rejects
it before training. A second engine control confirms authenticated live RVL
rollouts can proceed without replay-only member indices. Existing engine tests
continue to cover fail-closed per-experience authorization.

## Limits

This establishes admission behavior in the VARE Python process, not the
truthfulness of a grader or isolation against malicious code with access to
that process. The signer must remain operator-controlled, and the candidate
agent must not receive its key. The direct-`put` then `rewrite_verified()` hash
mismatch remains fail-closed pending a versioned RVL replay-hash contract.
Read-time snapshots and the SQLite behavior digest are diagnostics, not
signatures. This change does not pass the retired model/task gate or authorize
training on a new cohort.
