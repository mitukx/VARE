# PR #91: same-epoch prepare review

Review base: `436a96e86b704ad981b05237b41fdfb846f18817`.
Fixed/frozen review head: `ba94a9e513fdaf882cde091766007a88b9edaeee`.
PR: https://github.com/mitukx/Recursive-Verification-Lag/pull/91

## Contract findings

The original docs describe monotonically increasing immutable artifacts, but
neither declare same-epoch prepare serialization nor bind an activation request
to an artifact digest. Publication prevents duplicate version files within a
single root; separate roots can publish different bytes under one version.
Workers permit both candidates before activation. Once a version is active,
conflicting manifest identity is rejected. An external global version uniqueness
service was neither implemented nor documented; assuming one cannot close the
observed API failure.

The original coordinator checks prepare version/epoch, not ACK digest. Activation
specifies only version/epoch. Both its ACK check and post-activation health check
can therefore accept the wrong artifact. Version/epoch equality is insufficient.

## Reproduced defect, rather than a source-only concern

The locked schedule blocks A's loader, prepares B under the **same epoch**, records
B's ACK, then releases A before any activation. A overwrites prepared B. On the
review base the coordinator declares success for B, yet active/durable SHA and
reward belong to A: reward **0.1 rather than intended 0.9**. The conflicting
artifacts share version 1. Attempting to repair this already-activated wrong
version by redeploying B is rejected by the existing active-identity invariant.
The identical-artifact overlap is a passing control. See `baseline.json`.

The initial reproducer attempted recovery without catching its exception and
produced no JSON. It was repaired to retain that baseline recovery failure as a
field before the recorded baseline/fixed comparisons. No schedule, decision rule,
or expected artifact changed.

## Minimal contract correction

- Serialize public coordinator prepare/activate operations and whole deploy calls
  within one instance/event loop. Separate prepare/activate is explicitly a
  latest-pending-candidate interface, not a caller-owned lease.
- Require SHA-256 in activation requests. Check prepared or already-active digest
  before success, and include digest in activation ACKs.
- Check prepare ACK digest, activation ACK epoch/digest, and active health digest.
- Preserve the earlier post-loader epoch/active-manifest fence.

A raw management caller can still overwrite prepared state. The corrected
coordinator then rejects activation, leaves the bootstrap verifier active, and
explicit redeploy of B succeeds. The contract is fail-closed identity, not
progress under arbitrary competing raw writers. The identical-artifact control
still activates successfully. See `patched.json`.

## Verification and freeze

The final local suite ran **398 tests, 4 skipped, no failures** on Python 3.12.12.
It covers all previous fencing schedules plus conflicting/identical same-epoch
prepares, coordinator serialization, incorrect ACK/health identities, and
missing/wrong activation digests including idempotent activation. Earlier focused
26-test output predates addition of the final missing-digest/idempotent control;
the final full suite includes it. The deployment CI now explicitly runs both
fencing modules. Four GitHub workflows succeeded at the fixed head (recorded in
`github-ci.json`). The Linux full-suite job also ran 398 tests, with 29 optional
backend skips; this is not 398 executed backend tests. Its log is retained. PR is Ready for review; **not merged**.

This concludes the requested limited review. No further neighboring-bug search
is part of this cycle.

## Remaining limits

The required digest changes the management client protocol; old version-only
activation clients fail closed and must upgrade. This is not fleet-wide atomic
activation or rollback of workers already activated. The lock is not a
cross-process lock; epoch allocation/persistence retain their prior scope.
Health is a point-in-time observation, not protection against subsequent
independent higher-epoch deployments. Semantic verifier correctness, checksum
collision security beyond SHA-256 assumptions, operational incidence, and
pretrained-model learning impact were not measured. Previous controlled trainer
results remain scoped to their original protocol; this review did not rerun or
reinterpret them as a capability result.
