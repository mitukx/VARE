# Independent reproduction of a finite GRPO audit witness (v2)

## Outcome

A separately authored standard-library implementation reproduced the v1 finite result on a four-member binary-reward local score update. The corrected implementation's two retained runs pass every exact check in the unchanged v2 lock. The first implementation attempt is a retained non-pass: its count-two reward branch used an incorrect algebraic simplification. The lock's decision rule is therefore preserved literally; the initial failure is not erased or called a pass.

This is same-host exact reproduction, selected after the original result was already known. It is not a confirmatory study, external reproduction, novelty result, audit-efficiency result, or evidence of model learning.

## Exact result

The reproducer independently enumerates all 16 action vectors and all 16 reward vectors for each action. For target action `0001`, using the explicit convention that the leftmost bit is coordinate 0, the Walsh coefficient at `T={0,1,2}` is

\[
\alpha_T=-\frac{3}{8}+\frac{\sqrt{3}}{4}.
\]

For `epsilon=1/2`, the constructed laws are normalized, have minimum point mass `1/32`, and have identical marginals on all subsets of order at most two. Their selected `012` marginal has total variation `1/2`. Their conditional expected-update gap is `-3/8 + sqrt(3)/4`; the action probability is `27/256`; the unconditional gap is `-81/2048 + 27*sqrt(3)/1024`. All 16 degree checks agree: the two constant actions have the zero update function and each of the other 14 actions has degree three; all degree-four coefficients vanish.

There is a scope ambiguity in the original v1 lock's phrase “each order-three marginal observation differs.” Taken literally to mean all four size-three subsets, that statement is false for this parity pair. A marginal of the constructed pair differs exactly when the observed coordinate set contains the perturbed character set `T`; among size-three subsets, only `012` contains `T`. The other triples `013`, `023`, and `123` have TV zero. This agrees with the v1 report and retained output, which identify only the selected three-label marginal as different. The reproduction therefore supports that narrower, explicitly selected witness and does not claim all third-order marginals differ.

## Protocol and implementation record

The frozen lock is [`grpo_audit_independent_reproduction_v2.lock.json`](../protocols/grpo_audit_independent_reproduction_v2.lock.json). It hashes the v1 lock, report, and both retained outputs before implementation. The new runner imports no VARE package, prior runner, or prior auditor. It uses `fractions.Fraction` to represent rational-plus-`sqrt(3)` values and fails closed on any changed input hash or exact check.

Attempt 1 failed its exact checks. The author recorded that the count-two standardized advantage was coded as `(2*y-count)/2`; the correct simplification is `(4*y-count)/2`. Its failure log and source hash are retained, but the source bytes were not, so this causal attribution cannot be independently checked from the bundle. Attempt 2's result and runner hash are retained, but its exact source bytes are also absent. Attempt 3 used the then-final source; its recorded runner hash matches the retained source for that run. A later hardening replaced Python `assert` gates with explicit errors so input hashes and exact checks still fail closed under optimized Python. The hardening leaves the mathematical checks unchanged and is covered by tampered-input and false-check tests under both ordinary and optimized interpreters. Historical attempt outputs retain their original runner hashes. See [`attempt-history.json`](../results/grpo-audit-independent-reproduction-v2/run-1/attempt-history.json), [attempt 1 log](../results/grpo-audit-independent-reproduction-v2/run-1/attempt-1-failure.log), [attempt 2 output](../results/grpo-audit-independent-reproduction-v2/run-1/attempt-2-reproduction.json), and [final-source output](../results/grpo-audit-independent-reproduction-v2/run-2/reproduction.json).

Reproduce to a fresh path from the repository root:

```bash
python scripts/reproduce_grpo_audit_independent_reproduction_v2.py \
  --output /tmp/vare-grpo-audit-independent-reproduction-v2.json
```

At the reproduction revision, all 12 exact checks passed; Ruff check and format check passed; Python compilation and `git diff --check` passed. The current source also reproduces all 12 checks, and its fail-closed behavior is mutation-tested under ordinary and optimized Python. The original reproduction used Python 3.12.12 on macOS arm64 and the standard library only. No model, optimizer, GPU, network, or paid service was used.

## Limits

This verifies one exact finite Boolean-function result and the constructed parity pair. It does not measure sample complexity, audit cost, real verifier noise, a learned policy, held-out task success, or capability. Related-work novelty remains unresolved. The user’s overall research goal—an inexpensive post-training/RL method with independently measured real-model improvement—remains open.
