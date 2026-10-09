# Replay group prompt identity v1

## Question

Can grouped replay admit or sample members with one declared GRPO group ID even when they were generated for different prompts?

## Prior state and scope

The earlier [rollout-group report](rollout-group-integrity-v1-report.md) explicitly left task-prompt equivalence to callers. A frozen CPU regression now tests that boundary. This closes a documented limitation; it is not a novelty claim or a model-learning result.

The contract is intentionally narrow: one comparison group must have a single exact prompt string. Distinct task IDs with the same prompt remain valid. This protocol does not validate policy ID/version, generation parameters, task metadata, verifier correctness, or the full trainer's interpretation of a batch.

## Frozen protocol

The [protocol lock](../protocols/replay_group_prompt_identity_v1.lock.json) and [regression test](../tests/test_replay_group_prompt_identity.py) were written before the baseline run. The fixture covers:

1. Atomic admission rejects a mixed-prompt group and leaves a valid existing group intact.
2. Legacy per-item insertion cannot make either grouped sampler return a mixed-prompt group.
3. A same-prompt group with distinct task IDs remains valid.

## Result

At baseline revision `5192cbaf5aa1ac02e010933b0ac1ba0f553b69e1`, the frozen test produced two failures and one pass: atomic admission accepted mixed prompts, and legacy grouped sampling returned all four mixed-prompt experiences. After the fix, all three tests passed. The adjacent group-integrity, freshness, and RVL-hook suite passed 20 tests.

The fix makes `_group_is_complete` reject a group unless every member has the same string prompt. Atomic `add_group` checks this before admission. The same shared completeness predicate makes `sample_grouped` and grouped `sample_current` fail closed on legacy mixed-prompt groups.

Raw baseline/fixed logs, result summary, and SHA-256 manifest are retained in [`results/replay-group-prompt-identity-v1/`](../results/replay-group-prompt-identity-v1/). Runtime: Python 3.12.12, pytest 8.4.2, CPU.

## Claim boundary and decision

This is one deterministic replay contract, not evidence that a GRPO trainer is correct, a policy update improves an outcome, or a model capability increased. No external reproduction was performed. Keep it as supporting post-training integrity evidence; the central missing result remains independently measured task success after a real policy update. Do not expand this fix into a general task/provenance validator without a separate counterexample and protocol.
