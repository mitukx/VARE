# Structured-edit Coding Agent Feasibility v1

**Decision: NO-GO for policy learning.** This development-only experiment established that the pinned Qwen3-4B model can run on Apple Metal and can emit/apply structured source edits, but none of 18 patches passed the independent grader. All binary rewards were zero, so neither SFT nor GRPO was run. No held-out task was opened.

## Hypothesis and frozen evaluation

The hypothesis was that a bounded structured-edit interface, automatic public-test feedback, and an independent parent-side grader would let a local coding agent solve at least one of two fresh-to-this-agent Transformers development tasks. The model was `mlx-community/Qwen3-4B-Instruct-2507-4bit`, snapshot `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`; the 2,263,022,417-byte `model.safetensors` has SHA-256 `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f`.

The task set contained only two Development tasks: tokenizer routing (`transformers-49116-tokenizer-routing-v2`) and fusion-registration idempotence (`transformers-49412-fusion-registration-idempotence-v2`). Each received three attempts under each of protocol v2, v3, and v4. Protocols and their locks are retained at [protocol directory](../experiments/coding_agent_phase_a_structured_v1/). V1 is retained as a preflight failure; its task contract omitted the candidate-visible public test, and no model rollout began under it. Every subsequent protocol change has a new version and lock.

The frozen binary reward is one only when a source edit is applied, the current public test passes, the diff remains within the target allowlist, and the parent-side hidden grader passes. `pass@3` is one if any of three attempts passes. The GO condition was at least one verified success with usable nonconstant reward on multiple development tasks; no training or held-out evaluation was authorized by these protocols.

## Calibration and execution boundary

Both task public tests fail on pinned baseline and pass on the pinned known fix in the digest-pinned ARM64 Colima image. The separate parent-side grader also rejects both baselines and accepts both fixes. The hidden expected outcomes are held by the parent process; candidate workspaces have no Git history and do not mount the grader. A source-only adversarial probe attempted grader/environment access, subprocess use, and forged success output; the parent grader failed closed with `missing_result`.

These checks establish calibration for these two narrow graders, not arbitrary-code sandbox soundness. Public tests are intentionally candidate-visible. For the fusion task, the hidden adapter does not separately test that a genuinely conflicting conversion mapping continues to be rejected; this is a semantic coverage limitation. The candidate-call adapter is present in the subject environment, while hidden expected values and the trusted result frame remain parent-side.

## Results

| Task | Attempts | Applied edits | Verified success | `pass@3` | Reward variance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tokenizer routing | 9 | 5 | 0 | 0/3 in every protocol | 0 |
| Fusion registration | 9 | 2 | 0 | 0/3 in every protocol | 0 |
| **Total** | **18** | **7** | **0** | **0** | **0** |

The initial v2 attempts selected structured tool actions, but the 512-token response ceiling truncated their calls before a complete action could be parsed. V3 increased the ceiling and produced edits, but patches were incomplete or incorrect and the agent did not complete a passing public-test/repair cycle. V4 automatically ran the public test after each successfully applied edit and returned its result in the tool response. On tokenizer routing, some attempts added target rows but omitted required override entries; repeated edits did not repair the missing entries. On fusion registration, all three v4 attempts ended without an edit. The final hidden grader result for seed 603 was `wrong_output` with an empty diff.

Across all attempts the agent generated 33,586 tokens and used 2,476.568 seconds of rollout wall time. Maximum recorded MLX peak memory was 4,216,388,718 bytes; maximum process RSS was 2,650,275,840 bytes. The runtime reported `apple_metal` on all rollouts; this was not a CPU fallback. These are aggregate rollout measurements, not isolated prefill/decode throughput or optimizer cost. No optimizer update, checkpoint reload, async learner run, or task-success held-out evaluation was performed. Host pre-run memory pressure and swap were elevated, so these results do not establish a general 4B training budget.

Raw rollouts and grader outputs are retained under [results](../results/transformers-phase-a-structured-v1/). [summary-v1.json](../results/transformers-phase-a-structured-v1/summary-v1.json) aggregates the records, and [sha256-manifest-v1.json](../results/transformers-phase-a-structured-v1/sha256-manifest-v1.json) hashes the raw bundle and frozen protocol files.

## Decision and next experiment

Phase A is **NO-GO**: both Development task `pass@3` values are zero, all 18 rewards are zero, and the sample reward variance is zero. Do not start SFT or GRPO from these rollouts. Do not access the reserved held-out pool to tune the agent. The failure points to an action-planning/interface issue as well as task-specific reasoning: structured editing removed malformed unified-diff failures, but did not yield complete semantically correct patches. The next useful step is a separately versioned Development-only protocol that materially changes the patch planning or test-driven repair strategy and uses new unconsumed tasks; it must retain independent grading and preregistered stopping criteria.

The model's exposure to historical upstream fixes cannot be ruled out from repository chronology; these results do not claim pretraining novelty. The two-task cohort is too small to support broad coding-capability claims. No task was promoted to training data, and no SFT/GRPO, held-out, or capability-improvement claim is made.

The v2 and v3 locks retain hashes for their superseded `scripts/run_local_agent.py` revisions, while this PR contains the final v4 runner source. The historical raw transcripts, task results, and lock hashes are preserved, but the exact v2/v3 runner source snapshots are not included here, so those rounds are not source-replayable from this PR alone. V4's locked implementation hashes match the included sources.

## Reproduction

Re-run the pinned preflight and calibration/rollout commands recorded in the protocol locks and raw JSON records. The model snapshot must be locally cached; no paid API or external compute was used. The runtime image digest, task source revisions, target and public-test hashes, model file hashes, seeds, generation budgets, source diffs, and grader results are included in the lock/evidence files. Reproducing the historical model outputs requires the same MLX/MLX-LM versions and Apple Metal runtime; bitwise generation is not claimed across OS/runtime changes.
