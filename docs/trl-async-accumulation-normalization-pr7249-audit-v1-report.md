# TRL Async accumulation normalization PR #7249 audit v1

**Status: pinned-source reproduction pass for the PR's token-mean contract; no new VARE implementation or model-quality claim.**

## Question

For the token-mean objective asserted in [TRL issue #7206](https://github.com/huggingface/trl/issues/7206), does [open PR #7249](https://github.com/huggingface/trl/pull/7249) make the real `AsyncGRPOTrainer` and `AsyncDistillationTrainer` accumulation path invariant to unequal completion-token counts across microbatches, compared with the pinned base revision?

The estimand is the sum of active token losses over all microbatches in one optimizer window divided by the total active completion-token count across that window. Distributed gradients are averaged across ranks. This audit tests that stated token-mean contract; it does not establish that every asynchronous training use must choose that objective.

## Method

Before execution, protocol `trl_async_accumulation_normalization_pr7249_v1` was frozen in VARE commit `bee2242`. It pins TRL base `a98fa6a4428f9aae58dfb26d729d7437f662f27a`, open PR head `5234eb7c70f4ca7eb92fe8e01611a33eaff17f40`, the production trainer-file hashes, and the exact focused test hash.

The unmodified CPU test file added by PR #7249 was run at both revisions. It constructs deterministic token-local model instances and exercises the actual Hugging Face `Trainer` training loop, including a one-step optimizer comparison. It covers unequal/equal/zero completion counts, two and four accumulation slots, a short final window, masked positions, simulated two-rank averaging, and a constant auxiliary-loss control. No pretrained weights, GPU, asynchronous worker, external compute, or paid API were used.

Environment: Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, Accelerate 1.12.0, pytest 8.4.2.

## Results

| Arm | Focused test result | Primary 100/900-token comparison |
|---|---:|---:|
| Base revision | 17 failed, 8 passed, 1 skipped | Full-batch token mean `0.65`; accumulated microbatch means `1.25` |
| PR #7249 head | 25 passed, 1 skipped | The production Trainer test matches the full-batch loss and gradient |

The mismatch is directly predicted by the frozen example: `(100×2 + 900×0.5)/1000 = 0.65`, while averaging the two microbatch means gives `(2+0.5)/2 = 1.25`. For the test's AsyncDistillation arm, the base returned `0.19409984` against a full-batch reference of `0.05492845`; the PR test passes the candidate against that reference. The same test-file bytes were used for both revisions. The base failures include the predeclared uneven-token normalization mismatch; equal-token controls pass.

The complete losslessly compressed candidate and baseline logs, wrapper note, commands, hashes, and summary are retained in [run-1](../results/trl-async-accumulation-normalization-pr7249-v1/run-1/). A read-only internal audit confirmed the source/test identities and arithmetic. This is independent local execution of a PR-authored test, not an independently authored test suite or an outside human reproduction.

## Scope and limitations

- This confirms a source-level accumulation defect under the issue/PR's stated token-mean contract and reproduces the proposed fix. It does not establish the training frequency or effect on a pretrained model or downstream task success.
- The model is a small deterministic test model; the cross-rank case simulates gradient averaging in one process, not distributed hardware.
- The focused test originates in PR #7249. Its passing candidate result is therefore evidence that the PR's own regression passes in this environment, not novelty.
- The auxiliary-loss case uses a constant scalar. The implementation intentionally retains equal-microbatch auxiliary-loss scaling; variable MoE auxiliary-loss semantics were not tested and are not adjudicated as a defect here.
- PR #7249 is already open upstream. VARE did not make a competing patch or contact the author/maintainers.

## Decision

**Accept the narrow reproduction; do not duplicate the open PR.** This is stronger than a toy formula alone because the frozen upstream production trainer and actual Trainer loop were exercised at base and candidate revisions. It remains an implementation-correctness result, not evidence of improved model capability. The next major portfolio gap remains a no-cost real-model update with independently measured task success; current evidence does not yet provide a fresh pairing that has cleared its base-feasibility gate.

## Reproduction

The immutable source revisions and exact test command are recorded in [the protocol lock](../protocols/trl_async_accumulation_normalization_pr7249_v1.lock.json) and [the command record](../results/trl-async-accumulation-normalization-pr7249-v1/run-1/commands.txt). The PR test SHA-256 is checked against the lock before interpreting the output.
