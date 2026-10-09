# RVL GRPO group-boundary real-trainer audit v1 — gate failure

## Question and scope

Does VARE's explicit per-rollout-group advantage override change a real pinned RVL GRPO optimizer update when two VARE groups share one prompt ID, while the one-group control exactly matches RVL's native prompt grouping?

This frozen CPU-only run was an integration audit with synthetic labels. It was not an efficiency study, policy-quality evaluation, or capability test. The protocol and source hashes were committed before model execution (`9f59354e027768822addaba9077f71045ca422d4`).

## Outcome

**Gate failure before any trainer or optimizer step.** The frozen prompt asked the model for one letter. The eight responses were token sequences `A, E, E, D, U, E, D, A`; the positive-reward row was `E`, and a zero-reward row was also `E`. Since identical completion tokens received conflicting synthetic rewards within the first group, the prespecified response-diversity gate failed. The runner stopped without resampling; no arm ran, no gradient or parameter update was measured, and the central integration hypothesis remains untested by this experiment.

The model loaded on CPU in 0.80 seconds. Recorded peak RSS was 3,453,927,424 bytes and elapsed time through gate failure was 1.47 seconds. These are setup observations, not evidence of training feasibility or efficacy.

The first invocation mistakenly used the repository's `.venv/bin/python`, which lacks PyTorch. It stopped before model import or sampling and created no retained measurements. The valid invocation used the host Python 3.12.12 environment with PyTorch 2.9.1 and Transformers 4.57.3. This environment correction is retained in `results/rvl-grpo-group-boundary-real-trainer-v1/run-1/preflight-attempt.json`.

## Audit and limitations

The committed protocol pins the runner and auditor digests. During post-run review, the success-path auditor's expected prompt fields were found not to match the runner's response record schema. The frozen auditor was therefore not represented as having passed; its issue is corrected only in any later protocol version. The failed run is not repaired or reinterpreted.

The result is a valid failure of the frozen gate and identifies two setup weaknesses: the response condition was too restrictive for the sampled model output, and the runner/auditor record contract was not checked end-to-end before preregistration. It says nothing about group-normalization correctness, update direction, verifier reliability, sample efficiency, task success, or model capability.

## Decision

**Do not claim a research result.** Keep this run immutable. A separately versioned protocol may use a prompt that elicits longer, naturally distinct responses and must align/audit the complete record schema before execution. It must define one fixed prompt, seed, reward assignment, and no-resampling rule before any new sample is generated. If that gate fails, stop and retain the failure.

## Reproduction

The raw run is at `results/rvl-grpo-group-boundary-real-trainer-v1/run-1/summary.json` and `progress.json`. Its protocol is `protocols/rvl_grpo_group_boundary_real_trainer_v1.lock.json`; runner and auditor are pinned in commit `9f59354`.
