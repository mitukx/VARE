# Stage 0 research gate — 2026-10-11

## Decision

**Do not start a new training run or request GPU time yet.** The strongest visible mechanism is policy drift while reusing fixed GRPO rollouts, but the general mechanism is now directly covered by recent off-policy GRPO work. The VARE result is a useful local diagnostic, not a distinct method or causal finding. No other candidate survives the novelty and data-feasibility gates. This is a bounded no-go decision, not evidence that the broader research problem is solved.

No files, model weights, evaluation examples, or checkpoints were changed. No GPU, MPS, paid service, or external service was used. The OpenBookQA confirmation cohort remains retired and was not read or rescored in this gate.

## Current evidence and implementation path

The latest completed model comparison is the frozen OpenBookQA/Qwen2.5-0.5B GRPO/SFT study. It used three seeds and 128 confirmation questions: base exact success was 45.31%, matched SFT averaged 33.33%, and GRPO produced 0/128 parseable answers for each seed. The study's independent audit reconstructed the metrics and intervals. Its report explicitly does not identify the cause of collapse and retires that exact pairing; its source/data license also prevents distributing the local raw bundle.

The training script generates one base-policy rollout set, then reuses it across 16 optimizer batches per seed ([runner](../scripts/run_openbookqa_qwen_grpo_sft_v5.py#L344)). RVL reads the stored behavior token log-probabilities, recomputes current-policy token log-probabilities in the trainer, and applies the clipped surrogate ([pinned trainer source at RVL `27ebf7f`](https://github.com/mitukx/Recursive-Verification-Lag/blob/27ebf7fe239d97504eeb960bf1407de220584dac/src/rvl_systems/hf_trainer.py#L70)). This is rollout reuse/off-policy drift; it is **not** reuse of stored parameter-gradient vectors.

A read-only aggregation of the three retained training progress records found:

| Diagnostic | First update mean | Updates 2–16 mean | All 48 updates mean |
| --- | ---: | ---: | ---: |
| Token clip fraction | 0.000 | 0.919 | 0.861 |
| Behavior-KL diagnostic | approximately 0 | 10.045 | 9.417 |
| Maximum absolute log-ratio | less than 0.0001 | 19.660 | 18.432 |

The update-2–16 log-ratio diagnostic reached its configured cap of 20 in 41 of 45 updates. These values independently reproduce the aggregate diagnostics in the confirmation report. They establish rapid divergence on this run. They do not establish that staleness caused the invalid outputs: learning rate, full-parameter AdamW, fixed rollouts, token-level clipping, tiny training set, and the output contract were not isolated in a causal comparison.

## Candidate screen

Scores are qualitative Stage 0 judgments, not measured study outcomes.

| Candidate | Research question and VARE evidence | Novelty / impact / CPU feasibility | Decision |
| --- | --- | --- | --- |
| Fixed-rollout reuse and format/task collapse | Does update age cause the OpenBookQA collapse, independently of optimizer step size and model drift? The observed clipping change is large, but comes from one retired model/task/prompt pairing. | High practical impact; low novelty after [Mu-GRPO](https://arxiv.org/abs/2605.17570), [Scaling Laws for Collapse in Asynchronous GRPO](https://arxiv.org/abs/2607.01083), and [PNPO](https://arxiv.org/abs/2608.01418), which study stale rollout reuse, stability/collapse, and off-policy correction directly. CPU inference/training is possible for small models, but no unused task/base pair has passed a reward-and-format gate. | **Do not select for a new-method claim.** A fresh low-resource replication might add boundary evidence, but needs a genuinely new task and cannot use the retired confirmation cohort. |
| Verifier error under policy-induced shift | When does a verifier's conditional false-accept law change enough that audited correction stops protecting the true task objective? VARE already has policy-shift and higher-order audit studies; they either reduce to standard propensity correction or remain synthetic identification results. | Very high impact; very low novelty without a distinct estimand. The recent [Verifier Errors in RLVR](https://arxiv.org/abs/2609.35677) already gives fixed-verifier gradient-flow conditions, an identifiability limit, and an audit-based selective correction with neural and language-model experiments. A credible extension needs a new real-model, independently labeled dataset and is not CPU-gated today. | **Do not select.** No distinct claim has survived comparison. |
| Position-balanced preference optimization | Can counterbalanced training/evaluation turn preference loss improvements into semantic task gains? VARE's GSM8K/ASDiv evidence found lower pairwise NLL while answer-content margin did not improve; that line is retired. | Moderate impact; low novelty because counterbalancing is a standard control and the existing VARE result already exposes the metric confound. CPU inference is feasible, but no fresh model/task/update protocol is ready and reusing archived cohorts would be invalid. | **Do not reopen.** A fresh task and new hypothesis would be required. |

## Selected hypothesis and minimum decisive experiment

No candidate is selected for Stage 1. The strongest conditional hypothesis is:

> On a fresh, independently graded task where the base model has a nonzero parse rate and mixed rewards, reusing each rollout group for additional learner updates increases policy drift and output-contract failure relative to refreshing rollouts, at equal optimizer-update and total generated-token budgets.

This is falsifiable, but the broad stale-data mechanism is not novel by itself. The observation in VARE is confounded and comes from a retired task, so it cannot be used as a success claim or to tune an intervention.

The minimum decisive comparison, **if a fresh task/base pair is later justified**, is: freeze a new train/dev/confirmation split and checker; measure the unchanged model; then compare unchanged base, matched SFT, and GRPO with rollout refresh every update against GRPO reusing the same rollouts for a prespecified number of updates. Match optimizer steps, per-update batch size, prompt distribution, seeds, and token processing in the learner as closely as possible. Rollout generation cost cannot also be matched: the fresh arm must generate new completions, while the reuse arm does not. Record generated tokens and rollout wall time separately, and report the task-success-versus-generation-cost trade-off rather than hiding that difference. The primary metric is held-out executable task success; secondary measures are parse failure, false verifier acceptance, token clip fraction, KL, wall time, and peak memory. Failure to show a task-success difference while only KL/clipping changes rejects the capability claim. Any protocol would need to pin exact revisions, freeze criteria before training, and leave confirmation examples untouched.

This experiment is **not ready to run**: every currently attempted local task family is either retired, consumed, or failed its base interaction/reward gate. The missing prerequisite is a fresh task contract with a validated independent checker and an unused confirmation composition—not another prompt tweak to an old cohort.

## Feasibility and compute

Current host snapshot: Apple arm64, 32 GiB RAM, Python 3.9.6 with PyTorch 2.8.0; CUDA unavailable. MPS is present but is treated as unavailable under the zero-GPU rule. A previous CPU-only six-arm Qwen2.5-0.5B comparison measured 244.7 s for its base gate, 1,067.8 s for training, and 688.6 s for evaluation, with 11.7 GiB peak RSS during training. These measurements are a local reference only; they do not estimate another dataset or model reliably.

No GPU authorization is requested: the candidate has not passed the novelty/data gate, so a GPU smoke would not answer a defensible research question. Colab availability, GPU type, session duration, and Compute Unit cost were not inspected or assumed.

## Work completed

- Re-read `AGENTS.md`, the latest evidence gate, OpenBookQA report and lock, model-study stop decisions, GRPO trainer source, and existing policy-shift/staleness reports.
- Inspected the actual frozen-rollout-to-current-logprob path in the OpenBookQA runner and pinned RVL trainer.
- Independently re-aggregated the 48 retained training-update records without reading confirmation responses; the values above match the retained summary.
- Compared three candidate questions with recent primary papers; none warrants a new training or GPU run.
- No tests were run because no implementation changed. This report records a Stage 0 decision; it is not a research result, a causal finding, or evidence of model improvement.

## References

- Mu-GRPO, [arXiv:2605.17570](https://arxiv.org/abs/2605.17570).
- Scaling Laws for Collapse in Asynchronous GRPO, [arXiv:2607.01083](https://arxiv.org/abs/2607.01083).
- Prefix-Normalized Policy Optimization, [arXiv:2608.01418](https://arxiv.org/abs/2608.01418).
- Verifier Errors in RLVR, [arXiv:2609.35677](https://arxiv.org/abs/2609.35677).
