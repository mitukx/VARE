# Post-training plan under limited compute

## Research question

Can a preference-optimization pipeline produce the expected policy change on a small controlled problem while preserving independent evaluation, and can its diagnostics detect when noisy or shifted preferences make that change unreliable?

This is an experiment-design and measurement question. A synthetic policy is not a language model and cannot support claims about reasoning, truthfulness, broad alignment, or production behavior.

## Why this direction

Post-training depends on reward and preference signals, optimization objectives, policy/reference versions, and independent evaluation. VARE already contains useful supporting pieces: replay provenance/freshness controls, promotion checks, calibrated preference-adjacent trainer tasks, and an RVL integration contract. Those contracts have not produced a measured learner update. The central missing evidence is therefore an executed learning experiment with verifiable math and held-out outcomes.

## Phase 1: synthetic preference control (CPU, no external dependencies)

The first executable version is frozen as [`synthetic_dpo_cpu_v1`](../protocols/synthetic_dpo_cpu_v1.lock.json) and its result is described in [`synthetic-dpo-report.md`](synthetic-dpo-report.md). It used a contextual linear-softmax policy over four actions and eight features, with synthetic pairwise labels from a known utility function and 10 fixed seeds. The analytical gradient passed central finite differences. Clean-label optimization improved held-out NLL, but exceeded the frozen KL ceiling, so the acceptance rule failed. The reference-policy accuracy field also exposed a tie-scoring bug; that field is invalid and a regression test now covers the correction.

Do not alter v1. The next run must have a new versioned protocol and a new seed cohort. Select any lower update budget using only training-set objective and KL diagnostics in a separately labeled development run. Fix tie handling (a zero-logit reference gets 0.5 pairwise accuracy), freeze the next protocol before its confirmatory outcomes, and retain its full raw bundle.

Required checks and measurements for the next version:

- Compare the analytical gradient with central finite differences at fixed parameter vectors; fail on a declared absolute/relative tolerance breach.
- Include a zero-update/reference-policy control and a known-preference positive control.
- Measure held-out pairwise preference accuracy as the primary outcome; also retain objective values, KL from the reference policy, parameter delta, updates, per-seed outcomes, runtime, and peak memory if available.
- Repeat across multiple deterministic seeds. Report every seed and uncertainty descriptively; do not infer population-level effects from a tiny synthetic task.
- Add separately labeled preference-label noise and a shifted held-out preference distribution. Use these as falsification conditions, not tuning data.
- Retain raw examples or their deterministic generation seeds, exact config, source revision, environment, logs, and hashes.

Predeclared interpretation: the positive control passes only if optimization improves held-out preference NLL over the zero-update reference in the clean condition without violating the frozen KL ceiling. Report held-out accuracy with explicit tie handling. Failure, no change, or seed disagreement is a valid result. A noisy/shifted condition should be interpreted against its own frozen expectation; it is not permissible to adjust thresholds after seeing results.

Before running, freeze a versioned machine-readable protocol specifying the exact synthetic utility, data generator/split, seeds, optimizer and beta, update budget, tolerances, primary metric, KL ceiling, noise/shift conditions, and decision rule. The protocol should be independently auditable and use only the Python standard library. No GPU, paid API, model-weight download, or cloud compute is in scope.

## Phase 2: optional tiny real-model smoke test

Only consider this after Phase 1 and only if already-cached weights, installed dependencies, and local CPU capacity make a bounded run practical. First benchmark loading and one forward/backward step on a tiny sample. Set hard limits for elapsed time and memory, and stop immediately if either is exceeded. Do not download weights or install dependencies that trigger large model/artifact downloads.

If a smoke test is feasible, freeze a separate protocol and report the actual base model/revision, tokenizer, data, trainable parameter count, before/after adapter hashes, objective, update count, held-out preference result, KL/drift, runtime, memory, and all failures. A smoke test with no independent held-out gain remains a smoke test; it does not inherit the synthetic result's interpretation.

The existing L2 RVL/Qwen protocol remains frozen and unrun. Its declared 0.5B model and 3-arm/3-seed workload are outside the present budget. Do not modify that lock to make the project appear to have run it.

## Current gaps and claim boundary

There is no real-model training update or held-out model improvement. The v1 synthetic policy changed and improved its held-out synthetic NLL, but failed acceptance due to policy drift and has an invalid reference accuracy field. Existing historical-task calibration, grader mutation testing, replay/promotion contracts, and local reliability measurements are supporting engineering evidence. The previous small-model coding-agent pilot is a negative tool-use result and is not post-training evidence.

Phase 1 can establish objective/gradient correctness and sensitivity in a toy controlled system. It cannot establish that VARE improves a language model or produces useful real-world behavior. Phase 2 would provide only a small-model smoke result unless a sufficiently powered, preregistered held-out study is actually completed.
