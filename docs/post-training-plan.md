# Post-training plan under limited compute

## Research question

Can a preference-optimization pipeline produce the expected policy change on a small controlled problem while preserving independent evaluation, and can its diagnostics detect when noisy or shifted preferences make that change unreliable?

This is an experiment-design and measurement question. A synthetic policy is not a language model and cannot support claims about reasoning, truthfulness, broad alignment, or production behavior.

## Why this direction

Post-training depends on reward and preference signals, optimization objectives, policy/reference versions, and independent evaluation. VARE already contains useful supporting pieces: replay provenance/freshness controls, promotion checks, calibrated preference-adjacent trainer tasks, and an RVL integration contract. Those contracts have not produced a measured learner update. The central missing evidence is therefore an executed learning experiment with verifiable math and held-out outcomes.

## Phase 1: synthetic preference control (CPU, no external dependencies)

The first executable version is frozen as [`synthetic_dpo_cpu_v1`](../protocols/synthetic_dpo_cpu_v1.lock.json) and its result is described in [`synthetic-dpo-report.md`](synthetic-dpo-report.md). It used a contextual linear-softmax policy over four actions and eight features, with synthetic pairwise labels from a known utility function and 10 fixed seeds. The analytical gradient passed central finite differences. Clean-label optimization improved held-out NLL, but exceeded the frozen KL ceiling, so the acceptance rule failed. The reference-policy accuracy field also exposed a tie-scoring bug; that field is invalid and a regression test now covers the correction.

Do not alter v1. The v2 follow-up uses a new seed cohort, corrected half-credit tie scoring, and a budget chosen from training-only diagnostics before confirmation outcomes. Its protocol and raw bundle are retained separately.

That follow-up is retained as [`synthetic-dpo-v2-report.md`](synthetic-dpo-v2-report.md). A reported, unretained training-only sweep informed the 100-update budget; a separate development bundle that applies the recorded rule was produced after confirmation and is a reproducibility check. The separate 10-seed confirmation met its recorded rule: all seeds improved held-out synthetic pairwise NLL; mean improvement was 0.3074 nats/pair with paired bootstrap 95% interval [0.2928, 0.3226]; mean KL was 0.3874 under the 0.5 ceiling. The audited bundle reconstructs every seed-by-arm result, and identifies the invalid shuffled-ID arm. Treat this as a narrow synthetic mechanism result only, not independently preregistered evidence.

Required checks and measurements for the next version:

- Compare the analytical gradient with central finite differences at fixed parameter vectors; fail on a declared absolute/relative tolerance breach.
- Include a zero-update/reference-policy control and a known-preference positive control.
- Measure held-out pairwise preference accuracy as the primary outcome; also retain objective values, KL from the reference policy, parameter delta, updates, per-seed outcomes, runtime, and peak memory if available.
- Repeat across multiple deterministic seeds. Report every seed and uncertainty descriptively; do not infer population-level effects from a tiny synthetic task.
- Add separately labeled preference-label noise and a shifted held-out preference distribution. Use these as falsification conditions, not tuning data.
- Retain raw examples or their deterministic generation seeds, exact config, source revision, environment, logs, and hashes.

Predeclared interpretation: the positive control passes only if optimization improves held-out preference NLL over the zero-update reference in the clean condition without violating the frozen KL ceiling. Report held-out accuracy with explicit tie handling. Failure, no change, or seed disagreement is a valid result. A noisy/shifted condition should be interpreted against its own frozen expectation; it is not permissible to adjust thresholds after seeing results.

Before running, freeze a versioned machine-readable protocol specifying the exact synthetic utility, data generator/split, seeds, optimizer and beta, update budget, tolerances, primary metric, KL ceiling, noise/shift conditions, and decision rule. The protocol should be independently auditable and use only the Python standard library. No GPU, paid API, model-weight download, or cloud compute is in scope.

## Phase 2: bounded real-model preference update

An initial feasibility-only check found a cached Qwen2.5-0.5B-Instruct snapshot and a CPU-capable Transformers/PyTorch runtime. A separate, single-pair feasibility probe changed an output-head adapter margin, but it had no held-out data and is not evidence of learning. That probe is not part of the formal result.

The new [`cpu_lm_dpo_head_v1` protocol](../protocols/cpu_lm_dpo_head_v1.lock.json) freezes an offline, CPU-only study before generating any formal outputs. It uses only the exact cached model revision when supplied by the operator, custom rank-4 residual parameters on the two response-label token columns, three initialization seeds, 24 arithmetic-choice training pairs and 32 disjoint held-out pairs per seed, full-batch DPO updates, and a 15-minute/6-GiB abort boundary. It records model-file hashes, prompts and labels, per-example choice margins, adapter parameters, metrics, environment and a manifest. It never downloads weights. The runner is [`run_cpu_lm_dpo_head.py`](../scripts/run_cpu_lm_dpo_head.py).

This study measures conditional preference over two one-token answer labels. Its task and parameterization are deliberately narrow: a frozen backbone and custom output-head slice are not a general PEFT implementation or a standard TRL run. A passing result would be small-model held-out preference evidence only. A non-pass, null, timeout or missing cached dependency is retained without changing the locked rule.

The existing L2 RVL/Qwen protocol remains frozen and unrun. Its declared 0.5B model and 3-arm/3-seed workload are outside the present budget. Do not modify that lock to make the project appear to have run it.

## Current gaps and claim boundary

Until the frozen real-model protocol has actually run and been audited, there is still no retained real-model update or held-out model improvement. v1 remains a visible non-pass with its invalid reference-accuracy metric; v2 is a separate accepted synthetic run and cannot establish language-model behavior. Historical task calibration, grader mutation testing, replay/promotion contracts, and local reliability measurements are supporting engineering evidence. The previous small-model coding-agent pilot is a negative tool-use result and is not post-training evidence.

Phase 1 can establish objective/gradient correctness and sensitivity in a toy controlled system. It cannot establish that VARE improves a language model or produces useful real-world behavior. Phase 2 would provide only a small-model smoke result unless a sufficiently powered, preregistered held-out study is actually completed.
