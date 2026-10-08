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


The frozen [`vare-synthetic-preference-robustness-v1`](../protocols/synthetic_preference_robustness_v1.lock.json) study has completed. Its source-snapshotting runner and auditor regenerate every design and label and reconstruct all reported metrics. The clean arm passed its base-teacher held-out NLL/KL rule on all ten seeds; the 20% and 40% flip arms had worse mean NLL on every seed, and the clean-trained policy scored worse on average under the declared shifted teacher. See the [report and exact limits](synthetic-preference-robustness-v1-report.md) and [retained bundle](../results/synthetic-preference-robustness-v1/confirmation/). This narrows the synthetic measurement gap, but it does not add model-level evidence.

## Phase 2: bounded real-model preference update

An initial feasibility-only check found a cached Qwen2.5-0.5B-Instruct snapshot and a CPU-capable Transformers/PyTorch runtime. A separate, single-pair feasibility probe changed an output-head adapter margin, but it had no held-out data and is not evidence of learning. That probe is not part of the formal result.

The new [`cpu_lm_dpo_head_v1` protocol](../protocols/cpu_lm_dpo_head_v1.lock.json) froze an offline, CPU-only study before generating formal outputs. It used only the exact cached model revision, custom rank-4 residual parameters on the two response-label token columns, three initialization seeds, 24 arithmetic-choice training pairs and 32 disjoint held-out pairs per seed, full-batch DPO updates, and a 15-minute/6-GiB abort boundary. It records model-file hashes, prompts and labels, per-example choice margins, adapter parameters, metrics, environment and a manifest. It never downloaded weights. The run completed but failed its predeclared held-out NLL and KL rule; see the [v1 report](cpu-lm-dpo-head-v1-report.md).

This study measures conditional preference over two one-token answer labels. Its task and parameterization are deliberately narrow: a frozen backbone and custom output-head slice are not a general PEFT implementation or a standard TRL run. A passing result would be small-model held-out preference evidence only. A non-pass, null, timeout or missing cached dependency is retained without changing the locked rule.

The existing L2 RVL/Qwen protocol remains frozen and unrun. Its declared 0.5B model and 3-arm/3-seed workload are outside the present budget. Do not modify that lock to make the project appear to have run it.

## Current gaps and next phase

The separate [`cpu_lm_gsm8k_dpo_confirmation_v1`](../protocols/cpu_lm_gsm8k_dpo_confirmation_v1.lock.json) study completed after a train-only update-budget selection. Across three new seeds and all 1,319 public GSM8K test questions, mean conditional preference NLL changed by −0.00697 nats/question (paired 95% interval [−0.00800, −0.00591]); all seeds improved and KL stayed far below the 0.5 ceiling. Accuracy remained near chance (0.4943 → 0.4956). The study uses verifier-generated answer choices and updates only two output columns, so it does not demonstrate free-form generation, general reasoning, human preference alignment, or capability gain. The prior arithmetic-choice model run remains a separate non-pass; the two results are not pooled. See the [report](cpu-lm-gsm8k-dpo-confirmation-v1-report.md).

Subsequent full-sequence DPO development has not yet established free-form improvement. The initial numeric answer setup was weakly aligned with generation; a rationale-versus-base-rollout setup had improved verifier-preference NLL but zero exact-match for both base and updates on 16 questions. A longer generation budget breached the 6-GiB memory cap before metrics. All those cohorts are consumed and retained separately. A candidate confirmation must use fresh rows, a nonzero baseline, a strict minimum exact-match gain, the KL cap and multi-seed consistency; verifier-generated comparisons remain distinct from human preference evidence.

The follow-up rationale run fit verifier preferences at lower KL but every base and updated generation was cut off before the requested answer marker under the CPU memory budget. The current next probe returns to numeric-only responses for both chosen and rejected sequences, with actual base-model outputs providing rejected examples. It uses a fresh cohort and a non-vacuous pass gate; even a development pass will require a separately locked, larger confirmation.

The larger rank-8 numeric study confirmed that preference NLL is not a reliable selection target for task success on this setup: epoch 2 lowered NLL but reduced exact-match, while epoch 4 breached KL. The next development protocol compares every eligible epoch on fresh verifier-checked exact-match and uses that same held-out development cohort only for selection. Confirmation remains separate and untouched.

The follow-up implemented exact-match selection over all NLL/KL-eligible checkpoints. It selected epoch 2, but base exact-match was 0/64 and updated scores were 2/64, 0/64, and 2/64; the nonzero-baseline and minimum-gain gates failed. Next, repeat this selection on a larger fresh validation cohort with a lower learning rate and rank-16 adapter. A development pass must still be confirmed on new rows.

The larger rank-16 run selected epoch 1 at 9/128, 8/128, and 9/128 versus base 9/128. Later epochs continued reducing verifier-preference NLL while Exact Match decreased. The next experiment is a matched SFT control: use the same verified chosen completions, rows, initializations and update counts to determine whether pairwise DPO provides value over supervised answer imitation.

The matched DPO/SFT study found DPO higher on verifier-pair accuracy, but lower on exact-match than SFT and with worse preference NLL. SFT preserved the base score; neither arm improved. A new SFT-anchored DPO development study can test whether auxiliary answer likelihood limits task regression while keeping the rejected-response signal, with fresh rows and matched DPO/SFT controls.

The sequence-level development candidate's four-epoch setting was confirmed on 128 new held-out questions. Preference NLL improved with a paired 95% interval below zero and exact-match increased in each seed, but mean KL was 0.737 against a 0.5 ceiling; see the [confirmation v2 non-pass](cpu-lm-gsm8k-sequence-dpo-confirmation-v2-report.md). A fresh lower-rate development v3 kept KL at 0.252 but exact-match fell from 1.56% to 1.04%; see its [non-pass report](cpu-lm-gsm8k-sequence-dpo-development-v3-report.md). The next cycle must change the development task/completion setup using new rows before another confirmation. Confirmation v1 was incomplete at its two-hour CPU limit, and v2 held-out rows are consumed. General answer quality remains unproven. The larger frozen L2 trainer-integrated campaign remains unrun; neither cached-model study substitutes for it.
